# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import aiohttp
import logging
import re
import hashlib
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

def get_request_hash(request_str: str) -> str:
    """Generates a SHA256 hash for a given request string."""
    return hashlib.sha256(request_str.encode('utf-8')).hexdigest()


class AsyncOpenAIWorker:
    """
    异步 OpenAI API 调用器，支持真正的异步 I/O 并行处理。
    允许多个请求并发执行，不阻塞主线程。
    """

    def __init__(self, api_key: str, model_name: str, api_endpoint: str,
                 system_prompt: str, reward_coefficient: float = 1.0):
        self.api_key = api_key
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.system_prompt = system_prompt
        self.reward_coefficient = reward_coefficient
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_cache = {}  # 缓存已处理的结果

        # 预配置的请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def _call_openai_api(self, request_str: str, request_hash: str) -> float:
        """
        异步调用 OpenAI API
        """
        # 检查缓存
        if request_hash in self._request_cache:
            return self._request_cache[request_hash]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request_str},
            ],
            "temperature": 0.0,
        }

        try:
            async with self.session.post(self.api_endpoint, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()

                model_output = response_json["choices"][0]["message"]["content"]

                # 使用正则表达式提取评分
                match = re.search(r'\d+\.?\d*', model_output)
                if match:
                    score = float(match.group(1))
                    result = score * self.reward_coefficient
                    self._request_cache[request_hash] = result
                    return result
                else:
                    logger.warning(f"Could not parse OpenAI score from: {model_output}")
                    result = 0.0
                    self._request_cache[request_hash] = result
                    return result

        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API request failed for hash {request_hash}: {e}")
            result = 0.0
            self._request_cache[request_hash] = result
            return result
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenAI API response for hash {request_hash}: {e}")
            result = 0.0
            self._request_cache[request_hash] = result
            return result

    async def evaluate_single(self, response_str: str) -> float:
        """
        评估单个响应
        """
        request_hash = get_request_hash(response_str)
        return await self._call_openai_api(response_str, request_hash)

    async def evaluate_batch(self, responses: List[str]) -> List[float]:
        """
        批量评估响应 - 真正的异步并行执行
        使用 asyncio.gather 实现并发请求，不阻塞主线程
        """
        if not responses:
            return []

        # 创建任务列表
        tasks = []
        request_hashes = []

        for response_str in responses:
            request_hash = get_request_hash(response_str)
            request_hashes.append(request_hash)

            # 创建异步任务
            task = self._call_openai_api(response_str, request_hash)
            tasks.append(task)

        # 并发执行所有任务
        logger.info(f"Starting {len(tasks)} concurrent OpenAI API requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task for request {request_hashes[i]} failed: {result}")
                final_results.append(0.0)
            else:
                final_results.append(result)

        logger.info(f"Completed {len(tasks)} concurrent OpenAI API requests")
        return final_results


class AsyncOpenAIManager:
    """
    异步 OpenAI 任务管理器 - 负责协调 GPU 计算和 API 调用的真正并行
    """

    def __init__(self, api_key: str, model_name: str, api_endpoint: str,
                 system_prompt: str, reward_coefficient: float = 1.0,
                 max_concurrent: int = 10):
        self.api_key = api_key
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.system_prompt = system_prompt
        self.reward_coefficient = reward_coefficient
        self.max_concurrent = max_concurrent
        self.worker: Optional[AsyncOpenAIWorker] = None

        # 结果缓存和任务状态
        self.pending_tasks: Dict[str, asyncio.Task] = {}
        self.results_cache: Dict[str, float] = {}
        self._session_lock = asyncio.Lock()

    async def initialize(self):
        """初始化异步工作器"""
        async with self._session_lock:
            if not self.worker:
                self.worker = AsyncOpenAIWorker(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    api_endpoint=self.api_endpoint,
                    system_prompt=self.system_prompt,
                    reward_coefficient=self.reward_coefficient
                )
                await self.worker.__aenter__()

    async def shutdown(self):
        """关闭异步工作器"""
        async with self._session_lock:
            if self.worker:
                await self.worker.__aexit__(None, None, None)
                self.worker = None

    async def submit_and_get_batch_results(
        self,
        responses_to_evaluate: List[tuple[str, float]],
        check_interval: float = 0.01
    ) -> List[float]:
        """
        提交任务并返回结果 - 非阻塞方式

        这个方法是真正的关键：
        1. 立即提交所有任务到异步执行
        2. 返回 Future 对象，主线程可以继续 GPU 计算
        3. 通过定期检查收集结果，不阻塞 GPU 计算

        Args:
            responses_to_evaluate: [(response_str, format_reward), ...]
            check_interval: 检查间隔时间（秒）

        Returns:
            Future 对象，用于获取最终结果
        """
        # 过滤掉格式奖励为0的响应
        filtered_responses = []
        filtered_indices = []

        for i, (response_str, format_reward) in enumerate(responses_to_evaluate):
            if format_reward > 0:
                filtered_responses.append(response_str)
                filtered_indices.append(i)

        if not filtered_responses:
            # 如果没有需要评估的响应，直接返回
            return [0.0] * len(responses_to_evaluate)

        await self.initialize()

        # 创建异步任务
        tasks = []
        request_hashes = []

        for response_str in filtered_responses:
            request_hash = get_request_hash(response_str)
            request_hashes.append(request_hash)

            # 检查是否已有缓存结果
            if request_hash in self.results_cache:
                # 如果有缓存，跳过创建任务
                tasks.append(None)
            else:
                # 创建新的异步任务
                task = asyncio.create_task(
                    self.worker._call_openai_api(response_str, request_hash),
                    name=f"openai_eval_{request_hash[:8]}"
                )
                tasks.append(task)
                self.pending_tasks[request_hash] = task

        # 创建协程来收集结果（不阻塞）
        async def collect_results():
            # 等待所有任务完成
            for i, (task, request_hash) in enumerate(zip(tasks, request_hashes)):
                if task is not None:
                    try:
                        result = await task
                        self.results_cache[request_hash] = result
                        self.pending_tasks.pop(request_hash, None)
                    except Exception as e:
                        logger.error(f"Task failed for {request_hash}: {e}")
                        self.results_cache[request_hash] = 0.0
                        self.pending_tasks.pop(request_hash, None)

            # 构建完整结果
            full_results = [0.0] * len(responses_to_evaluate)
            for idx, (request_hash, original_idx) in enumerate(zip(request_hashes, filtered_indices)):
                full_results[original_idx] = self.results_cache.get(request_hash, 0.0)

            return full_results

        # 返回协程对象，主线程可以继续执行 GPU 计算
        return await collect_results()

    async def wait_for_results_with_callback(
        self,
        future: asyncio.Future,
        progress_callback=None,
        timeout: float = 300.0
    ) -> List[float]:
        """
        等待结果，支持进度回调和超时

        Args:
            future: 协程返回的Future对象
            progress_callback: 进度回调函数
            timeout: 超时时间（秒）

        Returns:
            最终结果列表
        """
        try:
            start_time = asyncio.get_event_loop().time()
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API evaluation timed out after {timeout} seconds")
            raise
