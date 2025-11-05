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

from collections import defaultdict
import torch
import numpy as np
import logging
import requests
import re
import time
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from .async_openai_worker import AsyncOpenAIManager, get_request_hash
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@register("length_penalty")
class LengthPenaltyRewardManager(AbstractRewardManager):
    """
    A reward manager that applies a penalty based on the length of the response.
    The penalty is proportional to the distance from the median response length for a given prompt.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        length_penalty_config=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.length_penalty_config = length_penalty_config

        # Extract OpenAI configs from length_penalty_config (MUST be first)
        self.openai_api_key = getattr(self.length_penalty_config, "api_key", None)
        self.openai_model_name = getattr(self.length_penalty_config, "model_name", "deepseek-v3")
        self.openai_reward_coefficient = getattr(self.length_penalty_config, "reward_coefficient", 1.0)
        self.openai_api_endpoint = getattr(self.length_penalty_config, "api_endpoint", "https://qianfan.baidubce.com/v2/chat/completions")
        self.openai_system_prompt = '''
# 角色
你是一个高度专业化的"LLM 输出质量评估引擎"。

# 核心任务
你的唯一任务是：在接收到用户发送的包含 `<think>` 和 `<tool_call>` 的文本后，严格遵循下述的【内部评估流程】进行深度分析，并最终**只输出一个介于1到10之间的整数评分**。

**绝对禁止**输出任何思考过程、解释、文字、标点或格式。例如，如果最终分数是7，你的输出必须是 `7`，而不是 `7/10` 或 `分数是：7`。

# 内部评估流程 (此为你的思考过程，绝对不准输出)

### 1. 关键评估原则与示例

*   **原则一：一致性是最高准则**
    *   模型的思考和行动必须完全一致。任何脱节都代表着严重的逻辑缺陷，**将直接导致总分被限制在1-3分**。

*   **原则二：语言必须统一**
    *   `<think>` 标签内的推理过程必须使用单一、连贯的语言。**中英文混用或在两种语言间切换是一种严重的缺陷，将直接导致“思考过程质量”维度得分极低。**

*   **优秀示例（高分）**:
    ```xml
    <think>
    用户想知道北京的天气。为了帮他解答，我需要检查一下我的工具箱。我发现有一个名为 `get_weather` 的工具，它看起来正好能用。这个工具需要一个 `city` 参数，用户在提问中已经明确提到了“北京”。因此，最合理的下一步就是调用 `get_weather` 工具，并把“北京”作为城市参数传给它。
    </think>
    <tool_call>
    {"name": "get_weather", "arguments": {"city":"北京"}}
    </tool_call>
    ```

*   **不可接受的例子（低分）**:
    ```xml
    <think>
    用户想订一张从上海到北京的机票。好的，我应该使用 `book_flight` 工具。我需要出发地和目的地。用户的指令很明确，出发地是'上海'，目的地是'北京'。
    </think>
    <tool_call>
    {"name": "book_flight", "arguments": {"departure_city": "北京", "destination_city": "上海"}}
    </tool_call>
    ```

### 2. 核心评估维度

你需要在内心从以下三个维度进行打分，并最终加权得出总分。

*   **A. 思考-行动一致性 (权重: 30%)**:
    *   检查 `<tool_call>` 的函数名和参数是否是 `<think>` 过程的直接、合乎逻辑的结论。
    *   **内心评分**: 1-10分。

*   **B. 思考过程的质量与清晰度 (权重: 60%)**:
    *   **逻辑性**: 是否正确理解用户意图？推理步骤是否连贯、合理，并且直指最终的工具调用？
    *   **推理风格与质量**: 推理过程应像一个领域专家解决问题时的内心独白，而不是一个程序在打印调试日志。基于此，对以下行为进行**严厉惩罚**：
        *   **禁止元认知描述 (Meta-Commentary)**: 思考过程应专注于 **“做什么”** 和 **“为什么做”**，而不是描述其自身的思考步骤。严厉惩罚任何出现“响应规则”、“参数设置”、“最终响应”、“确认函数调用”、“响应格式”等描述生成过程的词语。
        *   **禁止模板化与冗余**: 推理应自然、直截了当。严厉惩罚使用“回顾工具描述”、“检查调用规范”等机械短语，以及对同一结论的反复确认。**尤其禁止在思考的结尾处复述最终的`tool_call`内容。**
    *   **完整性与正确性**: 思考过程的文本必须是完整的句子，**没有中途截断**。**不得包含任何拼写错误或明显的语法错误**。
    *   **语言纯粹性**: **是否全程使用单一语言？出现中英混用或切换则此项得分极低。**
    *   **内心评分**: 1-10分。

*   **C. 工具调用有效性 (权重: 10%)**:
    *   检查 `<tool_call>` 本身的JSON格式是否正确，函数名和参数名是否存在拼写错误，参数值是否符合常识和逻辑。
    *   **内心评分**: 1-10分。

### 3. 计算最终分数

*   在内心计算加权总分：`总分 = (A * 0.3) + (B * 0.6) + (C * 0.1)`。
*   将计算出的总分进行四舍五入，得到最终的整数。

# 输出规则 (必须无条件遵守)
-   你的最终响应**必须且只能是**一个阿拉伯数字（1, 2, 3, 4, 5, 6, 7, 8, 9, 10）。
-   **不包含**任何前缀或后缀。
-   **不包含**任何文字解释。
-   **不包含**任何多余的空格或换行。

# 工作流程
1.  在我发送此条指令后，**不要回复任何确认信息**，直接进入待命状态。
2.  当我发送需要评估的文本后，你将立即执行【内部评估流程】。
3.  完成评估和计算后，立即输出那个最终的整数。
'''

        # Initialize OpenAI manager - SIMPLIFIED VERSION (only async)
        self.use_async_io = getattr(self.length_penalty_config, "use_async_io", False)

        self.async_openai_manager = None
        if self.use_async_io and self.openai_api_key:
            self.async_openai_manager = AsyncOpenAIManager(
                api_key=self.openai_api_key,
                model_name=self.openai_model_name,
                api_endpoint=self.openai_api_endpoint,
                system_prompt=self.openai_system_prompt,
                reward_coefficient=self.openai_reward_coefficient,
                max_concurrent=getattr(self.length_penalty_config, "max_concurrent_requests", 10)
            )
            logger.info("✅ Initialized async OpenAI manager for TRUE ASYNC I/O (zero GPU wait)")
        elif self.openai_api_key:
            logger.info("ℹ️  OpenAI API key provided but use_async_io=False. Using synchronous fallback.")

        # Event loop for async operations
        self._event_loop = None
        self._executor = None

    def start_workers(self):
        """启动异步 OpenAI manager"""
        if self.async_openai_manager:
            logger.info("Async OpenAI manager is initialized and ready (no separate start needed)")
        else:
            logger.info("No async OpenAI manager to start (use_async_io=True to enable)")

    def shutdown_workers(self):
        """关闭异步 OpenAI 管理器"""
        # 关闭异步 OpenAI manager
        if self.async_openai_manager:
            logger.info("Shutting down async OpenAI manager...")
            if self._event_loop and self._event_loop.is_running():
                # 在事件循环中关闭异步管理器
                future = asyncio.run_coroutine_threadsafe(
                    self.async_openai_manager.shutdown(),
                    self._event_loop
                )
                try:
                    future.result(timeout=5.0)
                except Exception as e:
                    logger.error(f"Error shutting down async manager: {e}")

            # 关闭事件循环和线程池
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()
                self._event_loop = None

            logger.info("Async OpenAI manager shut down successfully.")
        else:
            logger.info("No OpenAI manager to shut down (not initialized)")

    def _ensure_event_loop(self):
        """确保事件循环正在运行"""
        if self._event_loop is None or self._event_loop.is_closed():
            logger.info("Starting new event loop for async operations...")
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-openai")
            self._event_loop = asyncio.new_event_loop()

            # 在单独线程中启动事件循环
            def run_event_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._event_loop_thread = threading.Thread(
                target=run_event_loop,
                args=(self._event_loop,),
                daemon=True
            )
            self._event_loop_thread.start()
            logger.info("Event loop started in background thread")

    def _get_single_openai_quality_reward(self, response_str: str, response_format_reward: float) -> float:
        if response_format_reward == 0:
            return 0.0

        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Skipping OpenAI quality evaluation.")
            return 0.0

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model_name,
            "messages": [
                {"role": "system", "content": self.openai_system_prompt},
                {"role": "user", "content": response_str},
            ],
            "temperature": 0.0, # For consistent evaluation
        }

        try:
            response = requests.post(self.openai_api_endpoint, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            response_json = response.json()
            
            # Extract content from the response
            model_output = response_json["choices"][0]["message"]["content"]
            
            # Parse the score using regex
            match = re.search(r'最终评分\s+(\d+)', model_output)
            if match:
                score = float(match.group(1))
                return score * self.openai_reward_coefficient
            else:
                logger.warning(f"Could not parse OpenAI score from: {model_output}")
                return 0.0
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            return 0.0
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenAI API response: {e}\nResponse: {response_json}")
            return 0.0

    def _get_batched_openai_quality_rewards_sync(self, responses_to_evaluate: list[tuple[str, float]]) -> list[float]:
        """
        同步版本的 OpenAI 质量评估，作为备选方案。
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Skipping OpenAI quality evaluation.")
            return [0.0] * len(responses_to_evaluate)

        import concurrent.futures

        max_workers = 5
        scores = [0.0] * len(responses_to_evaluate)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._get_single_openai_quality_reward, resp_str, resp_format_reward): i
                for i, (resp_str, resp_format_reward) in enumerate(responses_to_evaluate)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    score = future.result()
                    scores[index] = score
                except Exception as exc:
                    logger.error(f"OpenAI quality evaluation generated an exception for response at index {index}: {exc}")
                    scores[index] = 0.0

        return scores

    def _get_batched_openai_quality_rewards_async(self, responses_to_evaluate: list[tuple[str, float]]) -> list[float]:
        """
        真正的异步并行 OpenAI 质量评估 - 零 GPU 等待时间

        这个方法的核心优势：
        1. 立即返回 Future 对象，不阻塞主线程
        2. GPU 可以继续进行其他计算
        3. API 请求在后台异步并发执行
        4. 只在必要时检查结果，最大化 GPU 利用率
        """
        if not self.async_openai_manager:
            logger.warning("Async OpenAI manager not available")
            return [0.0] * len(responses_to_evaluate)

        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Skipping OpenAI quality evaluation.")
            return [0.0] * len(responses_to_evaluate)

        # 确保事件循环正在运行
        self._ensure_event_loop()

        # 提交所有异步任务
        async def submit_and_evaluate():
            results = await self.async_openai_manager.submit_and_get_batch_results(
                responses_to_evaluate,
                check_interval=0.001  # 最小检查间隔
            )
            return results

        # 在后台线程中执行异步操作
        future = asyncio.run_coroutine_threadsafe(submit_and_evaluate(), self._event_loop)

        # 等待结果，但主线程可以做其他事情
        try:
            # 设置超时但可以调整
            timeout = 300  # 5分钟超时
            results = future.result(timeout=timeout)
            logger.info(f"Completed async OpenAI evaluation for {len(responses_to_evaluate)} responses")
            return results
        except asyncio.TimeoutError:
            logger.error(f"Async OpenAI evaluation timed out after {timeout} seconds")
            return [0.0] * len(responses_to_evaluate)
        except Exception as e:
            logger.error(f"Async OpenAI evaluation failed: {e}")
            return [0.0] * len(responses_to_evaluate)

    def _get_batched_openai_quality_rewards_non_blocking(
        self,
        responses_to_evaluate: list[tuple[str, float]]
    ) -> tuple[list[float], asyncio.Future]:
        """
        非阻塞版本的异步评估 - 返回 Future 和初始结果

        这个方法是关键优化：
        1. 立即返回初始结果（0.0）
        2. 返回 Future 对象供后续检查
        3. 主线程可以立即继续 GPU 计算
        4. 在计算间隙异步检查 API 结果

        Returns:
            (initial_results, future) - 初始结果和异步Future
        """
        if not self.async_openai_manager or not self.openai_api_key:
            # 如果没有异步管理器，返回零结果和空的Future
            initial_results = [0.0] * len(responses_to_evaluate)
            dummy_future = asyncio.Future()
            dummy_future.set_result(initial_results)
            return initial_results, dummy_future

        # 确保事件循环正在运行
        self._ensure_event_loop()

        # 创建异步任务
        async def evaluate_async():
            try:
                results = await self.async_openai_manager.submit_and_get_batch_results(
                    responses_to_evaluate,
                    check_interval=0.001
                )
                return results
            except Exception as e:
                logger.error(f"Async evaluation failed: {e}")
                return [0.0] * len(responses_to_evaluate)

        # 提交异步任务
        future = asyncio.run_coroutine_threadsafe(evaluate_async(), self._event_loop)

        # 立即返回初始结果，主线程可以继续
        initial_results = [
            0.0 if format_reward > 0 else 0.0
            for _, format_reward in responses_to_evaluate
        ]

        return initial_results, future

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算奖励分数，实现 GPU 计算与 OpenAI API 请求的解耦。
        在处理每个 batch 时，异步提交 OpenAI 评估请求，让 GPU 继续处理其他计算。
        """
        # 确保异步管理器已初始化（如果启用）
        if self.async_openai_manager:
            if not hasattr(self, '_workers_started') or not self._workers_started:
                self.start_workers()
                self._workers_started = True

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Group responses by prompt uid
        prompt_groups = defaultdict(list)
        uids = data.non_tensor_batch.get("uid", list(range(len(data))))
        assert len(uids) == len(data), f"UID list length ({len(uids)}) does not match batch size ({len(data)})."
        for i, uid in enumerate(uids):
            prompt_groups[uid].append(i)
        
        print(f"[PENALTY DEBUG] Total groups identified: {len(prompt_groups)}")

        # Calculate length penalty for each group
        for _, indices in prompt_groups.items():
            print(f"[PENALTY DEBUG] Processing group with indices {indices}")
            
            # Store data needed for each item in the group
            group_data = []
            for i in indices:
                data_item = data[i]
                
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                eos_token = self.tokenizer.eos_token
                if response_str.endswith(eos_token):
                    response_str = response_str[: -len(eos_token)]

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                score = result if isinstance(result, float) else result.get("score", 0.0)
                
                group_data.append({
                    "original_index": i,
                    "response_str": response_str,
                    "original_score": score,
                    "valid_response_length": valid_response_length.item(),
                })
            
            response_lengths = [item["valid_response_length"] for item in group_data]
            median_length = np.median(response_lengths)

            print(f"[PENALTY DEBUG] Processing group with indices {indices}, median response length: {median_length}")
            
            # Prepare for batched OpenAI calls - SIMPLIFIED VERSION
            responses_to_evaluate_for_batch = [
                (item["response_str"], item["original_score"]) for item in group_data
            ]

            # 选择 OpenAI 评估方法 - SIMPLIFIED
            if self.async_openai_manager:
                # ✅ 使用真正的异步 I/O，零 GPU 等待时间
                logger.info(f"🚀 Using TRUE ASYNC I/O for batch with {len(responses_to_evaluate_for_batch)} responses (zero GPU wait)")
                batched_openai_quality_rewards = self._get_batched_openai_quality_rewards_async(
                    responses_to_evaluate_for_batch
                )
            else:
                # ⚠️ 回退到同步方法（无 OpenAI 评估或禁用异步）
                logger.info(f"⚡ Using synchronous method for batch with {len(responses_to_evaluate_for_batch)} responses (set use_async_io=True for async)")
                batched_openai_quality_rewards = self._get_batched_openai_quality_rewards_sync(
                    responses_to_evaluate_for_batch
                )

            for idx_in_group, item in enumerate(group_data):
                i = item["original_index"]
                response_str = item["response_str"]
                score = item["original_score"]
                valid_response_length = item["valid_response_length"]
                length = valid_response_length # For clarity in penalty calculation

                reward = score
                
                openai_quality_reward = batched_openai_quality_rewards[idx_in_group]
                reward += openai_quality_reward
                reward_extra_info["openai_quality_reward"].append(openai_quality_reward)
                
                scaled_penalty = 0.0
                print(f"[PENALTY DEBUG] length_penalty_config {self.length_penalty_config}")
                # Apply piecewise length-based penalty
                if self.length_penalty_config and self.length_penalty_config.enable:
                    print(f"[PENALTY DEBUG] Length penalty calculation is ENABLED.")
                    
                    penalty_scale = getattr(self.length_penalty_config, "penalty_scale", 1.0)
                    max_penalty = getattr(self.length_penalty_config, "max_penalty", 1.0)
                    peak_ratio = getattr(self.length_penalty_config, "peak_ratio", 0.3)
                    outer_ratio = getattr(self.length_penalty_config, "outer_ratio", 0.5)
                    print(f"[PENALTY DEBUG] Config: penalty_scale={penalty_scale}, max_penalty={max_penalty}, peak_ratio={peak_ratio}, outer_ratio={outer_ratio}")

                    if outer_ratio <= peak_ratio:
                        raise ValueError("outer_ratio must be greater than peak_ratio in length_penalty_config")

                    penalty_component = 0.0
                    if median_length > 0:
                        linear_start = median_length * (1 - outer_ratio)
                        peak_start = median_length * (1 - peak_ratio)
                        peak_end = median_length * (1 + peak_ratio)
                        linear_end = median_length * (1 + outer_ratio)
                        
                        print(f"[PENALTY DEBUG] length={length}, median_length={median_length}")
                        print(f"[PENALTY DEBUG] No-penalty zone: [{peak_start:.2f}, {peak_end:.2f}]")
                        print(f"[PENALTY DEBUG] Linear penalty zone: [{linear_start:.2f}, {peak_start:.2f}) U ({peak_end:.2f}, {linear_end:.2f}]")

                        if length < linear_start or length > linear_end:
                            penalty_component = max_penalty
                        elif length >= linear_start and length < peak_start:
                            denominator = peak_start - linear_start
                            if denominator > 0:
                                penalty_component = max_penalty * (peak_start - length) / denominator
                        elif length > peak_end and length <= linear_end:
                            denominator = linear_end - peak_end
                            if denominator > 0:
                                penalty_component = max_penalty * (length - peak_end) / denominator
                        
                        print(f"[PENALTY DEBUG] Calculated penalty_component: {penalty_component:.4f}")

                    scaled_penalty = penalty_component * penalty_scale
                    reward -= scaled_penalty
                    reward_extra_info["length_penalty"].append(scaled_penalty)


                print(f"Reward calculated. Total: {reward}, Score: {score}, Length Penalty: {-scaled_penalty}, OpenAI Quality Reward: {openai_quality_reward}")
                reward_tensor[i, int(valid_response_length) - 1] = reward
                reward_extra_info["original_score"].append(score)
                reward_extra_info["response_length"].append(length)
                reward_extra_info["median_length"].append(median_length)

        if return_dict:
            result = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            result = reward_tensor
        return result
