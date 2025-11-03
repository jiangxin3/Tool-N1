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
from .openai_worker import get_request_hash, OpenAIWorkerManager

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

        # Initialize OpenAI worker manager for asynchronous API calls
        num_workers = getattr(self.length_penalty_config, "num_async_workers", 4) if self.length_penalty_config else 4
        self.openai_worker_manager = OpenAIWorkerManager(self.length_penalty_config, num_workers=num_workers)

        # Extract OpenAI configs from length_penalty_config
        self.openai_api_key = getattr(self.length_penalty_config, "api_key", None)
        self.openai_model_name = getattr(self.length_penalty_config, "model_name", "deepseek-v3")
        self.openai_reward_coefficient = getattr(self.length_penalty_config, "reward_coefficient", 1.0)
        self.openai_api_endpoint = getattr(self.length_penalty_config, "api_endpoint", "https://qianfan.baidubce.com/v2/chat/completions")
        self.openai_system_prompt = '''
# 角色
你将扮演一位资深的"LLM 输出质量审查员"。你的核心专长是分析大型语言模型在执行任务时的内部思考链（`<think>` 标签内的内容）与其最终执行的动作（`<tool_call>` 标签内的内容）之间的逻辑一致性。

# 核心任务
你的唯一任务是接收一段包含 `<think>` 和 `<tool_call>` 的文本，并对其进行严格的逻辑审查。你必须判断模型的思考过程是否合理、清晰、对读者有帮助，以及其最终的工具调用是否是该思考过程的直接、合理且唯一的产物。

# 关键评估原则：一致性是最高准则
模型的思考和行动必须完全一致。任何脱节都代表着严重的逻辑缺陷，**将直接导致评分极低**。

*   **优秀的例子（高分）**:
    ```xml
    <think>
    用户想知道北京的天气。为了帮他解答，我需要检查一下我的工具箱。我发现有一个名为 `get_weather` 的工具，它看起来正好能用。这个工具需要一个 `city` 参数，用户在提问中已经明确提到了“北京”。因此，最合理的下一步就是调用 `get_weather` 工具，并把“北京”作为城市参数传给它。
    </think>
    <tool_call>
    {"name": "get_weather", "arguments": {"city":"北京"}}
    </tool_call>
    ```
    *评价：思考过程不仅逻辑严密，而且语言流畅，清晰地展示了“发现问题 -> 寻找工具 -> 匹配参数 -> 决定行动”的全过程，让读者感觉模型在积极地帮助自己。*

*   **不可接受的例子（低分） - [新版示例]**:
    ```xml
    <think>
    用户想订一张从上海到北京的机票。好的，我应该使用 `book_flight` 工具。我需要出发地和目的地。用户的指令很明确，出发地是'上海'，目的地是'北京'。
    </think>
    <tool_call>
    {"name": "book_flight", "arguments": {"departure_city": "北京", "destination_city": "上海"}}
    </tool_call>
    ```
    *评价：这是一个典型的思考与行动不一致的严重错误。尽管思考过程正确地识别了用户的意图和所需信息，但在生成最终行动时，却将出发地和目的地两个关键参数完全搞反了。这直接违背了思考过程的结论，并会导致完全错误的操作，因此应给予极低分。*

# 评估维度
你将从以下三个专业维度进行分析，并写入报告：

1.  **思考-行动一致性 (Reasoning-Action Consistency)**:
    *   **核心问题**: `<tool_call>` 中的函数名和参数是否是 `<think>` 过程的直接、合乎逻辑的结论？
    *   **评估要点**: 检查是否存在函数名不匹配、参数不一致、或者思考结论与实际行动相悖的情况。**这是最重要的评分项，一旦出现问题，总分将被限制在1-3分。**

2.  **思考过程的质量与清晰度 (Reasoning Quality & Clarity)**:
    *   **核心问题**: `<think>` 块内的逻辑是否清晰、合理、高效，并且对人类读者友好且有帮助？
    *   **评估要点**: 
        *   **逻辑性**: 是否正确理解用户意图？推理步骤是否连贯、无遗漏？
        *   **可读性与流畅度**: 思考过程的文字描述是否清晰、易于人类理解？**是否像一个连贯的内心独白，而不是零散的指令集？**
        *   **辅助性与透明度**: **思考过程是否能让审查者清晰地看到‘为什么’模型会做出最终决策？是否体现了积极解决问题的态度，让读者觉得这个过程本身就很有帮助？**

3.  **工具调用有效性 (Tool Call Validity)**:
    *   **核心问题**: `<tool_call>` 本身的格式是否正确，参数是否合理？
    *   **评估要点**: 函数名和参数的格式是否符合规范？即使思考过程有误，调用的参数值本身是否符合常识？

# 输出格式 (必须严格遵守)
你的输出必须包含详细的分析报告和最终的单一数字评分。

## LLM 行为逻辑审查报告

### 维度分析
*   **思考-行动一致性**: [对此项进行详细的文字评价，明确指出思考和行动是否一致，并解释原因。]
*   **思考过程的质量与清晰度**: [对此项进行详细的文字评价，综合分析其逻辑性、可读性和辅助性。]
*   **工具调用有效性**: [对此项进行详细的文字评价，分析工具调用本身的格式和内容。]

### 核心问题诊断
*   [用一句话精准概括本次输出最严重的问题。]

### 改进建议
*   [针对核心问题，提出具体的改进方向。]

---

## 最终评分
[此处必须且只能输出一个1到10的整数，该分数是你综合以上所有分析后得出的最终结论]
'''

    def start_workers(self):
        """启动异步 OpenAI worker 进程"""
        if self.openai_worker_manager.is_enabled:
            logger.info("Starting OpenAI worker processes for asynchronous API calls...")
            self.openai_worker_manager.start()
            logger.info("OpenAI worker processes started successfully.")

    def shutdown_workers(self):
        """关闭异步 OpenAI worker 进程"""
        if self.openai_worker_manager.is_enabled:
            logger.info("Shutting down OpenAI worker processes...")
            self.openai_worker_manager.shutdown()
            logger.info("OpenAI worker processes shut down successfully.")

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

    def _get_batched_openai_quality_rewards(self, responses_to_evaluate: list[tuple[str, float]]) -> list[float]:
        """
        使用异步 worker 进程批量获取 OpenAI 质量评估奖励，实现请求与 GPU 计算的解耦。
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Skipping OpenAI quality evaluation.")
            return [0.0] * len(responses_to_evaluate)

        # 如果 OpenAI worker manager 未启用，回退到同步方法
        if not self.openai_worker_manager.is_enabled:
            logger.info("OpenAI worker manager not enabled, falling back to synchronous method.")
            return self._get_batched_openai_quality_rewards_sync(responses_to_evaluate)

        logger.info(f"Using asynchronous OpenAI workers to evaluate {len(responses_to_evaluate)} responses")

        # 获取任务队列和结果字典
        task_queue = self.openai_worker_manager.get_task_queue()
        results_dict = self.openai_worker_manager.get_results_dict()

        if task_queue is None or results_dict is None:
            logger.warning("Task queue or results dict not available, falling back to synchronous method.")
            return self._get_batched_openai_quality_rewards_sync(responses_to_evaluate)

        # 收集所有需要评估的响应，为每个响应生成哈希和请求
        request_hashes = []
        requests_to_submit = []

        for response_str, response_format_reward in responses_to_evaluate:
            if response_format_reward == 0:
                # 如果响应格式奖励为0，跳过 OpenAI 评估
                request_hashes.append(None)
                requests_to_submit.append(None)
            else:
                # 生成请求哈希和请求字符串
                request_hash = get_request_hash(response_str)
                request_hashes.append(request_hash)
                requests_to_submit.append(response_str)

        # 提交任务到队列
        for request_hash, request_str in zip(request_hashes, requests_to_submit):
            if request_hash is not None and request_str is not None:
                # 只有当结果尚不存在时才提交任务
                if request_hash not in results_dict:
                    task_queue.put((request_hash, request_str))

        # 等待所有任务完成并收集结果
        scores = [0.0] * len(responses_to_evaluate)
        timeout_seconds = 300  # 5分钟超时

        start_time = time.time()
        completed_count = 0
        total_to_process = sum(1 for h in request_hashes if h is not None)

        # 逐个等待结果，但使用较短的超时时间以便及时处理
        for i, (request_hash, _) in enumerate(zip(request_hashes, requests_to_submit)):
            if request_hash is None:
                # 原响应格式奖励为0，保持分数为0
                completed_count += 1
                continue

            # 等待结果，设置超时和进度日志
            wait_start = time.time()
            while request_hash not in results_dict:
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Timeout waiting for OpenAI API result for request {request_hash}")
                    break
                # 每10秒记录一次进度
                if time.time() - wait_start > 10.0:
                    completed = sum(1 for j in range(i + 1) if request_hashes[j] is not None and request_hashes[j] in results_dict)
                    progress = (completed / total_to_process) * 100 if total_to_process > 0 else 0
                    logger.info(f"Waiting for OpenAI results... {progress:.1f}% complete ({completed}/{total_to_process})")
                    wait_start = time.time()
                time.sleep(0.1)  # 短暂等待后重试

            # 获取结果（如果超时，结果可能不存在）
            if request_hash in results_dict:
                score = results_dict[request_hash]
                scores[i] = score
                completed_count += 1
                logger.debug(f"Received OpenAI score {score} for request {request_hash}")
            else:
                logger.warning(f"Failed to get OpenAI result for request {request_hash}, using 0.0")
                scores[i] = 0.0
                completed_count += 1

            # 每完成25%的任务记录一次进度
            if (i + 1) % max(1, len(responses_to_evaluate) // 4) == 0:
                progress = (completed_count / len(responses_to_evaluate)) * 100
                logger.info(f"OpenAI quality evaluation progress: {progress:.1f}% ({completed_count}/{len(responses_to_evaluate)})")

        logger.info(f"Completed asynchronous OpenAI quality evaluation for {len(responses_to_evaluate)} responses")
        return scores

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

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算奖励分数，实现 GPU 计算与 OpenAI API 请求的解耦。
        在处理每个 batch 时，异步提交 OpenAI 评估请求，让 GPU 继续处理其他计算。
        """
        # 启动异步 OpenAI workers（如果尚未启动）
        if self.openai_worker_manager.is_enabled:
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
            
            # Prepare for batched OpenAI calls
            responses_to_evaluate_for_batch = [
                (item["response_str"], item["original_score"]) for item in group_data
            ]
            batched_openai_quality_rewards = self._get_batched_openai_quality_rewards(responses_to_evaluate_for_batch)

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
