# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-20.0
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
import requests # Added for OpenAI API calls
import re # Added for regex parsing

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

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

        # Extract OpenAI configs from length_penalty_config
        self.openai_api_key = getattr(self.length_penalty_config, "api_key", None)
        self.openai_model_name = getattr(self.length_penalty_config, "model_name", "deepseek-v3")
        self.openai_reward_coefficient = getattr(self.length_penalty_config, "reward_coefficient", 1.0)
        self.openai_api_endpoint = getattr(self.length_penalty_config, "api_endpoint", "https://qianfan.baidubce.com/v2")
        self.openai_system_prompt = '''
# 角色
你将扮演一位资深的“LLM 输出质量审查员”。你的核心专长是分析大型语言模型在执行任务时的内部思考链（`<think>` 标签内的内容）与其最终执行的动作（`<tool_call>` 标签内的内容）之间的逻辑一致性。

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

    def _get_openai_quality_reward(self, response_str: str, response_format_reward: float) -> float:
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
            match = re.search(r'最终评分\s(\d+)', model_output)
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

    def __call__(self, data: DataProto, return_dict: bool = False):
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
            response_lengths = []
            for i in indices:
                prompt_length = data[i].batch["prompts"].shape[-1]
                valid_response_length = data[i].batch["attention_mask"][prompt_length:].sum()
                response_lengths.append(valid_response_length.item())
            
            median_length = np.median(response_lengths)

            print(f"[PENALTY DEBUG] Processing group with indices {indices}, median response length: {median_length}")
            
            for i, length in zip(indices, response_lengths):
                data_item = data[i]
                
                # Calculate original score
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
                reward = score
                # Add OpenAI quality reward
                openai_quality_reward = self._get_openai_quality_reward(response_str, reward)
                reward += openai_quality_reward
                reward_extra_info["openai_quality_reward"].append(openai_quality_reward)
                scaled_penalty = 0.0
                print(f"[PENALTY DEBUG] length_penalty_config {self.length_penalty_config}")
                # Apply piecewise length-based penalty
                if self.length_penalty_config and self.length_penalty_config.enable:
                    print(f"[PENALTY DEBUG] Length penalty calculation is ENABLED.")
                    # This implements a true penalty system using an "inverted trapezoid" function.
                    # A penalty is calculated based on the response length (L) relative to the median length (M).
                    #
                    # Let p = peak_ratio, o = outer_ratio. The penalty formula is:
                    #
                    # Penalty =
                    #   - 0, if M*(1-p) <= L <= M*(1+p) (zero penalty zone)
                    #   - linear increase from 0 to max_penalty, if M*(1-o) <= L < M*(1-p)
                    #   - linear increase from 0 to max_penalty, if M*(1+p) < L <= M*(1+o)
                    #   - max_penalty, if L < M*(1-o) or L > M*(1+o)
                    #
                    # The final reward is `reward = score - (Penalty * penalty_scale)`.
                    
                    # Configuration for the penalty function
                    penalty_scale = getattr(self.length_penalty_config, "penalty_scale", 1.0)
                    max_penalty = getattr(self.length_penalty_config, "max_penalty", 1.0)
                    peak_ratio = getattr(self.length_penalty_config, "peak_ratio", 0.3)
                    outer_ratio = getattr(self.length_penalty_config, "outer_ratio", 0.5)
                    print(f"[PENALTY DEBUG] Config: penalty_scale={penalty_scale}, max_penalty={max_penalty}, peak_ratio={peak_ratio}, outer_ratio={outer_ratio}")

                    if outer_ratio <= peak_ratio:
                        raise ValueError("outer_ratio must be greater than peak_ratio in length_penalty_config")

                    penalty_component = 0.0
                    # Only apply penalty if median_length is positive to avoid division by zero.
                    if median_length > 0:
                        # Define the points of the inverted trapezoid function
                        linear_start = median_length * (1 - outer_ratio)
                        peak_start = median_length * (1 - peak_ratio)
                        peak_end = median_length * (1 + peak_ratio)
                        linear_end = median_length * (1 + outer_ratio)
                        
                        print(f"[PENALTY DEBUG] length={length}, median_length={median_length}")
                        print(f"[PENALTY DEBUG] No-penalty zone: [{peak_start:.2f}, {peak_end:.2f}]")
                        print(f"[PENALTY DEBUG] Linear penalty zone: [{linear_start:.2f}, {peak_start:.2f}) U ({peak_end:.2f}, {linear_end:.2f}]")

                        if length < linear_start or length > linear_end:
                            # Outside the outer bounds: maximum penalty
                            penalty_component = max_penalty
                        elif length >= linear_start and length < peak_start:
                            # On the ascending slope of the penalty (for short responses)
                            denominator = peak_start - linear_start
                            if denominator > 0:
                                penalty_component = max_penalty * (peak_start - length) / denominator
                        elif length > peak_end and length <= linear_end:
                            # On the ascending slope of the penalty (for long responses)
                            denominator = linear_end - peak_end
                            if denominator > 0:
                                penalty_component = max_penalty * (length - peak_end) / denominator
                        # Inside the peak_start to peak_end range, penalty_component remains 0.
                        
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
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
