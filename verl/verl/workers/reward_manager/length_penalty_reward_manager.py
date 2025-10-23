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

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

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
        
        # Group responses by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(data)):
            prompt_ids = data[i].batch["prompts"]
            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompt_groups[prompt_str].append(i)

        # Calculate length penalty for each group
        for prompt_str, indices in prompt_groups.items():
            response_lengths = []
            for i in indices:
                response_ids = data[i].batch["responses"]
                valid_response_length = data[i].batch["attention_mask"][response_ids.shape[-1]:].sum()
                response_lengths.append(valid_response_length.item())
            
            median_length = np.median(response_lengths)
            
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

                # Apply piecewise length-based reward
                if self.length_penalty_config and self.length_penalty_config.get("enable", False):
                    # Implements a trapezoidal reward function based on the response length relative to the median length.
                    # This is configured via `length_penalty_config` with fields like:
                    # - `length_reward_scale`: A scaling factor for the calculated length reward component.
                    # - `max_bonus`: The maximum reward value at the peak of the trapezoid (before scaling).
                    # - `peak_ratio`: The radius of the flat top, as a fraction of the median length (e.g., 0.1 for +/- 10%).
                    # - `outer_ratio`: The radius of the base, as a fraction of the median length (e.g., 0.5 for +/- 50%).
                    
                    length_reward_scale = self.length_penalty_config.get("length_reward_scale", 1.0)
                    max_bonus = self.length_penalty_config.get("max_bonus", 1.0)
                    peak_ratio = self.length_penalty_config.get("peak_ratio", 0.3)
                    outer_ratio = self.length_penalty_config.get("outer_ratio", 0.5)

                    if outer_ratio <= peak_ratio:
                        raise ValueError("outer_ratio must be greater than peak_ratio in length_penalty_config")

                    length_reward_component = 0.0
                    # Only apply length reward if median_length is positive to avoid division by zero.
                    if median_length > 0:
                        # Define the points of the trapezoid function using ratios of the median length
                        linear_start = median_length * (1 - outer_ratio)
                        peak_start = median_length * (1 - peak_ratio)
                        peak_end = median_length * (1 + peak_ratio)
                        linear_end = median_length * (1 + outer_ratio)
                        
                        if length >= peak_start and length <= peak_end:
                            # Inside the peak interval: maximum bonus
                            length_reward_component = max_bonus
                        elif length >= linear_start and length < peak_start:
                            # On the ascending slope of the trapezoid
                            denominator = peak_start - linear_start
                            if denominator > 0:
                                length_reward_component = max_bonus * (length - linear_start) / denominator
                        elif length > peak_end and length <= linear_end:
                            # On the descending slope of the trapezoid
                            denominator = linear_end - peak_end
                            if denominator > 0:
                                length_reward_component = max_bonus * (linear_end - length) / denominator
                        # Outside the `outer_ratio`, the length_reward_component remains 0.

                    scaled_length_reward = length_reward_component * length_reward_scale
                    reward += scaled_length_reward
                    reward_extra_info["length_reward"].append(scaled_length_reward)

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
