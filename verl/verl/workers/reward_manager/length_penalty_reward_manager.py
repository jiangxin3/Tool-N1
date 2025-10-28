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
                scaled_penalty = 0.0

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

                print(f"Reward calculated. Total: {reward}, Score: {score}, Length Penalty: {-scaled_penalty}")
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
