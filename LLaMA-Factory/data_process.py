# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/Tool-N1/blob/main/LICENSE

import json

def truncate_after_first_tool_call_end(s):
    marker = "</tool_call>"
    index = s.find(marker)
    if index != -1:
        return s[:index + len(marker)]
    return s

def truncate_before_expert_prompt(s):
    marker = "You are an expert in composing functions"
    index = s.find(marker)
    if index != -1:
        return s[index:]
    return s

# Define file paths
input_file = "path/to/distilled_data/sft_data.json"
output_file = "path/to/data/tool_sft.json"

# Load the input JSON data
with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Transform the data to match the Alpaca format
alpaca_data = []
for item in input_data:
    transformed_item = {
        "system_prompt": truncate_before_expert_prompt(item.get("system_prompt", "")),
        "query": item.get("query", ""),
        "output": truncate_after_first_tool_call_end(item.get("output", "")),
        "instruction": ""
    }
    alpaca_data.append(transformed_item)

# Save the transformed data to the output file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"Data has been successfully transformed and saved to {output_file}")