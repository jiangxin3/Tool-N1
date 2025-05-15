# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/Tool-N1/blob/main/LICENSE

import json
import os

def process_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    new_instruction = '''When you want to perform tool call, in each action step, providing a json object with function names and arguments within <tool_call></tool_call> XML tags. i.e., <tool_call>[{"name": <function-name>, "arguments": <args-json-object>}, {"name": <function-name2>, "arguments": <args-json-object2>}, ...]</tool_call>
A complete reply example is: <tool_call> [{"name": "email", "arguments": {"receiver": "Bob", "content": "I will bug banana through walmart"}}, {"name": "walmart", "arguments": {"input": "banana"}}]</tool_call>. Please make sure the type of the arguments is correct.'''

    for entry in data:
        if 'system_prompt' in entry and "In each action step, you MUST" in entry['system_prompt']:
            parts = entry['system_prompt'].split("In each action step, you MUST", 1)
            entry['system_prompt'] = parts[0] + new_instruction

        if 'output' in entry:
            think_start = entry['output'].find("<think>")
            think_end = entry['output'].find("</think>") + len("</think>")
            if think_start != -1 and think_end != -1:
                entry['output'] = entry['output'][think_end:].lstrip("\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

process_json("path/to/tool_sft.json", "path/to/data/raw_tool_sft.json")
