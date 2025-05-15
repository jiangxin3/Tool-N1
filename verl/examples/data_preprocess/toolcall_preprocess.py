# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random

qwen_tool_prompts ="""# Tool

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{tools}
</tools>

In each action step, you MUST: 
1. Think about the reasoning process in the mind and enclosed your reasoning within <think> </think> XML tags.
2. Then, provide a json object with function names and arguments within <tool_call></tool_call> XML tags. i.e., <tool_call>[{{"name": <function-name>, "arguments": <args-json-object>}}, {{"name": <function-name2>, "arguments": <args-json-object2>}}, ...]</tool_call>
3. Make sure both the reasoning and the tool call steps are included together in one single reply.
A complete reply example is: <think>To address the query, I need to send the email to Bob and then buy the banana through walmart. </think> <tool_call> [{{"name": "email", "arguments": {{"receiver": "Bob", "content": "I will bug banana through walmart"}}}}, {{"name": "walmart", "arguments": {{"input": "banana"}}}}]</tool_call>. Please make sure the type of the arguments is correct. 
"""

def dict2hg(data):
    hg_data = {"tools": [], "conversations": [], "answer": [], "raw_system": []}
    for item in data:
        tool = json.dumps(item["tools"])
        hg_data["tools"].append(tool)
        hg_data["conversations"].append(item["conversations"])
        answer = json.dumps(item["answer"])
        hg_data["answer"].append(answer)
        hg_data["raw_system"].append(item["raw_system"])
    hg_data= {"tools": hg_data["tools"][:], "conversations": hg_data["conversations"][:], "answer": hg_data["answer"][:], "raw_system": hg_data["raw_system"][:]}
    hg_data = Dataset.from_dict(hg_data)
    return hg_data

def construct_prompt(dp):

    def format_tools(tools):
        tools = json.loads(tools)
        string = ""
        for tool in tools:
            string += json.dumps({"type": "function", "function": tool}) + "\n"
        if string[-1] == "\n":
            string = string[:-1]
        return string

    tools = format_tools(dp["tools"])
    tool_prompt = qwen_tool_prompts.format(tools = tools)
    system = dp["raw_system"]
    conversations = dp["conversations"]
    prompt = []
    prompt.append({"role": "system", "content": system + tool_prompt})
    for tem in conversations:
        if tem["from"] == "human":
            prompt.append({"role": "user", "content": tem["value"]})
        elif tem["from"] == "gpt":
            prompt.append({"role": "assistant", "content": tem["value"]})
        elif tem["from"] == "observation":
            prompt.append({"role": "tool", "content": tem["value"]})
        elif tem["from"] == "function_call":
            prompt.append({"role": "assistant", "content": json.dumps(tem["value"])})
    return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='path/to/processed/rl_data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'toolcall'

    with open('path/to/original_data.json', 'r') as file:
        raw_dataset = json.load(file)
    
    TRAIN_SIZE = int(len(raw_dataset)*0.9)
    TEST_SIZE = int(len(raw_dataset)*0.1)

    random.shuffle(raw_dataset)

    train_dataset = raw_dataset[:TRAIN_SIZE]
    test_dataset = raw_dataset[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]
    train_dataset = dict2hg(train_dataset)
    test_dataset = dict2hg(test_dataset)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = construct_prompt(example)
            solution = example["answer"]
            data = {
                "data_source": "toolcall",
                "prompt": prompt,
                "ability": "toolcall",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "prompt_type": "qwen",
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
