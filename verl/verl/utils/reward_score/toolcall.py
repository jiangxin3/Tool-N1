# Copyright 2025 NIO CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
import json
import random

def validate_result(result, answer):

    # 解决response中的数字类型，统一为字符串类型
    def normalize_value(value):
        if isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, dict):
            return {k: normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value


    if len(result) == 0 or len(answer) == 0:
        if len(result) == len(answer):
            return 2
        else:
            return 0
    else:
        try:
            counter1_full = Counter((item["name"], json.dumps(normalize_value(item["arguments"]), sort_keys=True)) 
                                    for item in result)
            counter2_full = Counter((item["name"], json.dumps(normalize_value(item["arguments"]), sort_keys=True)) 
                                    for item in answer)
        except TypeError:
            return 0
        if counter1_full == counter2_full:
            return 2
        
        counter1_name = Counter(item["name"] for item in result)
        counter2_name = Counter(item["name"] for item in answer)

        if counter1_name == counter2_name:
            return 1
        
        return 0

def extract_solution_v0(tool_call_str):
    # 查找从 marker 开始的内容
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]

    output_string = tool_call_str
 

    match = re.search(r'<tool_call>([^{}]*(\{.*?\})[^{}]*)</tool_call>', tool_call_str, re.DOTALL)
    

    if not match:
        return None, output_string
    
    content = match.group(1).strip()


    try:
        # 尝试解析为单个 JSON 对象或 JSON 数组
        result = json.loads(content)
        if isinstance(result, dict):
            return [result], output_string
        elif isinstance(result, list):
            return result, output_string
    except json.JSONDecodeError:
        # 如果是多个独立 JSON 对象，手动分割
        results = []
        for obj in re.finditer(r'\{(?:[^{}]|(?:\{[^{}]*\})*)*\}', content):
            try:
                results.append(json.loads(obj.group()))
            except:
                continue
        if results:
            return results, output_string
    
    return None, output_string


def acc_reward(solution_str: str, ground_truth: str) -> float:

    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v0(solution_str)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None
            
    if isinstance(result, dict):
        tem = []
        tem.append(result)
        result = tem

    if isinstance(answer, str):
        answer = json.loads(answer)

    do_print = random.randint(1, 64) == 1
    #do_print = 1

    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")
    if validate_result(result, answer) == 2:
        if do_print:
            print("--------"*5+"\n\n")
            print("get full core:", 1)
    return 1

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r".*<think>.*</think>.*<tool_call>([^{}]*(\{.*?\})[^{}]*)</tool_call>.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1 if match_result else 0

def compute_score_v0(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.1,  name_score = 0.6, score=1):

    format_score = format_reward(solution_str)

    acc_score = acc_reward(solution_str, ground_truth)

    score = (1.0 - format_score) * acc_score + format_score * format_score

    print("format_score: ",format_score, "acc_score: ", acc_score, "final_score: ", score)

    return score
