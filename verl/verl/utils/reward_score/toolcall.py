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

    # --- 开始修改 ---

    # 1. 对输入进行健壮性检查，防止传入 None
    if result is None: result = []
    if answer is None: answer = []

    # 2. 过滤掉不规范的工具调用项
    #    一个有效的工具调用 item 必须是字典，并且同时包含 'name' 和 'arguments' 键。
    sanitized_result = [
        item for item in result 
        if isinstance(item, dict) and 'name' in item and 'arguments' in item
    ]
    sanitized_answer = [
        item for item in answer 
        if isinstance(item, dict) and 'name' in item and 'arguments' in item
    ]

    # 如果过滤后任意一个列表为空，进行空列表的比较逻辑
    if not sanitized_result or not sanitized_answer:
        # 如果两者都为空，则认为完全匹配
        return 2 if len(sanitized_result) == len(sanitized_answer) else 0

    # 3. 使用过滤后的、干净的数据来构建 Counter
    #    现在可以安全地访问 item['name'] 和 item['arguments']
    try:
        counter1_full = Counter((item["name"], json.dumps(normalize_value(item["arguments"]), sort_keys=True)) 
                                for item in sanitized_result)
        counter2_full = Counter((item["name"], json.dumps(normalize_value(item["arguments"]), sort_keys=True)) 
                                for item in sanitized_answer)
    except (TypeError, KeyError) as e:
        # 增加一层额外的保护，尽管可能性很小
        print(f"!!! Unexpected error during Counter creation: {e}")
        return 0
        
    # 完全匹配：函数名和参数都一致
    if counter1_full == counter2_full:
        return 2
    
    # 部分匹配：只比较函数名
    counter1_name = Counter(item["name"] for item in sanitized_result)
    counter2_name = Counter(item["name"] for item in sanitized_answer)

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

    # 1. 尝试解析 ground_truth，增加健壮性
    try:
        answer = json.loads(ground_truth)
        if answer is None:
            answer = []
    except (json.JSONDecodeError, TypeError):
        # 如果 ground_truth 本身有问题，也记录下来
        print("!!!!!!!!!!!! WARNING: FAILED TO PARSE GROUND TRUTH !!!!!!!!!!!!")
        print(f"Ground Truth String: {ground_truth}")
        answer = []

    result, output_string = extract_solution_v0(solution_str)

    extraction_failed = result is None
    if extraction_failed:
        print("!!!!!!!!!!!! WARNING: FAILED TO EXTRACT TOOL CALL !!!!!!!!!!!!")
        print(f"Original solution_str:\n---\n{solution_str}\n---")
        result = []  # 设置为安全的默认值以便后续代码运行 

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
    else:
        if do_print:
            print("--------"*5+"\n\n")
            print("get partial core:", 0)
        return 0

def format_reward(predict_str: str) -> float:
    """
    检查模型输出是否严格遵循 <think>...</think><tool_call>...</tool_call> 的格式
    返回值:
        1.0: 严格遵循格式
        0.0: 不严格遵循格式
    """
    try:
        # 一次性获取所有关键位置
        start_think_pos = predict_str.find("<think>")
        end_think_pos = predict_str.find("</think>")
        start_tool_pos = predict_str.find("<tool_call>")
        end_tool_pos = predict_str.find("</tool_call>")

        # 检查点1: 所有标签必须都存在 (find不返回-1)
        if -1 in (start_think_pos, end_think_pos, start_tool_pos, end_tool_pos):
            return 0.0

        # 检查点2: 位置必须严格递增
        if start_think_pos < end_think_pos < start_tool_pos < end_tool_pos:
            return 1.0
        else:
            return 0.0
            
    except AttributeError:
        # 捕获 predict_str 不是字符串的异常
        return 0.0

def compute_score_v0(solution_str, ground_truth, method='strict', json_score=0.1, format_factor = 0.1,  name_score = 0.6, score=1):

    format_score = format_reward(solution_str)

    acc_score = acc_reward(solution_str, ground_truth)

    score = (1.0 - format_factor) * acc_score + format_factor * format_score

    print("format_score: ",format_score, "acc_score: ", acc_score, "final_score: ", score)

    return score
