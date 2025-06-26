# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

    if len(result) == 0 or len(answer) == 0:
        if len(result) == len(answer):
            return 2
        else:
            return 0
    else:
        try:
            counter1_full = Counter((item["name"], json.dumps(item["arguments"], sort_keys=True)) 
                                    for item in result)
            counter2_full = Counter((item["name"], json.dumps(item["arguments"], sort_keys=True)) 
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

def validate_format(tool_call_list):
    for item in tool_call_list:
        if not isinstance(item, dict):
            return 0
    for item in tool_call_list:
        if "name" not in item.keys() or "arguments" not in item.keys():
            return 0
    return 1

def extract_solution_llama(tool_call_str):
    
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
    output_string = tool_call_str
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))

    if not matches: # no tool call
        return None, output_string, False

    last_content = matches[-1].group(1).strip()
    
    try:
        return json.loads(last_content),output_string, True
    except json.JSONDecodeError:
        return None, output_string, True

def compute_score_llama(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)
    result, output_string, match = extract_solution_llama(solution_str)

    do_print = random.randint(1, 64) == 1

    if isinstance(result, str):
        result = json.loads(result)

    if isinstance(result, dict):
        tem = []
        tem.append(result)
        result = tem

    if isinstance(answer, str):
        answer = json.loads(answer)

    if do_print:
        print(solution_str)
        print(output_string)
        print(match)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    if match:

        if "<think>" not in output_string or "</think>" not in output_string:
            if do_print:
                print("--------"*5+"\n\n")
                print("not thinking when math:", -1)
            return 0

        if result is None:
            if do_print:
                print("--------"*5+"\n\n")
                print("result is None:", -1)
            return 0
        
        if not validate_format(result):
            if do_print:
                print("--------"*5+"\n\n")
                print("result wrong formate:",-1)
            return 0
        
        if validate_result(result, answer) == 2:
            if do_print:
                print("--------"*5+"\n\n")
                print("get full core:", 1)
            return 1
        else:
            if do_print:
                print("--------"*5+"\n\n")
                print("wrong answer", -1)
            return 0

    else:

        result = []
        if validate_result(result, answer) == 2:
            if do_print:
                print("--------"*5+"\n\n")
                print("get full core:", 1)
            return 1
        else:
            if do_print:
                print("--------"*5+"\n\n")
                print("wrong answer", -1)
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

def compute_score_v0(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)

    result, output_string = extract_solution_v0(solution_str)

    #do_print = random.randint(1, 64) == 1
    do_print = 1

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

    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    if result != None:
        if "<think>" not in output_string or "</think>" not in output_string:
            if do_print:
                print("--------"*5+"\n\n")
                print("not thinking:", -1)
            return 0

    if result is None:
        if do_print:
            print("--------"*5+"\n\n")
            print("result is None:", -1)
        return 0
    
    if not validate_format(result):
        if do_print:
            print("--------"*5+"\n\n")
            print("result wrong formate:",-1)
        return 0
    
    if validate_result(result, answer) == 2:
        if do_print:
            print("--------"*5+"\n\n")
            print("get full core:", 1)
        return 1
    else:
        if do_print:
            print("--------"*5+"\n\n")
            print("wrong answer", -1)
        return 0

def extract_solution_v1(tool_call_str):
    
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
        
    output_string = tool_call_str

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content),output_string
    except json.JSONDecodeError:
        return None, output_string

def compute_score_v1(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v1(solution_str)
    do_print = random.randint(1, 64) == 1
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


    if do_print:
        print(solution_str)

    # case 4.1
    if result is None: 
        if "<think>" in output_string and "</think>" in output_string:
            if do_print:
                print("result is None with reason:", 0)
            return 0
        else:
            if do_print:
                print("--------"*5+"\n\n")
                print("result is None without reason:", 0)
            return 0
    else:
        if validate_format(result) and validate_result(result, answer) == 2:
            # case 1
            if  ("<think>" in output_string and "</think>" in output_string):
                if do_print:
                    print("correct result with reason:", 1)
                return 1.2
            # case 2
            else: 
                if do_print:
                    print("correct result without reason:", 0.5)
                return 1
        else:
            # case 4.2
            if  ("<think>" in output_string and "</think>" in output_string):
                if do_print:
                    print("wrong result with reason:", -0.2)
                return -0.2
            # case 3
            else:
                if do_print:
                    print("wrong result without reason:", 0)
                return 0
            
def extract_solution_v2(tool_call_str):
    
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
        
    output_string = tool_call_str

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content),output_string
    except json.JSONDecodeError:
        return None, output_string

def compute_score_v2(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v2(solution_str)
    do_print = random.randint(1, 64) == 1

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

    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    if result is None:
        if do_print:
            print("--------"*5+"\n\n")
            print("result is None:", -1)
        return 0
    
    if not validate_format(result):
        if do_print:
            print("--------"*5+"\n\n")
            print("result wrong formate:",-1)
        return 0
        
    if validate_result(result, answer) == 2:
        if do_print:
            print("--------"*5+"\n\n")
            print("get full core:", 1)
        return 1
    else:
        if do_print:
            print("--------"*5+"\n\n")
            print("wrong answer", -1)
        return 0

def extract_solution_v3(tool_call_str):
    
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
        
    output_string = tool_call_str

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content),output_string
    except json.JSONDecodeError:
        return None, output_string

def compute_score_v3(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v4(solution_str)
    do_print = random.randint(1, 64) == 1
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
    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    total = 0
    if "<think>" in output_string and "</think>" in output_string:
        total += 0.2

    if result is None:
        if do_print:
            print("--------"*5+"\n\n")
            print("result is None:", total)
        return total
    if not validate_format(result):
        if do_print:
            print("--------"*5+"\n\n")
            print("result wrong formate:", total)
        return total
    if validate_result(result, answer) == 2:
        total += 0.8
        if do_print:
            print("--------"*5+"\n\n")
            print("get full core:", total)
        return total
    else:
        if do_print:
            print("--------"*5+"\n\n")
            print("wrong answer", total)
        return total

def extract_solution_v4(tool_call_str):
    
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
        
    output_string = tool_call_str

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content),output_string
    except json.JSONDecodeError:
        return None, output_string

def compute_score_v4(solution_str, ground_truth, method='strict', json_score=0.1, format_score = 0.3,  name_score = 0.6, score=1):

    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v4(solution_str)
    do_print = random.randint(1, 64) == 1
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
    if do_print:
        print("************solution_str************")
        print(solution_str)
        print(f"Extracted result: {result}")
        print(f"Solution string: {answer}")

    total = 0
    if "<think>" in output_string and "</think>" in output_string:
        total += 0.2

    if result is None:
        if do_print:
            print("--------"*5+"\n\n")
            print("result is None:", total)
        return total
    if not validate_format(result):
        if do_print:
            print("--------"*5+"\n\n")
            print("result wrong formate:", total)
        return total

    if validate_result(result, answer) == 2:
        total += 0.8
        if do_print:
            print("--------"*5+"\n\n")
            print("get full core:", total)
        return total
    elif validate_result(result, answer) == 1:
        total += 0.2
        if do_print:
            print("--------"*5+"\n\n")
            print("get func name correct:", total)
        return total
    else:
        if do_print:
            print("--------"*5+"\n\n")
            print("wrong answer", total)
        return total
