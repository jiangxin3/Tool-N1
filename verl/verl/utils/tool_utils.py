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


import os
import re
import ast
from collections import Counter
import json

def _split_top_level_args(params_str):
    items = []
    current = []
    
    bracket_level = 0
    brace_level = 0
    paren_level = 0
    
    quote_char = None  
    
    for char in params_str:
        if quote_char:
            if char == quote_char:
                quote_char = None
                current.append(char)
            else:
                current.append(char)
            continue
        else:
            if char in ["'", '"']:
                quote_char = char
                current.append(char)
                continue

        if char == '(':
            paren_level += 1
            current.append(char)
        elif char == ')':
            paren_level -= 1
            current.append(char)
        elif char == '[':
            bracket_level += 1
            current.append(char)
        elif char == ']':
            bracket_level -= 1
            current.append(char)
        elif char == '{':
            brace_level += 1
            current.append(char)
        elif char == '}':
            brace_level -= 1
            current.append(char)
        elif char == ',' and bracket_level == 0 and brace_level == 0 and paren_level == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        items.append("".join(current).strip())
    
    return items

def _parse_params(params_str):
    result = {}
    pairs = _split_top_level_args(params_str)
    
    for item in pairs:
        if '=' in item:
            key, val = item.split('=', 1)
            key = key.strip()
            val = val.strip()
            try:
                parsed_val = ast.literal_eval(val)
            except Exception:
                parsed_val = val.strip('\'"')
            result[key] = parsed_val
    return result

def _parse_function_string(function_string):

    function_pattern = r'([a-zA-Z0-9\._\?\|\s\-]+)\((.*?)\)'
    matches = re.findall(function_pattern, function_string)
    parsed_functions = []
    for func_name, params_str in matches:
        func_name = func_name.strip()
        params_dict = _parse_params(params_str)
        parsed_functions.append((func_name, params_dict))
    
    return parsed_functions

def _extract_functions_from_system(text): 
    pattern = r'Here is a list of functions in JSON format that you can invoke:\n(.*?)\nShould you decide to return the function call\(s\).'
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL允许'.'匹配换行符
    if match:
        s = match.group(1).strip()
        s = s[:-2] + "]"
        s = json.loads(s)
        return  s
    else:
        return None

def _validate_function_format(s):
    if not s or not isinstance(s, str):
        return False
 
    pattern = r'^\[\s*([a-zA-Z0-9\._\?\|\s\-/]+\(.*?\)\s*)(,\s*[a-zA-Z0-9\._\?\|\s\-/]+\(.*?\)\s*)*\]$'
    return bool(re.match(pattern, s.strip()))
