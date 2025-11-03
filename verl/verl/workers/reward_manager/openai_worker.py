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

import logging
import multiprocessing
import requests
import re
import time
import hashlib

logger = logging.getLogger(__name__)

def get_request_hash(request_str: str) -> str:
    """Generates a SHA256 hash for a given request string."""
    return hashlib.sha256(request_str.encode('utf-8')).hexdigest()

def openai_worker_process(
    task_queue: multiprocessing.Queue,
    results_dict: dict,
    shutdown_event: multiprocessing.Event,
    api_key: str,
    model_name: str,
    api_endpoint: str,
    system_prompt: str,
    reward_coefficient: float,
):
    """
    A worker process that fetches tasks from a queue, calls the OpenAI API,
    and puts the results into a shared dictionary.
    """
    logger.info(f"OpenAI worker process {multiprocessing.current_process().name} started.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    while not shutdown_event.is_set():
        try:
            # Wait for a task with a timeout to allow checking the shutdown event
            request_hash, request_str = task_queue.get(timeout=1.0)

            # Skip if result already exists (e.g., processed by another worker)
            if request_hash in results_dict:
                continue

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request_str},
                ],
                "temperature": 0.0,
            }

            try:
                response = requests.post(api_endpoint, headers=headers, json=payload)
                response.raise_for_status()
                response_json = response.json()
                
                model_output = response_json["choices"][0]["message"]["content"]
                
                match = re.search(r'最终评分\s*\n\s*(\d+)', model_output)
                if match:
                    score = float(match.group(1))
                    results_dict[request_hash] = score * reward_coefficient
                else:
                    logger.warning(f"Could not parse OpenAI score from: {model_output}")
                    results_dict[request_hash] = 0.0
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI API request failed for hash {request_hash}: {e}")
                results_dict[request_hash] = 0.0 # Store failure to prevent retries
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing OpenAI API response for hash {request_hash}: {e}")
                results_dict[request_hash] = 0.0

        except multiprocessing.queues.Empty:
            # This is expected when the queue is empty, just continue the loop
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred in the OpenAI worker: {e}")

    logger.info(f"OpenAI worker process {multiprocessing.current_process().name} shutting down.")


class OpenAIWorkerManager:
    """
    Manages a pool of worker processes for handling OpenAI API calls asynchronously.
    """
    def __init__(self, length_penalty_config, num_workers: int = 4):
        if not length_penalty_config or not length_penalty_config.enable_openai_reward:
            self.is_enabled = False
            return
        
        self.is_enabled = True
        self.num_workers = num_workers
        self.length_penalty_config = length_penalty_config

        # Extract OpenAI configs
        self.api_key = getattr(self.length_penalty_config, "api_key", None)
        self.model_name = getattr(self.length_penalty_config, "model_name", "deepseek-v3")
        self.reward_coefficient = getattr(self.length_penalty_config, "reward_coefficient", 1.0)
        self.api_endpoint = getattr(self.length_penalty_config, "api_endpoint", "https://qianfan.baidubce.com/v2/chat/completions")
        self.system_prompt = '''
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
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. OpenAI reward worker will be disabled.")
            self.is_enabled = False
            return

        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.results_dict = self.manager.dict()
        self.shutdown_event = self.manager.Event()
        self.pool = []

    def start(self):
        """Starts the worker processes."""
        if not self.is_enabled:
            return

        logger.info(f"Starting {self.num_workers} OpenAI worker processes...")
        for _ in range(self.num_workers):
            process = multiprocessing.Process(
                target=openai_worker_process,
                args=(
                    self.task_queue,
                    self.results_dict,
                    self.shutdown_event,
                    self.api_key,
                    self.model_name,
                    self.api_endpoint,
                    self.system_prompt,
                    self.reward_coefficient,
                ),
            )
            process.start()
            self.pool.append(process)

    def shutdown(self):
        """Signals workers to shut down and waits for them to terminate."""
        if not self.is_enabled:
            return

        logger.info("Shutting down OpenAI worker processes...")
        self.shutdown_event.set()
        for process in self.pool:
            process.join(timeout=5) # Wait for graceful shutdown
            if process.is_alive():
                logger.warning(f"Process {process.name} did not shut down gracefully, terminating.")
                process.terminate() # Force terminate if stuck
        logger.info("All OpenAI workers have been shut down.")

    def get_task_queue(self):
        return self.task_queue if self.is_enabled else None

    def get_results_dict(self):
        return self.results_dict if self.is_enabled else None
