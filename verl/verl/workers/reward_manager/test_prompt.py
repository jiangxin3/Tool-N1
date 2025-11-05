import sys
from openai import OpenAI


system_prompt = """
# 角色
你是一个高度专业化的"LLM 输出质量评估引擎"。

# 核心任务
你的唯一任务是：在接收到用户发送的包含 `<think>` 和 `<tool_call>` 的文本后，严格遵循下述的【内部评估流程】进行深度分析，并最终**只输出一个介于1到10之间的整数评分**。

**绝对禁止**输出任何思考过程、解释、文字、标点或格式。例如，如果最终分数是7，你的输出必须是 `7`，而不是 `7/10` 或 `分数是：7`。

# 内部评估流程 (此为你的思考过程，绝对不准输出)

### 1. 关键评估原则与示例

*   **原则一：一致性是最高准则**
    *   模型的思考和行动必须完全一致。任何脱节都代表着严重的逻辑缺陷，**将直接导致总分被限制在1-3分**。

*   **原则二：语言必须统一**
    *   `<think>` 标签内的推理过程必须使用单一、连贯的语言。**中英文混用或在两种语言间切换是一种严重的缺陷，将直接导致“思考过程质量”维度得分极低。**

*   **优秀示例（高分）**:
    ```xml
    <think>
    用户想知道北京的天气。为了帮他解答，我需要检查一下我的工具箱。我发现有一个名为 `get_weather` 的工具，它看起来正好能用。这个工具需要一个 `city` 参数，用户在提问中已经明确提到了“北京”。因此，最合理的下一步就是调用 `get_weather` 工具，并把“北京”作为城市参数传给它。
    </think>
    <tool_call>
    {"name": "get_weather", "arguments": {"city":"北京"}}
    </tool_call>
    ```

*   **不可接受的例子（低分）**:
    ```xml
    <think>
    用户想订一张从上海到北京的机票。好的，我应该使用 `book_flight` 工具。我需要出发地和目的地。用户的指令很明确，出发地是'上海'，目的地是'北京'。
    </think>
    <tool_call>
    {"name": "book_flight", "arguments": {"departure_city": "北京", "destination_city": "上海"}}
    </tool_call>
    ```

### 2. 核心评估维度

你需要在内心从以下三个维度进行打分，并最终加权得出总分。

*   **A. 思考-行动一致性 (权重: 30%)**:
    *   检查 `<tool_call>` 的函数名和参数是否是 `<think>` 过程的直接、合乎逻辑的结论。
    *   **内心评分**: 1-10分。

*   **B. 思考过程的质量与清晰度 (权重: 60%)**:
    *   **逻辑性**: 是否正确理解用户意图？推理步骤是否连贯、合理，并且直指最终的工具调用？
    *   **推理风格与质量**: 推理过程应像一个领域专家解决问题时的内心独白，而不是一个程序在打印调试日志。基于此，对以下行为进行**严厉惩罚**：
        *   **禁止元认知描述 (Meta-Commentary)**: 思考过程应专注于 **“做什么”** 和 **“为什么做”**，而不是描述其自身的思考步骤。严厉惩罚任何出现“响应规则”、“参数设置”、“最终响应”、“确认函数调用”、“响应格式”等描述生成过程的词语。
        *   **禁止模板化与冗余**: 推理应自然、直截了当。严厉惩罚使用“回顾工具描述”、“检查调用规范”等机械短语，以及对同一结论的反复确认。**尤其禁止在思考的结尾处复述最终的`tool_call`内容。**
    *   **完整性与正确性**: 思考过程的文本必须是完整的句子，**没有中途截断**。**不得包含任何拼写错误或明显的语法错误**。
    *   **语言纯粹性**: **是否全程使用单一语言？出现中英混用或切换则此项得分极低。**
    *   **内心评分**: 1-10分。

*   **C. 工具调用有效性 (权重: 10%)**:
    *   检查 `<tool_call>` 本身的JSON格式是否正确，函数名和参数名是否存在拼写错误，参数值是否符合常识和逻辑。
    *   **内心评分**: 1-10分。

### 3. 计算最终分数

*   在内心计算加权总分：`总分 = (A * 0.3) + (B * 0.6) + (C * 0.1)`。
*   将计算出的总分进行四舍五入，得到最终的整数。

# 输出规则 (必须无条件遵守)
-   你的最终响应**必须且只能是**一个阿拉伯数字（1, 2, 3, 4, 5, 6, 7, 8, 9, 10）。
-   **不包含**任何前缀或后缀。
-   **不包含**任何文字解释。
-   **不包含**任何多余的空格或换行。

# 工作流程
1.  在我发送此条指令后，**不要回复任何确认信息**，直接进入待命状态。
2.  当我发送需要评估的文本后，你将立即执行【内部评估流程】。
3.  完成评估和计算后，立即输出那个最终的整数。
"""

prompt = """
<think>

好的，用户现在说“支持即插即充”。首先，我需要分析用户的指令意图。用户之前在搜索充电站，现在添加了一个新的偏好：支持即插即充。这属于补能相关的偏好信息。

回顾提供的工具：有“navi_power_filter”函数用于处理补能相关的偏好。具体来说，这个函数的参数包括“charging_preference”，其中包含各种偏好选项，如“is_plug_charging”，表示是否支持即插即充。

检查“navi_power_filter”函数的描述：它用于获取用户查询中提到的补能相关偏好。参数中，“is_plug_charging”是一个布尔型参数，枚举值为true表示支持即插即充，false表示不支持。

用户说“支持即插即充”，这正好匹配“is_plug_charging”参数的true值。因此，应该调用“navi_power_filter”函数。

响应规则：当指令含义清晰且可以使用已有工具完成时，必须输出JSON格式的工具调用指令。响应必须包含函数名和arguments。对于“navi_power_filter”，函数名是“navi_power_filter”，参数包括“charging”（默认["充电"]）和“charging_preference”。

参数设置：“charging_preference”中的“is_plug_charging”设为true。

确认函数调用：确认函数名称和参数正确。函数名是“navi_power_filter”，参数是{"charging": ["充电"], "charging_preference": {"is_plug_charging": true}}。

响应格式：必须输出JSON对象，包含"name"和"arguments"。因此，响应应该是一个JSON对象：{"name": "navi_power_filter", "arguments": {"charging": ["充电"], "charging_preference": {"is_plug_charging": true}}}

最终响应：{"name": "navi_power_filter", "arguments": {"charging": ["充电"], "charging_preference": {"is_plug_charging": true}}}

</think>

<tool_call>
{"name": "navi_power_filter", "arguments": {"charging": ["充电"], "charging_preference": {"is_plug_charging": true}}}
</tool_call>
"""
# 初始化客户端
client = OpenAI(
    api_key="bce-v3/ALTAK-F0MuZLYwbxPXi6hqB5yfJ/590457ba8f59e1902197631dcb7dae3c36fd01a5",  # 你的授权令牌
    base_url="https://qianfan.baidubce.com/v2"  # 千帆 API 基础地址
)

# 发送请求
response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
)

# 打印响应
print(response.choices[0].message.content)