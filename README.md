<div align="center">

# ***Nemotron-Research-Tool-N1***: Tool-Using Language Models with Reinforced Reasoning

[![Report](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)](https://github.com/NVlabs/instant-ngp/blob/master/Tool-N1-report.pdf)
</div>

> - **We will release the code shortly in this repository, pending completion of NVIDIA's confidential review process.**

> - **More results will be released soon - stay tuned!**

We present Nemotron-Research-Tool-N1, a family of tool-using reasoning language models. We develop an R1-style RL training algorithm that employs a binary reward to supervise only the structural format and functional correctness of tool calls, without requiring explicit reasoning annotations. This approach enables models to generalize beyond strict token-level imitation and acquire reasoning capabilities directly from standard tool-calling data. We optimize the policy model using the GRPO algorithm. Experimental results on the BFCL and API-Bank benchmarks show that Nemotron-Research-Tool-N1-7B and Nemotron-Research-Tool-N1-14B, built upon Qwen2.5-7B/14B-Instruct, consistently outperform GPT-4o and other specialized open-source tool-calling models.

<p align="center">
<img src="./assets/overview.png" width="70%" alt="Overview" />
</p>

## Method

<p align="center">
<img src="./assets/thinking_template.png" width="70%" alt="thinking_template" />
</p>

- **Lightweight Reward Design:** Tool-N1 employs an R1-style binary reward that supervises only the structural validity and functional correctness of tool calls, without requiring detailed supervision of intermediate reasoning steps.

- **Reasoning Without Annotation:** The model is trained directly on existing tool-calling datasets without annotated reasoning trajectories. The model implicitly learns reasoning strategies through task success and format signal.

- **Flexible Supervision Mechanism:** Instead of strict string-level imitation (SFT), Rule-based reward could accommodate semantically equivalent tool calls with variations such as argument reordering, improving generalization beyond surface-level matching.

- **Optimization with GRPO:** Training is performed using the GRPO algorithm, which efficiently optimizes the model under the lightweight reward structure, leading to stable and effective policy learning.

## Empirical Results

<img src="./assets/exp_main.png" width="99%" alt="exp_main" />

We mainly perform evaluations on the BFCL and API Bank.

- RL offers a more effective paradigm for enhancing the tool-calling capabilities of LLMs compared to standard supervised fine-tuning.

- Performance improvements from post-training are limited for smaller models (0.5B and 1.5B), whereas larger models exhibit substantial gains.

- Qwen2.5-Instruct outperforms both LLaMA variants at the same model scale after training with the proposed method.

## Reward Designing and Training Data Composition

**RL Reward Designing:** Ablation study on reward granularity. We compare fine-grained reward designs, where partial credit is given for correct reasoning format and correct function names in function call stages, with binary rewards that assign full reward only when all conditions are fully satisfied. The results show that binary rewards consistently yield better performance, especially in the Live setting.

<img src="./assets/reward.png" width="80%" alt="reward" />

**Training Data Composition:** (1) ToolACE data yields particularly strong improvements in the live setting. (2) Compared to models trained using SFT on the same data, the R1-style training consistently yields better performance. Specifically, the Tool-N1-7B model trained solely on xLAM data outperforms the xLAM-8B SFT model by 6.36%, and the Tool-N1-7B model trained solely on the ToolACE subset exceeds the ToolACE-8B SFT model by 1.62%, despite using only a subset of the data.

<img src="./assets/data_composition.png" width="80%" alt="data_composition" />
