<div align="center">

# ***Nemotron-Research-Tool-N1***: Exploring Tool-Using Language Models with Reinforced Reasoning

[![Arxiv](https://img.shields.io/badge/paper-A82F27?style=for-the-badge&logo=arxiv)](https://arxiv.org/pdf/2505.00024)
</div>

> [!IMPORTANT]
> - **Please consider giving us a ⭐️ to stay updated on the upcoming code release!**

We present Nemotron-Research-Tool-N1, a family of tool-using reasoning language models. These models are trained with an R1-style reinforcement learning algorithm that uses a binary reward to supervise only the structural format and functional correctness of tool calls, without requiring explicit reasoning annotations. This allows the models to generalize beyond token-level imitation and acquire reasoning capabilities directly from standard tool-calling data. The policy is optimized using GRPO.
Experimental results on the BFCL, API-Bank, and AceBench benchmarks show that Tool-N1-7B and 14B, built on Qwen2.5-7B/14B-Instruct, significantly outperform GPT-4o and other leading open-source tool-calling models.
Additionally, we conduct a systematic study of rule-based RL strategies. Using 5,518 distilled reasoning trajectories, we compare pure RL, supervised fine-tuning (SFT), and the commonly used SFT-then-RL pipeline. Our analysis reveals that the SFT-then-RL approach does not consistently pure RL.

<p align="center">
<img src="./assets/overview.png" width="100%" alt="Overview" />
</p>

## How to Run

### Data Process & Environment

The ```data_process``` folder contains the script for initial preprocessing of the Huggingface datasets. To run the script, use the following commands:

```
cd data_process
python data_process.py
```

Please specify the paths to the downloaded ToolACE and xLAM datasets in the script before execution.

```
# verl
cd verl
pip3 install verl[vllm]

# LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```


### RL Training 

First ```cd verl``` in the begining.


- **Convert Raw Data to Verl Data**

```
cd examples/data_preprocess
python toolcall_preprocess.py
```

- **Start Training**

```
bash train_qwen.sh
```

- **Model Convert**

```
python model_convert.py
```

### Reasoning Data Distillition 

```
cd data_process
python distill_data.py
```

### SFT Training 

First ```cd LLaMA-Factory```.

- **Data Process**

```
python data_process.py
```


- **Model Training**

```
export PROJ_NAME="sft"
export OUTPUT_DIR="saves/qwen-7b/sft"

export MODEL_PATH="path/to/model/Qwen2.5-7B-Instruct"
export LOG_DIR="path/to/logs/$PROJ_NAME.txt"
export LR=2.0e-5
export EPOCH=20
export BATCH_SIZE=4
export G_ACC=8

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen_tool_call50.yaml \
    learning_rate=$LR \
    num_train_epochs=$EPOCH \
    per_device_train_batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$G_ACC \
    output_dir=$OUTPUT_DIR \
    run_name=$PROJ_NAME \
    model_name_or_path=$MODEL_PATH 2>&1 | tee -a "${LOG_DIR}"
```

- **Model Merge**

```
llamafactory-cli export examples/qwen_merge.yaml
```

### Evaluation 

Please find the BFCL model handler in ```eval``` folder.

## Method

<p align="center">
<img src="./assets/thinking_template.png" width="100%" alt="thinking_template" />
</p>

- **Lightweight Reward Design:** Nemotron-Research-Tool-N1 employs an R1-style binary reward that supervises only the structural validity and functional correctness of tool calls, without requiring detailed supervision of intermediate reasoning steps.

- **Reasoning Without Annotation:** The model is trained directly on existing tool-calling datasets without annotated reasoning trajectories. The model implicitly learns reasoning strategies through task success and format signal.

- **Flexible Supervision Mechanism:** Instead of strict string-level imitation (SFT), Rule-based reward could accommodate semantically equivalent tool calls with variations such as argument reordering, improving generalization beyond surface-level matching.

- **Optimization with GRPO:** Training is performed using the GRPO algorithm, which efficiently optimizes the model under the lightweight reward structure, leading to stable and effective policy learning.

## Empirical Results

<img src="./assets/exp_main.png" width="99%" alt="exp_main" />

We mainly perform evaluations on the BFCL, APIBank and ACEBench.

- RL offers a more effective paradigm for enhancing the tool-calling capabilities of LLMs compared to standard supervised fine-tuning.

- Performance improvements from post-training are limited for smaller models (0.5B and 1.5B), whereas larger models exhibit substantial gains.

- Qwen2.5-Instruct outperforms both LLaMA variants at the same model scale after training with the proposed method.

## Deep Analysis

**RL Reward Designing:** Ablation study on reward granularity. We compare fine-grained reward designs, where partial credit is given for correct reasoning format and correct function names in function call stages, with binary rewards that assign full reward only when all conditions are fully satisfied. The results show that binary rewards consistently yield better performance, especially in the Live setting.

<img src="./assets/reward.png" width="80%" alt="reward" />

**Training Data Composition:** (1) ToolACE data yields particularly strong improvements in the live setting. (2) Compared to models trained using SFT on the same data, the R1-style training consistently yields better performance. Specifically, the Tool-N1-7B model trained solely on xLAM data outperforms the xLAM-8B SFT model by 6.36%, and the Tool-N1-7B model trained solely on the ToolACE subset exceeds the ToolACE-8B SFT model by 1.62%, despite using only a subset of the data.

<img src="./assets/data_composition.png" width="65%" alt="data_composition" />


**SFT or RL?**  (1) Although the combination of SFT on reasoning trajectories followed by RL is commonly regarded as the best practice in many domains, we do not observe improved performance under equal data budgets in the tool-calling setting.
(2) Pure RL outperforms both Reason-SFT and No-Reason SFT under equal data budgets.
(3) Interestingly, No-Reason SFT performs only slightly worse than Reason-SFT, suggesting that providing reasoning traces during SFT offers limited additional benefit.

<img src="./assets/sft_or_rl.png" width="80%" alt="data_composition" />

## Citation
```md
@article{zhang2025nemotron,
  title={Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning},
  author={Zhang, Shaokun and Dong, Yi and Zhang, Jieyu and Kautz, Jan and Catanzaro, Bryan and Tao, Andrew and Wu, Qingyun and Yu, Zhiding and Liu, Guilin},
  journal={arXiv preprint arXiv:2505.00024},
  year={2025}
}
```