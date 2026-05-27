# 基于 ms-swift 的多模态模型微调与推理实战（Qwen2.5-VL / InternVL3.5）

本文基于我在 HPC 环境中的实际训练产物整理，目标是帮助读者复现「数据准备 -> LoRA 微调 -> 推理验证 -> 部署调用」完整流程。

## 1. 实际环境与产物路径（超算互联网SCNet昆山服务器）

- 远程主机：`cancon.hpccube.com:65023`
- 用户：`aco7liy2y0`
- 训练数据：`/public/home/aco7liy2y0/4N/能源/swift.json`
- 输出目录：`/public/home/aco7liy2y0/output`

当前已确认的关键产物：

- Qwen 路径：`/public/home/aco7liy2y0/output/Qwen2.5-VL-7B-Instruct/v4-20250918-125322/checkpoint-294`
- InternVL 路径：`/public/home/aco7liy2y0/output/checkpoint-4704`（其 `args.json` 对应 `OpenGVLab/InternVL3_5-14B-Instruct`）
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527161349.bmp "可用模型")
## 2. 环境准备

建议使用 Python 3.10+，并在虚拟环境中安装：

```bash
pip install "ms-swift[all]" -U
pip install modelscope
```

如果你走 Hugging Face 权重，也可额外安装：

```bash
pip install transformers accelerate
```
![LOG](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527160653.bmp "训练启动命令")
## 3. 数据准备（swift.json）

你的数据文件为：`/public/home/aco7liy2y0/4N/能源/swift.json`。  
每条样本包含：图像路径 + 指令 + 答案（按 ms-swift 多模态数据格式组织）。

最低要求：

1. 图像路径可在训练机访问；
2. 指令与答案字段非空；
3. 先抽样 20~50 条做小规模试跑，确认模板与字段匹配。
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527160523.bmp "数据样例")
## 4. LoRA 微调（Qwen2.5-VL）

你当前产物对应的核心参数：

- 基座：`Qwen/Qwen2.5-VL-7B-Instruct`
- `train_type=lora`
- `lora_rank=8`、`lora_alpha=32`、`lora_dropout=0.05`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`
- `num_train_epochs=3`
- `learning_rate=1e-4`
- `bf16=true`
- `deepspeed stage=1`

示例命令（按你的配置风格）：

```bash
swift sft \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model_type qwen2_5_vl \
  --template qwen2_5_vl \
  --dataset /public/home/aco7liy2y0/4N/能源/swift.json \
  --train_type lora \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --save_steps 500 \
  --logging_steps 5 \
  --bf16 true \
  --output_dir /public/home/aco7liy2y0/output/Qwen2.5-VL-7B-Instruct/your_run_name
```
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527160748.bmp "训练终端日志")
## 5. LoRA 微调（InternVL3.5）

你当前产物对应的核心参数：

- 基座：`OpenGVLab/InternVL3_5-14B-Instruct`
- `train_type=lora`
- `lora_rank=8`、`lora_alpha=32`、`lora_dropout=0.05`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1`
- `num_train_epochs=3`
- `learning_rate=1e-4`
- `bf16=true`
- `deepspeed stage=1`

示例命令：

```bash
swift sft \
  --model OpenGVLab/InternVL3_5-14B-Instruct \
  --model_type internvl3_5 \
  --template internvl3_5 \
  --dataset /public/home/aco7liy2y0/4N/能源/swift.json \
  --train_type lora \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --save_steps 500 \
  --logging_steps 5 \
  --bf16 true \
  --output_dir /public/home/aco7liy2y0/output/InternVL3_5-14B-Instruct/your_run_name
```
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527160918.bmp "训练过程")
## 6. 推理验证（LoRA 权重）

典型方式是指定 `--ckpt_dir` 指向 LoRA checkpoint，例如：

```bash
swift infer \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model_type qwen2_5_vl \
  --template qwen2_5_vl \
  --ckpt_dir /public/home/aco7liy2y0/output/Qwen2.5-VL-7B-Instruct/v4-20250918-125322/checkpoint-294
```
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/一般（QWEN）.bmp "可用模型")
InternVL 同理：

```bash
swift infer \
  --model OpenGVLab/InternVL3_5-14B-Instruct \
  --model_type internvl3_5 \
  --template internvl3_5 \
  --ckpt_dir /public/home/aco7liy2y0/output/checkpoint-4704
```
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/一般(INTERNVL).bmp "可用模型")
你现有的推理/部署结果样例目录：

- `.../infer_result/*.jsonl`
- `.../deploy_result/*.jsonl`

## 7. 部署与服务化（示例）

若采用 ms-swift 的部署命令，核心思想是加载基座 + LoRA checkpoint 暴露接口。  
执行后先用本地/远程请求压测，再固化到脚本（你已有 `deploy.sh` 可作为入口）。

建议在仓库中提供：

1. `train_qwen.sh`
2. `train_internvl.sh`
3. `infer_qwen.sh`
4. `infer_internvl.sh`
5. `deploy.sh`

## 8. 目录结构

```text
output/
├─ Qwen2.5-VL-7B-Instruct/
│  ├─ v4-20250918-125322/
│  │  ├─ args.json
│  │  ├─ checkpoint-294/
│  │  │  ├─ adapter_config.json
│  │  │  └─ adapter_model.safetensors
│  │  └─ runs/
│  └─ infer_result/
└─ checkpoint-4704/   # InternVL3.5 LoRA checkpoint
   ├─ args.json
   ├─ adapter_config.json
   ├─ adapter_model.safetensors
   ├─ infer_result/
   └─ deploy_result/
```
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527161349.bmp "可用模型")

## 常见问题

1. 显存不足：降低 `max_length`、减小 batch、提高梯度累积。
2. 推理效果不稳定：先核对训练/推理 `template` 与 `model_type` 是否一致。
3. 找不到数据：先在训练机 `ls` 验证 `swift.json` 与图像路径。
4. 部署输出异常：优先检查 `ckpt_dir` 是否指向包含 `adapter_model.safetensors` 的目录。

