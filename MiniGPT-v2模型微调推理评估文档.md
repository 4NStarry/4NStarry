# MiniGPT-v2 模型微调、推理与性能评估指南

本文档用于说明如何使用本项目完成 MiniGPT-v2 多模态模型的训练、部署推理和性能评估。当前项目的完整流程由 4 条核心命令依次完成：

1. 模型微调
2. Gradio 页面部署推理
3. VQA 问答任务评估
4. Grounding 目标定位任务评估

核心命令来源于：

```text
F:\ENVIRONMENTS\MiniGPT-4-main-evaluation\评估训练命令.txt
```

![SH展示](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260526151616.bmp "示例")

## 1. 项目准备

### 1.1 获取项目代码

```bash
git clone <ProjectEDAChat>
cd MiniGPT-4-main-evaluation
```

如果读者是直接下载压缩包，也需要进入项目根目录：

```bash
cd MiniGPT-4-main-evaluation
```

### 1.2 安装运行环境

推荐使用 Conda 创建环境：

```bash
conda env create -f gpt.yml
conda activate minigpt4
```

如果使用 pip：

```bash
pip install -r requirements_gpt.txt #在上一级目录
```

### 1.3 准备基础权重

本框架需要准备：

- LLaMA2 语言模型权重（基础权重且必须是单模态模型）
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260526154308.bmp "可用模型")
- MiniGPT-v2 预训练检查点（经第一二阶段训练的pth权重）
- 自定义训练数据集（光伏板电致发光图像集）

训练配置文件为：

```text
train_configs/minigptv2_finetune.yaml
```

需要重点修改以下字段：

```yaml
model:
  llama_model: "/path/to/Llama-2-7b-chat-hf"
  ckpt: "/path/to/checkpoint_stage3.pth"

run:
  output_dir: "/path/to/save/checkpoints"
```

字段说明：

| 字段 | 作用 |
| --- | --- |
| `model.llama_model` | LLaMA/LLaMA2 本地模型目录 |
| `model.ckpt` | MiniGPT-v2 初始检查点 |
| `run.output_dir` | 微调后 checkpoint 保存目录 |

> 截图建议：在本节后添加一张 `train_configs/minigptv2_finetune.yaml` 的截图，并用红框标出 `llama_model`、`ckpt`、`output_dir` 三个位置。建议图片命名为 `docs/images/train_config_paths.png`。

## 2. 数据集准备

### 2.1 训练数据格式

训练数据使用 `multitask_conversation` 数据集，配置文件为：

```text
minigpt4/configs/datasets/multitask_conversation/default.yaml
```

需要修改：

```yaml
datasets:
  multitask_conversation:
    data_type: images
    build_info:
      image_path: /path/to/train/images
      ann_path: /path/to/annotations_clean.json
```

其中：

| 字段 | 说明 |
| --- | --- |
| `image_path` | 训练图片所在目录 |
| `ann_path` | 训练标注 JSON 文件 |
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260526160630.bmp "JSON有中文有英文，LLAMA2别上中文")
训练标注文件示例（英文模型）：

```json
[
  {
    "id": "cell0004",
    "image": "cell0004.png",
    "conversations": [
      {
        "from": "question",
        "value": "[grounding]Has this solar photovoltaic module got any damage?"
      },
      {
        "from": "answer",
        "value": "No, in this picture, the photovoltaic module is defect-free."
      }
    ]
  }
]
```

说明：

- `image` 必须对应训练图片目录中的真实图片文件。
- `conversations` 中 `question` 为输入问题或任务指令。
- `conversations` 中 `answer` 为模型需要学习的目标答案。
- 如果是目标定位任务，答案可以使用 MiniGPT-v2 的框坐标格式，例如 `{<x1><y1><x2><y2>}`。
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527142645.bmp "电致发光图像数据集存放处")
![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527143138.bmp "中文模型JSON示例")

### 2.2 VQA 评估数据格式

VQA 评估配置文件为：

```text
eval_configs/minigptv2_benchmark_evaluation_VQA.yaml
```

当前项目使用 `vizwiz` 分支执行自定义 VQA 评估，需要修改：

```yaml
evaluation_datasets:
  vizwiz:
    eval_file_path: /path/to/damage_type_clean.json
    img_path: /path/to/val/images
    max_new_tokens: 512
    batch_size: 64
```

VQA 标注文件示例：

```json
[
  {
    "image": "cell0009.png",
    "question": "Has this solar photovoltaic module got any damage?",
    "answers": [
      {
        "answer": "defect-free",
        "answer_confidence": "yes"
      }
    ],
    "answer_type": "other",
    "answerable": 1
  }
]
```

![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527143609.bmp "EU_PVSEC数据集分类示例")

### 2.3 Grounding 目标定位评估数据格式

Grounding 评估配置文件为：

```text
eval_configs/minigptv2_benchmark_evaluation.yaml
```

需要修改：

```yaml
evaluation_datasets:
  refcoco:
    eval_file_path: /path/to/grounding/eval_root
    img_path: /path/to/val/images
    max_new_tokens: 1024
    batch_size: 64
```

脚本会在 `eval_file_path` 下读取类似如下结构的数据：

```text
eval_root/
└── refcoco/
    └── refcoco_testA.json
```

Grounding 标注文件示例：

```json
[
  {
    "img_id": "img000002",
    "sents": "裂缝",
    "bbox": [40, 256, 215, 368],
    "height": 1024,
    "width": 1024
  }
]
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `img_id` | 图像 ID |
| `sents` | 需要定位的目标描述 |
| `bbox` | 真实框，格式为 `[x, y, width, height]` |
| `height` | 原图高度 |
| `width` | 原图宽度 |

> ![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527143922.bmp "微观维度下的空间位置表示")

## 3. 步骤一：模型微调

确认数据路径、基础模型路径和 checkpoint 路径都配置正确后，执行：

```bash
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

如果使用多张 GPU，可以将 `--nproc-per-node` 改为 GPU 数量：

```bash
torchrun --nproc-per-node 4 train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

训练相关超参数位于：

```text
train_configs/minigptv2_finetune.yaml
```

重点字段：

```yaml
run:
  init_lr: 1e-5
  min_lr: 1e-6
  max_epoch: 50
  warmup_steps: 1000
  iters_per_epoch: 1000
  output_dir: "/path/to/save/checkpoints"
  wandb_log: True
```

训练完成后，模型权重会保存在 `run.output_dir` 指定目录下，常见文件名类似：

```text
checkpoint_49.pth
```

如果启用了 Weights & Biases，可以根据需要使用（W&B命令TXT第7-10行）：

```bash
wandb online
wandb offline
wandb login
```

![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527145032.bmp "training_terminal_log")

## 4. 步骤二：部署推理

微调完成后，需要将推理配置文件中的 `ckpt` 改为微调得到的权重。

推理配置文件为：

```text
eval_configs/minigptv2_eval.yaml
```

重点修改：

```yaml
model:
  ckpt: "/path/to/checkpoint_49.pth"
  lora_r: 64
  lora_alpha: 16
```

启动 Gradio 推理页面：

```bash
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
```

启动成功后，终端会输出本地访问地址，通常类似：

```text
Running on local URL: http://127.0.0.1:7860
```

在浏览器打开该地址后，可以上传图片并输入问题。

VQA 示例问题：

```text
Has this solar photovoltaic module got any damage?
```

Grounding 示例问题：

```text
[refer] give me the location of 裂缝
```

如果模型执行目标定位，可能返回类似：

```text
{<40><25><80><65>}
```

![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/微观.png "使用 EDAChat进行多轮交互的部署")

## 5. 步骤三：VQA 性能评估

VQA 评估命令：

```bash
torchrun --nproc_per_node 1 eval_vqa.py --cfg-path eval_configs/minigptv2_benchmark_evaluation_VQA.yaml --dataset vizwiz
```

执行前需要确认：

```yaml
model:
  llama_model: "/path/to/Llama-2-7b-chat-hf"
  ckpt: "/path/to/checkpoint_49.pth"

evaluation_datasets:
  vizwiz:
    eval_file_path: "/path/to/vqa/annotation.json"
    img_path: "/path/to/vqa/images"

run:
  save_path: "/path/to/save/evaluation/results"
```

评估流程：

1. 读取 VQA 标注文件。
2. 根据 `img_path` 加载验证图片。
3. 将问题转换为 MiniGPT-v2 VQA 指令。
4. 调用模型生成答案。
5. 将预测答案与真实答案进行匹配。
6. 在终端输出准确率。

主要输出文件：

```text
vizwiz.json
wrong_images.json
```

文件说明：

| 文件 | 说明 |
| --- | --- |
| `vizwiz.json` | 保存所有样本的模型预测答案 |
| `wrong_images.json` | 保存预测错误的图片名称，便于误差分析 |

终端会输出类似结果：

```text
vizwiz Acc: 85.0
```

![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527151329.bmp "JSON有中文有英文，LLAMA2别上中文")

## 6. 步骤四：Grounding 目标定位性能评估

Grounding 评估命令：

```bash
torchrun --nproc_per_node 1 eval_ref.py --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml --dataset refcoco,refcoco+,refcocog --resample
```

如果只评估一个数据集，可以执行：

```bash
torchrun --nproc_per_node 1 eval_ref.py --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml --dataset refcoco --resample
```

执行前需要确认：

```yaml
model:
  llama_model: "/path/to/Llama-2-7b-chat-hf"
  ckpt: "/path/to/checkpoint_49.pth"

evaluation_datasets:
  refcoco:
    eval_file_path: "/path/to/grounding/eval_root"
    img_path: "/path/to/grounding/images"

run:
  save_path: "/path/to/save/evaluation/results"
```

评估流程：

1. 读取 Grounding 标注文件。
2. 将 `sents` 字段转换为定位指令。
3. 模型生成 `{<x1><y1><x2><y2>}` 格式的预测框。
4. 脚本将模型输出坐标还原到原图尺寸。
5. 计算预测框与真实框之间的 EIoU。
6. 当 `EIoU > 0.25` 时，该样本计为定位正确。

`--resample` 的作用：

如果模型第一次输出的框格式不合法，脚本会进行多轮重新采样，以提高有效预测比例。

主要输出文件类似：

```text
['refcoco']_testA.json
```

终端会输出类似：

```text
refcoco testA: 72.5
```

![示例](https://raw.githubusercontent.com/4NStarry/4NStarry/main/20260527151555.bmp "grounding_prediction_visualization")

## 7. 四个核心命令汇总

### 7.1 模型微调

```bash
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

### 7.2 部署推理

```bash
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
```

### 7.3 VQA 评估

```bash
torchrun --nproc_per_node 1 eval_vqa.py --cfg-path eval_configs/minigptv2_benchmark_evaluation_VQA.yaml --dataset vizwiz
```

### 7.4 Grounding 评估

```bash
torchrun --nproc_per_node 1 eval_ref.py --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml --dataset refcoco,refcoco+,refcocog --resample
```