# `/public/home/aco7liy2y0/4N` 工作目录说明（协作版）

本文用于帮助后续成员快速理解远程工作目录`/public/home/aco7liy2y0/4N`的结构、关键文件和推荐接手路径。

## 1. 顶层结构（精简）

```text
/public/home/aco7liy2y0/4N
├─ 能源/                         # 当前最核心业务目录（多模态数据、脚本、训练输入）
├─ diffusers/                    # 训练与推理代码主仓（含自定义脚本）
├─ stable-diffusion-3.5-large/   # SD3.5 Large 本地模型目录
├─ stable-diffusion-xl-base-1.0/ # SDXL Base 本地模型目录
├─ 3.5Lcontrolnets/              # SD3.5 ControlNet 相关资源
├─ comfy/                        # ComfyUI 相关目录
├─ InternVL3_5-14B-Instruct/     # InternVL 相关训练输出/运行目录
├─ sd3.5-t2i-fp16-workflow.json  # ComfyUI/工作流文件
├─ Miniconda3-*.sh               # 环境安装脚本
└─ cudatoolkit-11.8.0-*.conda    # CUDA toolkit 离线包
```

## 2. 关键目录与用途说明

### 2.1 `能源/`（优先级最高）

该目录包含当前任务最关键的数据与处理脚本：

- 数据集与标注相关：
  - `swift.json`
  - `swift_grounding.json`
  - `sharegpt_qwen.json`
  - `sharegpt_grounding.json`
  - `train/`, `val/`, `data/`, `val_label/`, `train_separated_label/`, `多任务统一数据集/`
- 处理/转换脚本：
  - `format.py`
  - `filter.py`
  - `multi.py`
  - `multi_grounding.py`
  - `qwen2glm.py`
  - `翻译.py`
  - `ＪＳＯＮ多ＢＢＯＸ.py`
- 训练/推理中间产物：
  - `20250909-170639.jsonl`
  - `20250910-163038.jsonl`
  - `20250910-173209.jsonl`
  - `checkpoint_399.pth`

建议：后续人员先从`swift.json`和`format.py`读起，先确定数据格式，再看训练脚本调用链。

### 2.2 `diffusers/`（代码主仓）

该目录是训练与推理核心代码来源，包含：

- 配置与启动脚本：
  - `accelerate_config.yaml`
  - `accelerate_zero_offload_bf16.yaml`
  - `ds_config.json`
  - `train_sd3_5_de_lora.sh`
  - `train_sd3_5_ip2p_lora.sh`
  - `infer_batch.sh`
- 训练/推理 Python 脚本：
  - `train_sd3_5C.py`
  - `train_sd3_5DiffEditC.py`
  - `train_sd3_5DiffEditGPT.py`
  - `train_sd3_5GPT.py`
  - `train_instruct_pix2pix_sdxl.py`
  - `inference_batch.py`
  - `batch_inference.py`
- 数据准备脚本：
  - `create_dataset.py`
  - `create_parquet.py`
  - `generate_json.py`

建议：优先查看`.sh`入口脚本，再追踪对应`.py`文件，能最快还原训练流程。

### 2.3 `stable-diffusion-3.5-large/` 与 `stable-diffusion-xl-base-1.0/`

这两处是本地基础模型目录，主要用于训练或推理时加载权重：

- 关键文件：
  - `model_index.json`
  - `*.safetensors`
  - `README.md`
  - `LICENSE.md`
- 子目录（编码器/UNet/VAE/Tokenizer/Scheduler）完整，说明可本地离线加载。

### 2.4 `3.5Lcontrolnets/`

ControlNet 相关资源目录，包含：

- `config.json`, `configuration.json`
- `README.md`
- 示例图片：`sample_result.png`, `canny_header.png`

可作为 ControlNet 条件输入流程参考。

## 3. 建议的接手顺序

1. 先读`能源/`：明确任务数据格式和业务目标。  
2. 再读`diffusers/`：确认训练命令入口和参数。  
3. 最后核对模型目录：确认本地权重与脚本中的模型路径一致。  

## 4. 对后续协作者的操作建议

1. 统一新增脚本命名（如`train_xxx.sh`、`infer_xxx.sh`）。  
2. 将实验输出集中到固定目录（例如`/public/home/aco7liy2y0/output`）。  
3. 在`能源/`补充一个`README.md`，明确每个 JSON 文件用途与字段规范。  
4. 对关键流程（数据转换、训练、推理）增加最小可复现实验命令。  

## 5. 建议补充到 GitHub 的截图

1. `4N`顶层目录截图（让读者建立全局结构认知）。  
2. `能源/`目录截图（突出`swift.json`与处理脚本）。  
3. `diffusers/`目录截图（突出训练入口`.sh`与核心`.py`）。  
4. 一次训练日志截图（显示step/loss/保存checkpoint）。  
5. 一次推理结果截图（输入图 + 输出图/文本）。  

