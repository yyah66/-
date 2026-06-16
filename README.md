# 基于视觉语言大模型的中文图文问答助手

这是一个面向课程设计的中文图文问答原型，覆盖上传图片、中文提问、多轮对话、OCR证据增强、模型调用和结果展示。

## 功能

- 支持上传自然场景图片、文档截图、课件、表格和图表
- 支持中文多轮问答
- 自动提取 OCR 文本并作为证据增强提示
- 默认优先使用 PaddleOCR，提升中文截图和表格类图片的识别效果
- 支持 DashScope 的 OpenAI 兼容接口，也支持无 API Key 时的本地兜底模式
- 输出答案、证据、置信度和 OCR 片段，便于展示可追溯结果

## 目录

- `app.py`：Streamlit 前端入口
- `src/assistant.py`：问答编排与多轮对话状态管理
- `src/model.py`：视觉语言模型调用与兜底逻辑
- `src/ocr.py`：OCR 提取逻辑
- `src/preprocess.py`：尺寸归一、旋转校正与预处理摘要
- `src/prompt.py`：提示词构建
- `scripts/evaluate_dataset.py`：批量评测脚本
- `scripts/compare_models.py`：多模型对照评测脚本
- `scripts/lora_train.py`：LoRA 训练数据准备与训练建议脚本
- `scripts/prepare_chartqa_lora_package.py`：把本地 ChartQA 仓库整理成训练包
- `scripts/train_qwen2vl_lora.py`：ChartQA 的可运行 LoRA 训练入口
- `scripts/run_lora_training.ps1`：训练前检查与命令模板
- `scripts/run_chartqa_lora_training.ps1`：一键启动 ChartQA LoRA 训练
- `scripts/prepare_public_datasets.py`：公开数据集抽样与 JSONL 导出脚本

## 运行方式

### 前置步骤：修改 .env 配置

无论本地还是 AutoDL 运行，都请先编辑项目根目录下的 `.env` 文件，按实际环境修改以下配置：

```bash
# 模型路径：改为你自己的模型存放路径
LOCAL_MODEL_PATH=/your/path/to/Qwen2.5-VL-7B-Instruct

# 模型名称：与你下载的模型一致
VLM_MODEL=Qwen2.5-VL-7B-Instruct

# LoRA 适配器路径（可选，留空则使用基座模型）
LORA_PATH=outputs/lora/merged_ocr/final_adapter
```

### 本地运行

1. 安装依赖

```bash
pip install -r requirements.txt

# 本地 GPU 推理额外依赖（如使用 local 模式）
pip install transformers torch torchvision accelerate bitsandbytes peft
```

2. 启动应用

```bash
streamlit run app.py
```

3. 如果你有 DashScope API Key，可将 `.env` 中 `VLM_PROVIDER` 改为 `dashscope` 以启用云端推理。

### AutoDL 运行

1. 将项目上传至 `/root/autodl-tmp/` 目录

2. 安装依赖

```bash
cd /root/autodl-tmp/--main
pip install -r requirements.txt
pip install transformers torch torchvision accelerate bitsandbytes peft
```

3. 下载模型（如尚未下载）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir /root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct
```

4. 启动应用

```bash
streamlit run app.py \
  --server.port 6006 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

5. 通过 AutoDL「自定义服务」按钮访问，或直接访问 `http://<实例公网IP>:6006`。

> 如果需要在终端关闭后保持运行，使用 nohup：
> ```bash
> nohup streamlit run app.py --server.port 6006 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false > streamlit.log 2>&1 &
> ```

## 关键环境变量

- `VLM_PROVIDER`：`heuristic` 或 `dashscope`
- `DASHSCOPE_API_KEY`：DashScope API Key
- `VLM_BASE_URL`：OpenAI 兼容接口地址，默认使用 DashScope 兼容模式
- `VLM_MODEL`：模型名称，默认 `qwen2.5-vl-7b-instruct`
- `MAX_OCR_LINES`：拼接到提示词中的 OCR 行数上限
- `MAX_HISTORY_TURNS`：保留的历史轮次上限
- `PADDLEOCR_LANG`：PaddleOCR 语言，默认 `ch`

## OCR 后端优先级

当前默认优先级为 `rapidocr-onnxruntime` -> `paddleocr` -> `pytesseract`。在当前 Windows + Python 3.13 环境里，RapidOCR 是最稳定的可用后端；当它不可用时，系统会继续回退到后续后端，保证原型仍可运行。

## 评测思路

可以准备一个 JSONL 数据集，每行包含 `image_path`、`question`、`answer` 和可选的 `evidence` 字段，然后运行：

```bash
python scripts/evaluate_dataset.py --input your_dataset.jsonl --output results.jsonl
```

脚本会额外生成一个同名的 `results.summary.json`，里面包含样本数、Exact Match、平均重叠分数和失败类型统计，适合直接写进课程报告的实验分析部分。

## 说明

当前实现偏向课程项目落地，优先保证完整流程和可演示性。若后续需要进一步提升效果，可以把 OCR 后端切换为 PaddleOCR，并补充 LoRA 微调流程和更系统的定量评测。

## 演示样例

`data/demo/` 下提供了三类合成样例：文档、表格和图表。启动 `streamlit run app.py` 后，可以在侧边栏直接一键加载样例进行演示，也可以用 `scripts/evaluate_dataset.py` 对 `data/demo/demo_dataset.jsonl` 做快速评测。

## 预处理与对照实验

- 预处理包括 EXIF 方向修正、旋转校正、尺寸归一和 OCR 前的统一输入整理。
- 多模型对照可以运行 `scripts/compare_models.py`，对同一数据集比较 Qwen2.5-VL、Qwen3-VL 或 GLM-4V 等模型。
- LoRA 微调先用 `scripts/lora_train.py` 把标注数据转换成 SFT JSONL，再接入你自己的训练环境完成训练。

## 训练前准备

本地 ChartQA 已经拉到 [data/public/ChartQA](data/public/ChartQA)。你现在可以直接运行：

```bash
E:/py/python.exe scripts/prepare_chartqa_lora_package.py --dataset-root data/public/ChartQA --output-dir data/training/chartqa --limit-per-split 2000
```

这一步会生成：
- `data/training/chartqa/chartqa_train.jsonl`
- `data/training/chartqa/chartqa_val.jsonl`
- `data/training/chartqa/chartqa_package.json`

随后可以用 `scripts/run_lora_training.ps1` 查看训练前检查和命令模板。

如果要直接开跑 ChartQA LoRA，可以先安装依赖，再执行：

```bash
.\.venv\Scripts\python.exe scripts/train_qwen2vl_lora.py --train-jsonl data/training/chartqa/chartqa_train.jsonl --val-jsonl data/training/chartqa/chartqa_val.jsonl --image-root . --output-dir outputs/lora/chartqa
```

或者在 PowerShell 里运行 `scripts/run_chartqa_lora_training.ps1`。

## 公开数据集快速获取

你提到的 4 个公开数据集可以直接按下面的方式抽样成项目可用的 JSONL：

```bash
python scripts/prepare_public_datasets.py --dataset vqa_v2 --split train --limit 200 --output-dir data/public/vqa_v2
python scripts/prepare_public_datasets.py --dataset textvqa --split train --limit 200 --output-dir data/public/textvqa
python scripts/prepare_public_datasets.py --dataset docvqa --split train --limit 200 --output-dir data/public/docvqa
```

ChartQA 官方仓库一般需要先把 `ChartQA Dataset.zip` 解压到本地，再准备一个 JSON/JSONL 清单，然后运行：

```bash
python scripts/prepare_public_datasets.py --dataset chartqa --manifest path/to/chartqa_manifest.jsonl --image-root path/to/chartqa_images --limit 200 --output-dir data/public/chartqa
```

脚本会把样本统一导出成课程项目能直接使用的 JSONL 格式，并把图片复制到输出目录下，便于后续评测和演示。

## 最简下载方案

如果你的网络环境可以直连 Hugging Face，可以优先用下面这套最省事的方式先取小样本做课程项目：

- VQA-v2：`load_dataset("HuggingFaceM4/VQAv2", split="train[:500]")`
- TextVQA：`load_dataset("textvqa", split="train[:500]")`
- DocVQA：`load_dataset("nielsr/docvqa_1200_examples")`
- ChartQA：直接从 [ChartQA 官方仓库](https://github.com/vis-nlp/ChartQA) 下载 `ChartQA Dataset.zip`

对应到本项目里，可以直接改成 200 到 500 条的导出命令，用于课程实验和答辩演示。如果 Hugging Face 因为证书或网络问题拉不下来，就改用本仓库里的 `scripts/prepare_public_datasets.py` 做本地导出，或者先用 `data/demo/demo_dataset.jsonl` 的合成样例跑通流程。
