# Sa2VA 训练与预测完整指南

## 📚 目录

- [1. 概述](#1-概述)
- [2. 使用的数据集](#2-使用的数据集)
- [3. 使用的权重](#3-使用的权重)
- [4. 环境准备](#4-环境准备)
- [5. 如何开始训练](#5-如何开始训练)
- [6. 如何开始预测](#6-如何开始预测)
- [7. 模型转换](#7-模型转换)
- [8. 常见问题](#8-常见问题)

---

## 1. 概述

### Sa2VA是什么？

**Sa2VA (SAM2 + Vision-Language Assistant)** 是一个结合了SAM2分割模型和大型多模态语言模型（MLLM）的统一框架，用于图像和视频的密集理解与分割任务。

### 本项目的应用场景

本项目专注于 **OCT血管分割任务**，使用Sa2VA-InternVL3-8B模型在医学OCT（光学相干断层扫描）血管图像上进行fine-tune。

### 技术架构

```
Sa2VA-8B 模型架构
├── Vision Encoder (InternVL3-8B)    ← 冻结，提取图像特征
├── Language Model (8B)              ← LoRA微调，理解指令
├── Projector                        ← 视觉-语言特征对齐
└── SAM2 Decoder                     ← 可训练，生成分割mask
```

---

## 2. 使用的数据集

### 📊 OCT血管分割数据集

#### 数据集位置

```
/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/
├── images/          ← OCT血管图像（512x512）
├── masks/           ← 分割标注mask
├── json/            ← 标注信息（多边形）
├── annotations.json ← 数据集索引文件
└── README.md        ← 数据集说明
```

#### 数据集统计

| 指标 | 数值 |
|------|------|
| **图像数量** | 1,220张 |
| **图像尺寸** | 512 × 512 像素 |
| **图像格式** | JPG |
| **标注格式** | JSON (多边形坐标) |
| **总大小** | ~194MB |

#### 数据集结构

`annotations.json` 文件格式：

```json
[
  {
    "image": "images/patient_name_frame_000001.jpg",
    "mask": [
      {
        "segmentation": [[x1, y1, x2, y2, ...]], // 多边形坐标
        "area": 12345,
        "bbox": [x, y, width, height],
        "category_id": 1
      }
    ],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease segment the blood vessels."
      },
      {
        "from": "gpt",
        "value": "<p>blood vessels</p><vp>[[x1,y1,x2,y2,...]]</vp>[SEG]"
      }
    ]
  }
]
```

#### 数据准备

数据集已存放在HuggingFace：

```bash
# 下载数据集
huggingface-cli download \
    ly17/sa2va-vessel-dataset \
    --local-dir Segment_DATA_Merged_512 \
    --repo-type dataset
```

#### 数据预处理

数据集已经过以下预处理：
- ✅ 图像resize到512×512
- ✅ 标注转换为多边形格式
- ✅ 创建annotations.json索引
- ✅ 数据清洗和验证

---

## 3. 使用的权重

### 🎯 预训练权重

#### 3.1 基础视觉-语言模型

**InternVL3-8B**

```bash
# HuggingFace地址
https://huggingface.co/OpenGVLab/InternVL3-8B

# 本地缓存路径
/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9/
```

**组成部分**：
- Vision Encoder: InternViT-6B (冻结)
- Language Model: Qwen2.5-7B (LoRA微调)

#### 3.2 Sa2VA预训练权重

**Sa2VA-26B.pth**

```bash
# 本地路径
/home/ubuntu/Sa2VA-26B.pth

# 大小
~60GB

# 说明
Sa2VA-26B模型的预训练权重，用于初始化8B模型
只加载形状匹配的权重（知识蒸馏）
```

**来源**: ByteDance官方发布的Sa2VA-26B模型

#### 3.3 SAM2 Decoder权重

**sam2_hiera_large.pt**

```bash
# HuggingFace地址
https://huggingface.co/facebook/sam2-hiera-large

# 说明
SAM2分割模型的decoder部分
在Sa2VA中集成用于mask生成
```

### 📦 权重加载策略

#### 训练时的权重加载流程

```python
# 1. 加载InternVL3-8B基础模型
model_path = "/home/ubuntu/huggingface_cache/.../InternVL3-8B/"

# 2. 加载Sa2VA-26B预训练权重
pretrained_pth = "/home/ubuntu/Sa2VA-26B.pth"

# 3. 权重匹配与加载
# - Vision Encoder: 从InternVL3-8B加载（冻结）
# - Language Model: 从InternVL3-8B加载 + LoRA
# - Projector: 从Sa2VA-26B加载（形状匹配的部分）
# - SAM2 Decoder: 从Sa2VA-26B加载（可训练）
```

#### 权重初始化说明

| 模块 | 初始化来源 | 训练策略 |
|------|-----------|----------|
| **Vision Encoder** | InternVL3-8B | ❄️ 冻结 |
| **Language Model** | InternVL3-8B | 🔥 LoRA (r=64) |
| **Projector** | Sa2VA-26B (匹配部分) | 🔥 可训练 |
| **SAM2 Decoder** | Sa2VA-26B | 🔥 可训练 |
| **Embed/LM Head** | InternVL3-8B | 🔥 可训练 |

### 🎓 为什么使用26B权重初始化8B模型？

**知识蒸馏策略**：
1. ✅ 利用大模型的视觉-语言理解能力
2. ✅ 转移分割任务的先验知识
3. ✅ 加速小模型收敛
4. ✅ 提升最终性能

**实际效果**：
- 使用26B初始化：更快收敛，更好性能
- 从头训练：需要更多数据和时间

---

## 4. 环境准备

### 4.1 硬件要求

#### 训练

| 配置 | 推荐 | 最低 |
|------|------|------|
| **GPU** | 4×RTX 3090 (24GB) | 2×RTX 3090 |
| **内存** | 128GB+ | 64GB |
| **存储** | 500GB+ SSD | 200GB |
| **显存** | 24GB×4 | 24GB×2 |

**注意**：使用DeepSpeed ZeRO-3可以在有限显存下训练大模型

#### 推理

| 配置 | 推荐 | 最低 |
|------|------|------|
| **GPU** | 1×RTX 3090 (24GB) | 1×RTX 3080 (10GB) |
| **内存** | 32GB+ | 16GB |
| **显存** | 24GB | 12GB |

### 4.2 软件环境

#### 创建Conda环境

```bash
# 使用micromamba（推荐）
micromamba create -n sa2va python=3.10 -y
micromamba activate sa2va

# 或使用conda
conda create -n sa2va python=3.10 -y
conda activate sa2va
```

#### 安装依赖

```bash
cd /home/ubuntu/Sa2VA

# 安装PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装XTuner和依赖
pip install -e '.[all]'

# 或安装requirements.txt
pip install -r requirements.txt
```

#### 核心依赖版本

```
torch==2.1.0
transformers==4.37.2
xtuner>=0.1.17
deepspeed==0.12.6
peft==0.7.1
mmengine==0.10.1
opencv-python==4.9.0.80
pillow==10.2.0
huggingface-hub==0.20.3
```

### 4.3 下载预训练权重

#### InternVL3-8B

```bash
# 方法1: 使用huggingface-cli（推荐）
huggingface-cli download OpenGVLab/InternVL3-8B \
    --local-dir /home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9

# 方法2: Python代码
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)
"
```

#### Sa2VA-26B预训练权重

```bash
# 下载（如果没有）
# 注意：这是60GB的大文件
huggingface-cli download ByteDance/Sa2VA-26B \
    --local-dir /tmp/sa2va-26b

# 转换为.pth格式（如需要）
python tools/convert_hf_to_pth.py \
    --hf-model /tmp/sa2va-26b \
    --save-path /home/ubuntu/Sa2VA-26B.pth
```

---

## 5. 如何开始训练

### 5.1 训练配置文件

配置文件位置：
```
/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_vessel_finetune.py
```

### 5.2 关键训练参数

```python
# 模型
path = "/home/ubuntu/huggingface_cache/.../InternVL3-8B/"
pretrained_pth = "/home/ubuntu/Sa2VA-26B.pth"

# 数据
DATA_ROOT = '/home/ubuntu/Sa2VA/data/'
batch_size = 1              # 每GPU批次大小
accumulative_counts = 8     # 梯度累积（有效batch=32）
max_length = 4096           # 序列长度

# 优化器
lr = 2e-5                   # 学习率
weight_decay = 0.05
max_epochs = 1              # epoch数
warmup_ratio = 0.1          # warmup比例

# 保存
save_steps = 500            # 每500步保存一次
save_total_limit = 5        # 保留5个checkpoint

# LoRA配置
r = 64                      # LoRA rank
lora_alpha = 128
lora_dropout = 0.1
```

### 5.3 训练命令

#### 单机多卡训练（推荐）

```bash
cd /home/ubuntu/Sa2VA

# 4卡训练（使用DeepSpeed）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
xtuner train \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --work-dir work_dirs/vessel_segmentation \
    --deepspeed deepspeed_zero3
```

#### 2卡训练

```bash
# 调整配置中的accumulative_counts以保持有效batch size
CUDA_VISIBLE_DEVICES=0,1 \
xtuner train \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --work-dir work_dirs/vessel_segmentation \
    --deepspeed deepspeed_zero3
```

#### 使用自定义配置

```bash
# 复制配置文件
cp projects/sa2va/configs/sa2va_vessel_finetune.py \
   projects/sa2va/configs/my_config.py

# 编辑my_config.py修改参数

# 训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
xtuner train \
    projects/sa2va/configs/my_config.py \
    --work-dir work_dirs/my_experiment
```

### 5.4 完整训练脚本

创建 `train_vessel.sh`:

```bash
#!/bin/bash

# Sa2VA OCT血管分割训练脚本

set -e

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/ubuntu/Sa2VA:$PYTHONPATH

# 工作目录
WORK_DIR="work_dirs/vessel_segmentation_$(date +%Y%m%d_%H%M%S)"
CONFIG="projects/sa2va/configs/sa2va_vessel_finetune.py"

echo "========================================"
echo "Sa2VA OCT血管分割训练"
echo "========================================"
echo "配置文件: $CONFIG"
echo "工作目录: $WORK_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 检查数据集
if [ ! -d "Segment_DATA_Merged_512" ]; then
    echo "❌ 数据集不存在，请先下载数据集"
    exit 1
fi

echo "✅ 数据集已就绪"

# 检查权重
if [ ! -f "/home/ubuntu/Sa2VA-26B.pth" ]; then
    echo "⚠️  Sa2VA-26B.pth不存在，将使用默认初始化"
fi

# 开始训练
echo ""
echo "开始训练..."
echo ""

xtuner train \
    $CONFIG \
    --work-dir $WORK_DIR \
    --deepspeed deepspeed_zero3

echo ""
echo "========================================"
echo "✅ 训练完成！"
echo "========================================"
echo "检查点保存在: $WORK_DIR"
echo ""
```

使用：

```bash
chmod +x train_vessel.sh
bash train_vessel.sh
```

### 5.5 训练监控

#### 实时监控日志

```bash
# 方法1: 查看训练日志
tail -f work_dirs/vessel_segmentation/$(date +%Y%m%d)_*.log

# 方法2: 使用tmux/screen
tmux new -s training
# 在tmux中运行训练
# Ctrl+B, D 分离会话
# tmux attach -t training 重新连接
```

#### 查看GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用nvitop
nvitop
```

#### TensorBoard可视化（可选）

```bash
# 安装tensorboard
pip install tensorboard

# 启动tensorboard
tensorboard --logdir work_dirs/vessel_segmentation
```

### 5.6 训练输出

训练完成后，工作目录结构：

```
work_dirs/vessel_segmentation/
├── iter_500.pth          ← Checkpoint (500步)
├── iter_1000.pth         ← Checkpoint (1000步)
├── iter_12192.pth        ← 最终Checkpoint
├── 20251128_102030.log   ← 训练日志
├── tf_logs/              ← TensorBoard日志
└── vis_data/             ← 可视化数据（可选）
```

### 5.7 训练时间估计

基于4×RTX 3090 (24GB)：

| 数据量 | Batch Size | 预计时间 |
|--------|-----------|----------|
| 1,220张 (×10重复) | 32 (有效) | ~8-12小时 |
| 1,220张 (×5重复) | 32 (有效) | ~4-6小时 |
| 1,220张 (×1重复) | 32 (有效) | ~1-2小时 |

**实际训练**：
- iter_12192：约12,192步
- 平均速度：约3-4秒/步
- 总时间：约10-12小时

---

## 6. 如何开始预测

### 6.1 使用训练的Checkpoint预测

#### 准备工作

```bash
cd /home/ubuntu/Sa2VA

# 确保有训练好的checkpoint
ls work_dirs/vessel_segmentation/iter_12192.pth
```

#### 方法1: 使用HuggingFace格式模型（推荐）

首先转换checkpoint为HF格式：

```bash
# 转换checkpoint
python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --pth-model work_dirs/vessel_segmentation/iter_12192.pth \
    --save-path models/sa2va_vessel_hf
```

然后使用HF模型预测：

```python
# predict_hf.py
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np

# 加载模型
model_path = "models/sa2va_vessel_hf"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 加载图像
image_path = "Segment_DATA_Merged_512/images/sample.jpg"
image = Image.open(image_path).convert('RGB')

# 构建对话
question = "Please segment the blood vessels."
conversation = [
    {
        "role": "user",
        "content": f"<image>\n{question}"
    }
]

# 预测
with torch.no_grad():
    response, masks = model.chat(
        image=image,
        msgs=conversation,
        tokenizer=tokenizer,
        return_masks=True
    )

# 保存结果
if masks is not None:
    for i, mask in enumerate(masks):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(f"output_mask_{i}.png")

print(f"Response: {response}")
print(f"Saved {len(masks)} masks")
```

运行：

```bash
python predict_hf.py
```

#### 方法2: 使用预测脚本

```bash
# 单张图片预测
python demo/predict-img.py \
    --model_path models/sa2va_vessel_hf \
    --image_path Segment_DATA_Merged_512/images/sample.jpg \
    --output_dir predictions \
    --text "Please segment the blood vessels."

# 批量预测
python predict_5_videos.py  # 使用现有脚本
```

### 6.2 批量预测脚本

创建 `batch_predict.py`:

```python
"""
批量预测OCT血管分割
"""
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm

# 配置
MODEL_PATH = "models/sa2va_vessel_hf"
DATA_ROOT = "Segment_DATA_Merged_512"
OUTPUT_DIR = "predictions_batch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("加载模型...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# 加载数据集
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")

# 批量预测
results = []
for idx, sample in enumerate(tqdm(dataset)):
    image_path = os.path.join(DATA_ROOT, sample['image'])
    image = Image.open(image_path).convert('RGB')
    
    # 预测
    conversation = [{
        "role": "user",
        "content": "<image>\nPlease segment the blood vessels."
    }]
    
    with torch.no_grad():
        response, masks = model.chat(
            image=image,
            msgs=conversation,
            tokenizer=tokenizer,
            return_masks=True
        )
    
    # 保存mask
    if masks is not None and len(masks) > 0:
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{idx:04d}.png")
        mask_img = Image.fromarray((masks[0] * 255).astype(np.uint8))
        mask_img.save(mask_path)
        
        results.append({
            "image": sample['image'],
            "prediction": mask_path,
            "response": response
        })

# 保存结果
with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ 预测完成！保存在 {OUTPUT_DIR}")
```

运行：

```bash
python batch_predict.py
```

### 6.3 从HuggingFace下载模型预测

如果模型已上传到HuggingFace：

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# 直接从HF下载
model = AutoModel.from_pretrained(
    "ly17/sa2va-vessel-hf",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "ly17/sa2va-vessel-hf",
    trust_remote_code=True
)

# 预测
image = Image.open("test.jpg")
response, masks = model.chat(
    image=image,
    msgs=[{"role": "user", "content": "<image>\nSegment vessels."}],
    tokenizer=tokenizer,
    return_masks=True
)
```

### 6.4 预测参数调整

#### 温度采样

```python
# 更确定的输出
response = model.chat(
    image=image,
    msgs=conversation,
    tokenizer=tokenizer,
    temperature=0.1,  # 降低随机性
    top_p=0.9
)
```

#### Mask后处理

```python
import cv2

# 二值化
mask_binary = (masks[0] > 0.5).astype(np.uint8) * 255

# 形态学操作
kernel = np.ones((3, 3), np.uint8)
mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
```

### 6.5 可视化预测结果

```python
import matplotlib.pyplot as plt

def visualize_prediction(image, mask, save_path):
    """可视化预测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Prediction Mask')
    axes[1].axis('off')
    
    # 叠加
    axes[2].imshow(image)
    axes[2].imshow(mask, alpha=0.5, cmap='Reds')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# 使用
visualize_prediction(image, masks[0], "visualization.png")
```

---

## 7. 模型转换

### 7.1 Checkpoint转HuggingFace格式

```bash
# 基本转换
python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --pth-model work_dirs/vessel_segmentation/iter_12192.pth \
    --save-path models/sa2va_vessel_hf

# 指定配置
python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --pth-model work_dirs/vessel_segmentation/iter_12192.pth \
    --save-path models/sa2va_vessel_hf \
    --model-name "Sa2VA-Vessel-8B"
```

### 7.2 验证转换

```python
# verify_conversion.py
from transformers import AutoModel, AutoTokenizer

model_path = "models/sa2va_vessel_hf"

# 加载模型
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("✅ 模型加载成功")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"Tokenizer词表大小: {len(tokenizer)}")
```

### 7.3 上传到HuggingFace

```bash
# 登录
huggingface-cli login

# 上传
huggingface-cli upload \
    ly17/sa2va-vessel-hf \
    models/sa2va_vessel_hf \
    . \
    --repo-type model \
    --commit-message "Upload Sa2VA vessel segmentation model"
```

---

## 8. 常见问题

### 8.1 训练相关

#### Q1: 显存不足怎么办？

**A**: 
```python
# 减小batch size
batch_size = 1
accumulative_counts = 16  # 增加梯度累积

# 减小序列长度
max_length = 2048

# 使用DeepSpeed ZeRO-3
# 已在配置文件中启用

# 减少LoRA rank
r = 32
lora_alpha = 64
```

#### Q2: 训练速度太慢？

**A**:
```python
# 减少dataloader workers
dataloader_num_workers = 2

# 减少数据重复
repeats = 5  # 从10改为5

# 使用更少的卡但增加accumulation
CUDA_VISIBLE_DEVICES=0,1  # 2卡
accumulative_counts = 16   # 保持有效batch=32
```

#### Q3: 如何从checkpoint继续训练？

**A**:
```bash
xtuner train \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --work-dir work_dirs/vessel_segmentation \
    --resume work_dirs/vessel_segmentation/iter_5000.pth
```

#### Q4: 训练loss不下降？

**A**:
1. 检查学习率是否太大或太小
2. 检查数据是否正确加载
3. 增加warmup步数
4. 检查梯度裁剪参数

### 8.2 预测相关

#### Q5: 预测结果不理想？

**A**:
1. 检查使用的checkpoint是否正确
2. 尝试不同的temperature参数
3. 使用更多训练步数的checkpoint
4. 对mask进行后处理

#### Q6: 如何加速推理？

**A**:
```python
# 使用半精度
model = model.to(torch.bfloat16)

# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 批量推理
# 一次处理多张图片
```

#### Q7: 内存占用太大？

**A**:
```python
# 清理GPU缓存
torch.cuda.empty_cache()

# 使用更小的图像尺寸
# 在配置中修改target_length

# 及时释放不需要的变量
del intermediate_results
torch.cuda.empty_cache()
```

### 8.3 数据相关

#### Q8: 如何准备自己的数据？

**A**: 参考 `Segment_DATA_Merged_512/annotations.json` 格式：
```json
[
  {
    "image": "images/your_image.jpg",
    "mask": [
      {
        "segmentation": [[x1, y1, x2, y2, ...]],
        "category_id": 1
      }
    ],
    "conversations": [
      {"from": "human", "value": "<image>\n描述你的任务"},
      {"from": "gpt", "value": "<p>目标</p><vp>[[坐标]]</vp>[SEG]"}
    ]
  }
]
```

#### Q9: 数据增强如何配置？

**A**: 在配置文件中修改：
```python
extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
    # 可添加其他增强
)
```

### 8.4 环境相关

#### Q10: CUDA版本不匹配？

**A**:
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 安装对应版本的PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Q11: DeepSpeed安装失败？

**A**:
```bash
# 方法1: 使用预编译版本
pip install deepspeed --no-build-isolation

# 方法2: 从源码安装
DS_BUILD_OPS=1 pip install deepspeed

# 方法3: 使用conda
conda install -c conda-forge deepspeed
```

---

## 📚 附录

### A. 完整命令速查

```bash
# 环境激活
micromamba activate sa2va

# 训练（4卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 xtuner train \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --work-dir work_dirs/vessel_seg

# 转换模型
python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_vessel_finetune.py \
    --pth-model work_dirs/vessel_seg/iter_12192.pth \
    --save-path models/sa2va_vessel_hf

# 预测
python predict_hf.py

# 上传到HF
huggingface-cli upload ly17/sa2va-vessel-hf models/sa2va_vessel_hf .
```

### B. 目录结构

```
Sa2VA/
├── projects/sa2va/
│   ├── configs/
│   │   └── sa2va_vessel_finetune.py  ← 训练配置
│   ├── models/                        ← 模型定义
│   └── datasets/                      ← 数据加载
├── tools/
│   ├── train.py                       ← 训练入口
│   └── convert_to_hf.py              ← 模型转换
├── Segment_DATA_Merged_512/           ← 数据集
│   ├── images/
│   ├── masks/
│   └── annotations.json
├── work_dirs/                         ← 训练输出
│   └── vessel_segmentation/
│       └── iter_12192.pth
└── models/                            ← 转换后的模型
    └── sa2va_vessel_hf/
```

### C. 相关链接

- **Sa2VA官方**: https://github.com/magic-research/Sa2VA
- **InternVL**: https://huggingface.co/OpenGVLab/InternVL3-8B
- **SAM2**: https://github.com/facebookresearch/sam2
- **XTuner**: https://github.com/InternLM/xtuner
- **DeepSpeed**: https://www.deepspeed.ai/

### D. 引用

如果使用本项目，请引用：

```bibtex
@article{sa2va2025,
  title={Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and others},
  journal={arXiv preprint arXiv:2501.04001},
  year={2025}
}
```

---

**文档版本**: 1.0  
**创建日期**: 2025-11-28  
**适用模型**: Sa2VA-InternVL3-8B  
**应用场景**: OCT血管分割

**获取帮助**:
- GitHub Issues: https://github.com/qimingfan10/RLSa2va/issues
- HuggingFace: https://huggingface.co/ly17/sa2va-vessel-hf

**下一步**: 
1. ✅ 准备环境和数据
2. ✅ 开始训练
3. ✅ 转换和测试模型
4. ✅ 上传分享成果
