# 📥 模型下载

本项目的预训练模型托管在HuggingFace Model Hub上，代码托管在GitHub。

## 快速开始

### 1. 克隆代码仓库

```bash
git clone https://github.com/qimingfan10/RLSa2va.git
cd RLSa2va
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载预训练模型

**方法A: 使用自动脚本（推荐）**

```bash
bash scripts/download_models.sh
```

脚本会提示您选择要下载的模型：
- 选项1: sa2va-vessel-hf (iter_12192, 30GB)
- 选项2: sa2va-vessel-iter3672-hf (iter_3672, 30GB)  
- 选项3: 下载所有模型 (60GB)

**方法B: 手动下载**

```bash
# 安装HuggingFace CLI
pip install huggingface_hub

# 下载模型1 (iter_12192)
huggingface-cli download qimingfan10/sa2va-vessel-hf \
    --local-dir models/sa2va_vessel_hf

# 下载模型2 (iter_3672)
huggingface-cli download qimingfan10/sa2va-vessel-iter3672-hf \
    --local-dir models/sa2va_vessel_iter3672_hf
```

**方法C: 在Python中下载**

```python
from huggingface_hub import snapshot_download

# 下载模型1
snapshot_download(
    repo_id="qimingfan10/sa2va-vessel-hf",
    local_dir="./models/sa2va_vessel_hf"
)

# 下载模型2
snapshot_download(
    repo_id="qimingfan10/sa2va-vessel-iter3672-hf",
    local_dir="./models/sa2va_vessel_iter3672_hf"
)
```

## 可用模型

| 模型名称 | HuggingFace链接 | 大小 | 训练迭代 | IoU | Dice | 说明 |
|---------|----------------|------|----------|-----|------|------|
| sa2va-vessel-hf | [🤗 Hub](https://huggingface.co/qimingfan10/sa2va-vessel-hf) | 30GB | 12,192 | 0.6725 | 0.8005 | 基础训练版本 |
| sa2va-vessel-iter3672-hf | [🤗 Hub](https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf) | 30GB | 3,672 | 0.6725 | 0.8005 | 优化训练版本 |

## 模型详情

### sa2va-vessel-hf (iter_12192)

**训练配置**:
- Base Model: InternVL-8B + SAM2-Large
- Training Data: 9,346 OCT vessel images
- Batch Size: 16 (4×4 GPUs)
- Iterations: 12,192
- Training Time: ~72 hours (4× A6000)

**性能**:
- IoU: 0.6725
- Dice: 0.8005
- Precision: 0.8659
- Recall: 0.7539

### sa2va-vessel-iter3672-hf (iter_3672)

**训练配置**:
- Base Model: InternVL-8B + SAM2-Large  
- Training Data: 9,346 OCT vessel images
- Batch Size: 32 (8×4 GPUs)
- Iterations: 3,672
- Training Time: ~18 hours (4× A6000)

**改进**:
- 更大的batch size
- 更高效的数据加载
- 优化的内存管理

**性能**:
- IoU: 0.6725
- Dice: 0.8005
- Precision: 0.8659
- Recall: 0.7539

## 使用模型

### 基础推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "models/sa2va_vessel_hf",  # 或使用HF路径
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/sa2va_vessel_hf",
    trust_remote_code=True
)

# 推理
image = Image.open("your_image.jpg")
text = "<image>Please segment the blood vessel."

result = model.predict_forward(
    image=image,
    text=text,
    tokenizer=tokenizer
)

# 获取分割掩码
pred_mask = result['prediction_masks'][0]
```

### 使用提供的脚本

```bash
# 评估10张图片
bash run_evaluate_10_images.sh

# 评估100张图片
python evaluate_100_samples.py

# 视频预测
bash run_predict_5_videos.sh
```

## 注意事项

### 磁盘空间

- 每个模型约30GB
- 建议预留至少50GB空间
- 下载时间取决于网络速度（通常1-2小时）

### GPU要求

- 推荐: NVIDIA GPU with 24GB+ VRAM (A6000, A100, RTX 3090/4090)
- 最低: 16GB VRAM (可能需要降低batch size)
- CPU模式: 不推荐（速度极慢）

### 依赖版本

- Python: 3.10+
- PyTorch: 2.1.0+
- Transformers: 4.37.0+
- CUDA: 11.8+ (推荐12.1)

## 常见问题

### Q: 下载速度慢怎么办？

A: 可以使用HuggingFace镜像站：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

### Q: 模型下载中断了？

A: 使用`--resume-download`参数继续下载：
```bash
huggingface-cli download qimingfan10/sa2va-vessel-hf \
    --local-dir models/sa2va_vessel_hf \
    --resume-download
```

### Q: 如何验证模型是否正确下载？

A: 运行测试脚本：
```bash
python test_existing_hf_model.py
```

## 许可证

模型遵循 Apache 2.0 许可证。

## 引用

如果您使用我们的模型，请引用：

```bibtex
@misc{sa2va2024,
  title={Sa2VA: Segment Anything to Vessel Analysis},
  author={Qiming Fan},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/qimingfan10/sa2va-vessel-hf}}
}
```

## 相关链接

- 📄 [方法论文档](SA2VA_METHODOLOGY.md)
- 🔧 [技术细节](SA2VA_TECHNICAL_DETAILS.md)
- 📖 [完整文档索引](DOCUMENTATION_INDEX.md)
- 🐙 [GitHub仓库](https://github.com/qimingfan10/RLSa2va)
- 🤗 [HuggingFace模型](https://huggingface.co/qimingfan10)
