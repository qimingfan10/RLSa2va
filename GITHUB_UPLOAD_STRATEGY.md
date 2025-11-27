# Sa2VA GitHub上传策略

**目标仓库**: https://github.com/qimingfan10/RLSa2va.git

---

## 📊 项目大文件分析

### 当前空间占用

```
总计: ~100GB
├── models/              60GB   ⚠️ HuggingFace模型权重
├── work_dirs/           35GB   ⚠️ 训练checkpoints
├── pretrained/          857MB  ⚠️ SAM2预训练权重
├── data/                245MB  ⚠️ 数据集
├── Segment_DATA_Merged/ 194MB  ⚠️ 合并数据
├── assets/              33MB   ✅ 可上传
├── 其他代码/文档         ~500MB ✅ 可上传
```

### 具体大文件清单

**模型文件** (每个4-5GB):
```
models/sa2va_vessel_hf/
├── model-00001-of-00007.safetensors (4.5GB)
├── model-00002-of-00007.safetensors (4.6GB)
├── model-00003-of-00007.safetensors (4.6GB)
├── model-00004-of-00007.safetensors (4.7GB)
├── model-00005-of-00007.safetensors (4.7GB)
├── model-00006-of-00007.safetensors (4.0GB)
└── model-00007-of-00007.safetensors (3.1GB)
总计: ~30GB

models/sa2va_vessel_iter3672_hf/
└── (同样的7个文件，~30GB)
```

**训练checkpoints**:
```
work_dirs/vessel_segmentation/
├── iter_12192.pth (~5GB)
├── iter_12192_hf/ (~30GB)
└── ...

work_dirs/merged_vessel_segmentation/
├── iter_3672.pth (~5GB)
└── ...
```

---

## 🎯 上传策略

### 方案A: Git LFS (推荐用于小型模型文件)

**限制**: 
- GitHub免费版: 1GB存储 + 1GB/月带宽
- 付费版: $5/月 for 50GB存储 + 50GB带宽
- **不适合**: 我们的60GB模型

### 方案B: HuggingFace Model Hub (推荐) ⭐⭐⭐

**上传模型到HuggingFace**:
```bash
# 1. 安装huggingface_hub
pip install huggingface_hub

# 2. 登录
huggingface-cli login

# 3. 上传模型
huggingface-cli upload qimingfan10/sa2va-vessel-hf ./models/sa2va_vessel_hf
huggingface-cli upload qimingfan10/sa2va-vessel-iter3672-hf ./models/sa2va_vessel_iter3672_hf
```

**GitHub仓库引用**:
```python
# 在README.md中说明如何下载
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="qimingfan10/sa2va-vessel-hf",
    local_dir="./models/sa2va_vessel_hf"
)
```

**优势**:
- ✅ 无限存储（免费）
- ✅ 专为ML模型设计
- ✅ 自动版本管理
- ✅ 社区可见性高
- ✅ 支持模型卡片说明

### 方案C: 云存储 + 下载脚本

**使用云盘**:
- Google Drive
- Dropbox
- 百度网盘
- 阿里云OSS

**提供下载脚本**:
```bash
# download_models.sh
wget https://drive.google.com/xxx/sa2va_vessel_hf.tar.gz
tar -xzf sa2va_vessel_hf.tar.gz -C models/
```

---

## 📝 推荐上传方案

### 第一步: 更新.gitignore

```bash
# 添加到.gitignore
models/*.safetensors
models/*.bin
models/*/*.safetensors
models/*/*.bin
work_dirs/
pretrained/
data/*/images/
data/*/masks/
Segment_DATA_Merged_512/
*.pth
*.ckpt
*.tar.gz
*.zip
video_prediction_*/
*_results/
*.mp4
```

### 第二步: 上传代码到GitHub

```bash
cd /home/ubuntu/Sa2VA

# 初始化git（如果还没有）
git init
git remote add origin https://github.com/qimingfan10/RLSa2va.git

# 添加代码（排除大文件）
git add .
git commit -m "Initial commit: Sa2VA code and documentation"
git branch -M main
git push -u origin main
```

### 第三步: 上传模型到HuggingFace

**创建两个模型仓库**:

1. **sa2va-vessel-hf** (iter_12192，旧模型)
2. **sa2va-vessel-iter3672-hf** (iter_3672，新模型)

**上传脚本**:
```bash
# 安装工具
pip install huggingface_hub

# 登录HuggingFace
huggingface-cli login

# 上传模型1
cd /home/ubuntu/Sa2VA/models/sa2va_vessel_hf
huggingface-cli upload qimingfan10/sa2va-vessel-hf . \
  --repo-type model \
  --commit-message "Upload Sa2VA vessel segmentation model (iter_12192)"

# 上传模型2
cd /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf
huggingface-cli upload qimingfan10/sa2va-vessel-iter3672-hf . \
  --repo-type model \
  --commit-message "Upload Sa2VA vessel segmentation model (iter_3672)"
```

### 第四步: 创建模型下载脚本

**在GitHub仓库中添加**:

`scripts/download_models.sh`:
```bash
#!/bin/bash
echo "Downloading Sa2VA pre-trained models from HuggingFace..."

# 创建目录
mkdir -p models

# 下载模型1
echo "Downloading sa2va-vessel-hf (iter_12192)..."
huggingface-cli download qimingfan10/sa2va-vessel-hf \
  --local-dir models/sa2va_vessel_hf

# 下载模型2
echo "Downloading sa2va-vessel-iter3672-hf (iter_3672)..."
huggingface-cli download qimingfan10/sa2va-vessel-iter3672-hf \
  --local-dir models/sa2va_vessel_iter3672_hf

echo "✅ Models downloaded successfully!"
```

---

## 📦 GitHub仓库应包含的内容

### ✅ 应该上传

**代码** (~50MB):
- [x] Python脚本 (*.py)
- [x] Shell脚本 (*.sh)
- [x] 配置文件 (*.yaml, *.json, *.toml)
- [x] projects/ 目录 (模型定义)
- [x] tools/ 目录 (工具脚本)

**文档** (~10MB):
- [x] 所有Markdown文档 (*.md)
- [x] README.md ⭐
- [x] LICENSE
- [x] 方法论文档 (SA2VA_METHODOLOGY.md等)

**配置**:
- [x] .gitignore
- [x] pyproject.toml
- [x] requirements.txt (需要创建)

**小型资源** (<10MB):
- [x] assets/ (可视化图片)
- [x] demo/ (演示脚本)
- [x] docs/ (文档资源)

### ❌ 不应上传（使用外部存储）

**模型权重** (60GB):
- [ ] models/*.safetensors → HuggingFace
- [ ] models/*.bin → HuggingFace

**训练checkpoints** (35GB):
- [ ] work_dirs/*.pth → 云盘或HuggingFace
- [ ] work_dirs/*_hf/ → HuggingFace

**预训练权重** (857MB):
- [ ] pretrained/sam2_*.pt → 官方链接

**数据集** (245MB):
- [ ] data/images/ → 云盘
- [ ] data/masks/ → 云盘
- [ ] 或在README中说明如何获取

**生成结果** (~50MB):
- [ ] *_results/
- [ ] *.mp4
- [ ] evaluation_*/

**日志文件** (~5MB):
- [ ] *.log

---

## 🔧 需要创建的文件

### 1. requirements.txt

```txt
torch>=2.1.0
transformers>=4.37.0
mmengine>=0.10.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
huggingface_hub>=0.19.0
```

### 2. 更新README.md

添加模型下载说明：

```markdown
## 📥 模型下载

我们的预训练模型托管在HuggingFace上：

### 方法1: 使用脚本自动下载
```bash
bash scripts/download_models.sh
```

### 方法2: 手动下载
```bash
# 安装HuggingFace CLI
pip install huggingface_hub

# 下载模型
huggingface-cli download qimingfan10/sa2va-vessel-hf --local-dir models/sa2va_vessel_hf
huggingface-cli download qimingfan10/sa2va-vessel-iter3672-hf --local-dir models/sa2va_vessel_iter3672_hf
```

### 模型列表

| 模型 | HuggingFace链接 | 大小 | 训练迭代 | IoU | Dice |
|------|----------------|------|----------|-----|------|
| sa2va-vessel-hf | [🤗 Hub](https://huggingface.co/qimingfan10/sa2va-vessel-hf) | 30GB | 12,192 | 0.6725 | 0.8005 |
| sa2va-vessel-iter3672-hf | [🤗 Hub](https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf) | 30GB | 3,672 | 0.6725 | 0.8005 |
```

### 3. MODEL_CARD.md (用于HuggingFace)

```markdown
---
language: en
license: apache-2.0
tags:
- medical-imaging
- vessel-segmentation
- oct
- multimodal
- vision-language
datasets:
- custom-oct-vessel
metrics:
- iou
- dice
---

# Sa2VA: Segment Anything to Vessel Analysis

## Model Description

Sa2VA is a multimodal vision-language model for medical vessel segmentation...

## Model Details
- **Developed by**: qimingfan10
- **Model type**: Vision-Language Segmentation
- **Architecture**: InternVL-8B + SAM2-Large
- **Training data**: OCT retinal vessel images (9,346 images)

## Performance
- IoU: 0.6725
- Dice: 0.8005

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "qimingfan10/sa2va-vessel-hf",
    trust_remote_code=True
)
```
```

---

## ⚡ 执行清单

### 准备阶段

- [ ] 1. 检查并更新.gitignore
- [ ] 2. 创建requirements.txt
- [ ] 3. 创建scripts/download_models.sh
- [ ] 4. 更新README.md（添加模型下载说明）
- [ ] 5. 清理临时文件和日志

### GitHub上传

- [ ] 1. 初始化git仓库
- [ ] 2. 添加远程仓库
- [ ] 3. 提交代码和文档
- [ ] 4. 推送到GitHub

### HuggingFace上传

- [ ] 1. 注册/登录HuggingFace账号
- [ ] 2. 创建模型仓库 (sa2va-vessel-hf)
- [ ] 3. 创建模型仓库 (sa2va-vessel-iter3672-hf)
- [ ] 4. 编写MODEL_CARD.md
- [ ] 5. 上传模型1 (30GB，需要时间)
- [ ] 6. 上传模型2 (30GB，需要时间)

### 验证

- [ ] 1. 在新环境克隆GitHub仓库
- [ ] 2. 运行download_models.sh
- [ ] 3. 验证模型可用
- [ ] 4. 测试推理脚本

---

## 📌 注意事项

### GitHub限制
- 单文件 < 100MB（硬性限制）
- 仓库推荐 < 1GB
- 超过100MB需要Git LFS（不推荐用于ML模型）

### HuggingFace优势
- ✅ 无大小限制
- ✅ 免费托管
- ✅ 版本管理
- ✅ 模型卡片展示
- ✅ 社区可见

### 上传速度估算
- HuggingFace上传: ~1-2小时/30GB（取决于网速）
- 建议使用稳定网络环境
- 可以分批上传safetensors文件

---

## 🔗 相关链接

- **GitHub仓库**: https://github.com/qimingfan10/RLSa2va.git
- **HuggingFace模型1**: https://huggingface.co/qimingfan10/sa2va-vessel-hf (待创建)
- **HuggingFace模型2**: https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf (待创建)
- **HuggingFace文档**: https://huggingface.co/docs/hub/models-uploading

---

**创建时间**: 2025-11-27  
**状态**: 待执行  
**预计时间**: 3-4小时（包括上传）
