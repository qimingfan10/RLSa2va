# work_dirs 和 data 目录说明

## 📁 work_dirs 目录内容

**路径**: `/home/ubuntu/Sa2VA/work_dirs/`  
**总大小**: ~35GB

### 目录结构

```
work_dirs/
├── vessel_segmentation/           33GB    ← 主要训练目录
│   ├── iter_12192.pth            2.5GB   ← 训练checkpoint (12192步)
│   └── 大量训练日志文件
│
├── merged_vessel_segmentation/    2.5GB   ← 合并数据集训练
│   ├── iter_3672.pth             2.5GB   ← 训练checkpoint (3672步)
│   └── 训练日志
│
├── official_test_results/         132KB   ← 官方测试结果
│   └── 测试日志
│
├── hf_simple_training/            4KB     ← HF简单训练
└── vessel_simple/                 4KB     ← 简单训练
```

### 重要文件

| 文件 | 大小 | 说明 | 重要性 |
|------|------|------|--------|
| `vessel_segmentation/iter_12192.pth` | 2.5GB | 旧模型训练权重 | ⭐⭐⭐ 重要 |
| `merged_vessel_segmentation/iter_3672.pth` | 2.5GB | 新模型训练权重 | ⭐⭐⭐ 重要 |
| `vessel_segmentation/*.log` | ~30GB | 训练日志文件 | ⭐ 可选 |

### 作用说明

1. **iter_12192.pth**: 
   - 第一次训练的checkpoint
   - 训练步数: 12,192
   - 用于生成 `models/sa2va_vessel_hf/`

2. **iter_3672.pth**:
   - 第二次训练的checkpoint
   - 训练步数: 3,672
   - 用于生成 `models/sa2va_vessel_iter3672_hf/`

3. **日志文件**:
   - 训练过程的详细记录
   - 损失曲线、学习率变化等
   - 用于复现和调试

---

## 📊 data 目录内容

**路径**: `/home/ubuntu/Sa2VA/data/`  
**总大小**: ~246MB

### 目录结构

```
data/
├── vessel_data/                   144MB
│   └── images/                   (1,220张图片)
│       ├── Fang Kun^(...)_frame_000016.jpg
│       ├── He Gui Sheng(...)_frame_000024.jpg
│       └── ...
│
└── merged_vessel_data/            102MB
    └── images/                   (1,220张图片)
        ├── 相同的图片文件
        └── ...
```

### 文件统计

| 目录 | 大小 | 文件数 | 说明 |
|------|------|--------|------|
| `vessel_data/images/` | 144MB | 1,220 | OCT血管图片（原始） |
| `merged_vessel_data/images/` | 102MB | 1,220 | OCT血管图片（合并） |

### 数据说明

1. **图片格式**: JPG
2. **图片内容**: OCT视网膜血管图像
3. **命名规则**: `患者名(ID)_位置_帧号.jpg`
4. **用途**: 训练数据集

**注意**: 
- 两个目录的图片内容相同
- `merged_vessel_data` 可能是处理后的版本
- 实际标注数据在 `/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/`

---

## 🎯 上传建议

### 是否需要上传？

| 内容 | 是否上传 | 原因 |
|------|----------|------|
| **模型权重 (HF格式)** | ✅ 已上传 | 用户使用，30GB×2 |
| **训练checkpoint** | ⚠️ 可选 | 用于复现训练，5GB |
| **训练日志** | ❌ 不推荐 | 太大(30GB)，价值有限 |
| **数据集图片** | ⚠️ 可选 | 训练数据，246MB |
| **完整数据集** | ⚠️ 可选 | 包含标注，194MB |

### 推荐上传方案

#### 方案A: 最小化（推荐）⭐
```
只上传HuggingFace模型（已完成）
- 用户可以直接使用
- 节省存储空间
```

#### 方案B: 完整可复现
```
上传内容:
1. HuggingFace模型（已完成）          60GB
2. 训练checkpoint                     5GB
3. 完整数据集                        440MB
总计: ~65GB
```

#### 方案C: 研究用（最全）
```
上传内容:
1. HuggingFace模型（已完成）          60GB
2. 训练checkpoint                     5GB  
3. 完整数据集 + 标注                 440MB
4. 训练日志（压缩）                  ~5GB
总计: ~70GB
```

---

## 📦 打包上传命令

### 1. 上传训练checkpoint到HuggingFace

创建新仓库存放checkpoint：

```bash
# 打包checkpoint
cd /home/ubuntu/Sa2VA/work_dirs

# 上传checkpoint到HF
huggingface-cli upload ly17/sa2va-checkpoints \
    vessel_segmentation/iter_12192.pth \
    iter_12192.pth \
    --repo-type model \
    --commit-message "Training checkpoint at iteration 12192"

huggingface-cli upload ly17/sa2va-checkpoints \
    merged_vessel_segmentation/iter_3672.pth \
    iter_3672.pth \
    --repo-type model \
    --commit-message "Training checkpoint at iteration 3672"
```

### 2. 上传数据集

#### 选项A: 上传到HuggingFace Dataset

```bash
# 上传数据集
huggingface-cli upload ly17/sa2va-vessel-dataset \
    /home/ubuntu/Sa2VA/Segment_DATA_Merged_512 \
    . \
    --repo-type dataset \
    --commit-message "Sa2VA OCT vessel segmentation dataset"
```

#### 选项B: 打包上传到GitHub Release

```bash
# 打包数据集
cd /home/ubuntu/Sa2VA
tar -czf sa2va_dataset.tar.gz Segment_DATA_Merged_512/

# 然后通过GitHub Release上传
# 或使用gh命令行
gh release create v1.0.0 \
    sa2va_dataset.tar.gz \
    --title "Sa2VA Dataset v1.0" \
    --notes "OCT vessel segmentation dataset"
```

#### 选项C: 上传到云盘

```bash
# 打包所有数据
cd /home/ubuntu/Sa2VA

# 数据集
tar -czf dataset.tar.gz data/ Segment_DATA_Merged_512/

# Checkpoint
tar -czf checkpoints.tar.gz \
    work_dirs/vessel_segmentation/iter_12192.pth \
    work_dirs/merged_vessel_segmentation/iter_3672.pth

# 然后上传到Google Drive, Dropbox等
```

---

## 🚀 快速执行方案

### 推荐：上传checkpoint和数据集到HuggingFace

```bash
cd /home/ubuntu/Sa2VA

# 1. 创建checkpoint仓库并上传
huggingface-cli upload ly17/sa2va-checkpoints \
    work_dirs/vessel_segmentation/iter_12192.pth \
    iter_12192.pth

huggingface-cli upload ly17/sa2va-checkpoints \
    work_dirs/merged_vessel_segmentation/iter_3672.pth \
    iter_3672.pth

# 2. 创建数据集仓库并上传
huggingface-cli upload ly17/sa2va-vessel-dataset \
    Segment_DATA_Merged_512 \
    . \
    --repo-type dataset

# 预计上传时间: 
# - Checkpoint: ~30分钟 (5GB)
# - Dataset: ~5分钟 (440MB)
# 总计: ~35分钟
```

---

## 📋 上传后的仓库结构

### HuggingFace仓库列表

1. **ly17/sa2va-vessel-hf** (30GB)
   - 模型1: iter_12192转换的HF格式

2. **ly17/sa2va-vessel-iter3672-hf** (30GB)
   - 模型2: iter_3672转换的HF格式

3. **ly17/sa2va-checkpoints** (5GB) ← 新增
   - iter_12192.pth
   - iter_3672.pth

4. **ly17/sa2va-vessel-dataset** (440MB) ← 新增
   - images/ (图片文件)
   - masks/ (标注mask)
   - annotations.json (标注信息)

---

## 💡 建议

### 对于普通用户
- ✅ 只下载HF模型即可
- ✅ 可以直接推理使用

### 对于研究者
- ✅ 下载HF模型
- ✅ 下载checkpoint（如需从头训练）
- ✅ 下载数据集（如需复现训练）

### 对于开发者
- ✅ 所有内容都下载
- ✅ 包括训练日志（调试用）

---

**创建时间**: 2025-11-28  
**总结**: work_dirs主要是训练输出（35GB），data是训练数据（246MB）
