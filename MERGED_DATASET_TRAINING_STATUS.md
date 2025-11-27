# Merged Dataset训练状态

## 📊 数据集准备完成

### ✅ 数据集信息

**来源**: Segment_DATA_Merged_512.tar.gz
**样本数**: 1220张图片
**图像尺寸**: 512×512
**数据路径**: `/home/ubuntu/Sa2VA/data/merged_vessel_data/`

**数据统计**:
- 总图片数: 1220
- 所有图片都有JSON标注
- 多边形标注已转换为Sa2VA格式

### 数据格式

```json
{
  "image": "filename.jpg",
  "mask": [[x1, y1, x2, y2, ...], ...],  // 多边形坐标
  "text": "blood vessel",
  "conversations": [...]
}
```

## ❌ 当前问题: CUDA OOM

### 错误信息

```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 2.03 GiB. 
GPU 0 has a total capacity of 23.68 GiB of which 1.82 GiB is free.
```

### 错误位置

```python
File "transformers/modeling_utils.py", line 3566
    old_centered_embeddings = old_embeddings_weight - mean_embeddings
```

**发生在**: `resize_token_embeddings()` 调用时
**原因**: 添加特殊token时需要resize embedding层，这需要临时分配大量显存

### 为什么会OOM？

1. **模型初始化阶段**: 在训练开始前，模型初始化时就OOM
2. **DeepSpeed Zero-3**: 虽然使用了Zero-3，但在resize_token_embeddings时需要gather所有权重
3. **显存不足**: 4×24GB GPU，但初始化时单个GPU需要>22GB

## 💡 解决方案

### 方案1: 使用已转换的HuggingFace模型继续训练 ⭐推荐

**优势**:
- 已经有转换好的HuggingFace格式模型
- 可以直接使用HuggingFace的训练工具
- 避免mmengine的初始化问题

**步骤**:
1. 使用`/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf/`
2. 创建HuggingFace Trainer脚本
3. 在新数据集上fine-tune

### 方案2: 修改模型初始化流程

**修改点**:
- 在模型初始化前清理显存
- 使用更激进的CPU offload
- 修改resize_token_embeddings的实现

**难度**: 高，需要修改Sa2VA源码

### 方案3: 使用更少的GPU但更大的显存

**如果有**:
- 2×48GB GPU
- 或1×80GB GPU

**当前**: 4×24GB = 96GB总显存，但单GPU只有24GB

## 📝 已完成的工作

### ✅ 数据准备

1. **解压数据集**: ✅
   ```bash
   tar -xzf Segment_DATA_Merged_512.tar.gz
   ```

2. **转换格式**: ✅
   ```bash
   python3 prepare_merged_dataset.py
   ```
   - 读取JSON标注
   - 转换为Sa2VA格式
   - 添加text字段
   - 保存annotations.json

3. **数据验证**: ✅
   - 1220个样本全部成功处理
   - 所有图片已复制到目标目录
   - annotations.json格式正确

### ✅ 配置文件

创建了新的训练配置:
- `/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py`
- 数据路径: `merged_vessel_data/`
- 训练参数: 3 epochs, lr=2e-5
- 尝试从iter_12192.pth加载权重

### ✅ 训练脚本

创建了训练脚本:
- `/home/ubuntu/Sa2VA/train_merged_dataset.sh`
- 使用4个GPU
- DeepSpeed Zero-3

## 🎯 推荐的下一步

### 方案A: 使用HuggingFace格式模型 (推荐)

**原因**:
- 已有转换好的模型
- HuggingFace生态系统更成熟
- 更容易处理显存问题

**实现**:
```python
# 1. 加载HuggingFace模型
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 2. 准备数据集
# 3. 使用Trainer进行训练
```

### 方案B: 直接使用当前模型进行推理

**如果训练困难**:
- 当前模型(iter_12192)已经训练了12k步
- 在220张图片上训练
- Dice ~0.51

**可以**:
- 直接在1220张图片上进行推理
- 评估当前模型性能
- 决定是否真的需要重新训练

## 📊 当前模型性能

**在原始数据集(220张)**:
- 训练迭代: 12,192
- 最终损失: 0.1255 (下降87.45%)
- Dice: ~0.51
- IoU: ~0.36

**在Merged数据集(1220张)的预测**:
- 已完成10张图片的预测
- 模型能够识别血管
- 预测形状连续、平滑

## 🔧 技术细节

### 为什么resize_token_embeddings会OOM？

**过程**:
1. 加载预训练模型 (19.92 GiB)
2. 添加特殊token `[SEG]`, `<p>`, `</p>`, `<vp>`, `</vp>`
3. 需要resize embedding层
4. 计算新token的初始值 (使用现有token的平均值)
5. **这一步需要临时分配2.03 GiB**
6. 但GPU只剩1.82 GiB → OOM

### DeepSpeed Zero-3的限制

**Zero-3优势**:
- 分片模型参数
- 分片梯度
- 分片优化器状态

**Zero-3限制**:
- 某些操作需要gather所有参数
- resize_token_embeddings就是这样的操作
- 在gather时需要完整的模型权重

## 💬 总结

### ✅ 成功完成

1. 数据集准备 (1220张图片)
2. 格式转换
3. 配置文件创建
4. 训练脚本创建

### ❌ 遇到问题

1. CUDA OOM在模型初始化时
2. resize_token_embeddings需要过多显存
3. DeepSpeed Zero-3无法完全解决此问题

### 🎯 建议

**最佳方案**: 使用HuggingFace格式的模型
- 路径: `/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf/`
- 使用HuggingFace Trainer
- 更好的显存管理
- 更成熟的训练流程

**替代方案**: 直接使用当前模型
- 已经训练了12k步
- 性能合理 (Dice 0.51)
- 可以直接在新数据集上评估
- 如果性能足够，无需重新训练

---

**创建时间**: 2025-11-23 03:13 AM
**状态**: 数据准备完成，训练遇到OOM问题
**下一步**: 决定使用HuggingFace训练或直接评估当前模型
