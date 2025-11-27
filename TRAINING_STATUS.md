# 训练状态 - Merged Dataset

## ✅ 问题已修复

### 问题1: 坐标缩放 ✅ 已修复
- 600个样本的坐标从800×800缩放到512×512
- 验证通过，所有坐标在图像范围内

### 问题2: text字段格式错误 ✅ 已修复
**错误**: `text`是字符串 `"blood vessel"`，导致被当作字符列表
**修复**: 改为列表 `["blood vessel"] * len(masks)`
**结果**: mask长度 == text长度

## 🚀 当前训练状态

**启动时间**: 2025-11-23 04:13
**进程ID**: 908937
**日志文件**: `training_20251123_041337.log`

### GPU使用情况
```
GPU 0: 13022 MiB / 24576 MiB (53%)
GPU 1: 13022 MiB / 24576 MiB (53%)
GPU 2: 13022 MiB / 24576 MiB (53%)
GPU 3: 13022 MiB / 24576 MiB (53%)
总计: 52 GB / 96 GB
```

### 训练配置
- **数据集**: 1220张图片
- **Epochs**: 3
- **Batch size**: 1 per GPU
- **梯度累积**: 8步
- **有效batch size**: 32
- **学习率**: 2e-5
- **DeepSpeed**: Zero-3

### 当前阶段
✅ 模型加载完成 (04:16:38)
⏳ DeepSpeed初始化中...

## 📝 监控命令

**查看实时日志**:
```bash
tail -f /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/training_20251123_041337.log
```

**查看GPU状态**:
```bash
nvidia-smi
```

**查看进程**:
```bash
ps aux | grep 908937
```

**查看训练进度** (一旦开始):
```bash
grep -E "(epoch|iter.*loss)" /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/training_20251123_041337.log | tail -20
```

## 📊 预期

- **总步数**: 1220 / 32 ≈ 38步/epoch × 3 epochs ≈ 114步
- **保存频率**: 每500步 (可能只在最后保存)
- **预计时间**: 根据第一次训练，约2-3小时/epoch

## ⚠️ 注意事项

1. DeepSpeed初始化可能需要5-10分钟
2. 第一个batch可能需要更长时间
3. 如果长时间无输出，检查日志末尾是否有错误

---
**最后更新**: 2025-11-23 04:20
