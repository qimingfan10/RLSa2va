# 🚀 4-GPU LoRA训练配置

**时间**: 2025-11-30 13:42  
**状态**: 🟢 已启动4-GPU训练

---

## 📊 配置概览

### 硬件配置
```yaml
GPUs: 4 × RTX 3090 (24GB each)
总显存: 96GB
策略: DeepSpeed ZeRO-2 + CPU Offload
```

### 训练配置
```yaml
Batch配置:
  Per-device batch size: 1
  Gradient accumulation: 2
  有效batch size: 4 GPU × 1 × 2 = 8

数据:
  样本数: 1220
  重复次数: 5
  总训练样本: 6100

Epochs: 10
学习率: 1e-4
```

### DeepSpeed配置

**文件**: `deepspeed_zero2_offload.json`

```json
{
  "zero_optimization": {
    "stage": 2,  // ZeRO-2: 分片optimizer states
    "offload_optimizer": {
      "device": "cpu",  // Optimizer状态卸载到CPU
      "pin_memory": true
    }
  },
  "bf16": {
    "enabled": true  // 使用BF16混合精度
  }
}
```

**优势**:
- 减少GPU显存占用50%+
- Optimizer states存储在CPU
- 支持更大的模型和batch size

---

## 🔧 训练命令

### 启动训练
```bash
cd /home/ubuntu/Sa2VA

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DEEPSPEED=deepspeed_zero2_offload \
nohup bash tools/dist.sh train \
  projects/sa2va/configs/sa2va_vessel_lora_finetune.py 4 \
  > vessel_lora_training_4gpu.log 2>&1 &
```

### 监控命令
```bash
# 查看日志
tail -f vessel_lora_training_4gpu.log

# 查看loss
grep "loss" vessel_lora_training_4gpu.log | grep "iter"

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep train.py
```

### 停止训练
```bash
# 温柔停止
pkill -f train.py

# 强制停止
pkill -9 -f train.py
```

---

## 📈 预期性能

### 显存使用
```yaml
单GPU显存占用:
  模型权重: ~8GB
  Activations: ~4GB
  梯度: ~2GB (ZeRO-2分片)
  Optimizer: 卸载到CPU
  总计: ~14GB/24GB ✅

4 GPU总显存: ~56GB/96GB
```

### 训练速度
```yaml
单步时间: ~3-5秒
每epoch: ~40-60分钟
总时间 (10 epochs): 7-10小时

对比单GPU:
  单GPU: 20-30小时
  4 GPU: 7-10小时
  加速比: 2-3×
```

### Loss预期
```yaml
初始:
  loss_mask: ~2.0
  loss_dice: ~0.5
  llm_loss: ~1.0
  total: ~3.5

收敛 (epoch 10):
  loss_mask: ~0.3-0.5
  loss_dice: ~0.1-0.2
  llm_loss: ~0.3-0.5
  total: ~0.8-1.2

目标: Val Dice > 0.80
```

---

## 🔍 监控指标

### 关键日志
```bash
# 查看初始化
grep "GRADIENT STATUS" vessel_lora_training_4gpu.log

# 查看数据加载
grep "Loading" vessel_lora_training_4gpu.log | tail -20

# 查看训练步骤
grep "iter:" vessel_lora_training_4gpu.log | tail -20

# 查看错误
grep -i "error\|exception\|traceback" vessel_lora_training_4gpu.log
```

### Checkpoint
```bash
# 查看保存的checkpoint
ls -lh work_dirs/sa2va_vessel_lora_finetune/

# 每500步保存一次
# 最多保留3个checkpoint
```

---

## ⚠️ 故障排查

### 如果还是OOM

**方案1: 减少batch size**
```python
# 修改配置文件
batch_size = 1
accumulative_counts = 1  # 从2改为1
max_length = 4096  # 从8192减半
```

**方案2: 启用更激进的offload**
```json
{
  "zero_optimization": {
    "stage": 3,  // ZeRO-3: 分片所有参数
    "offload_param": {
      "device": "cpu"
    },
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

**方案3: 减少数据加载**
```python
dataloader_num_workers = 0  # 不使用多进程
repeats = 2  # 进一步减少重复
```

### 如果训练太慢

```python
# 增加accumulation
accumulative_counts = 4  # 更大的有效batch

# 减少验证频率
save_steps = 1000  # 从500改为1000

# 减少epoch
max_epochs = 5  # 从10改为5
```

### 如果进程卡住

```bash
# 检查所有GPU进程
fuser -v /dev/nvidia*

# 清理僵尸进程
pkill -9 -f train.py
pkill -9 -f deepspeed

# 清理NCCL
pkill -9 -f nccl
```

---

## 📊 与单GPU对比

| 指标 | 单GPU | 4 GPU | 提升 |
|------|-------|-------|------|
| 显存占用 | 23GB (OOM) | 14GB/GPU | -40% |
| 训练速度 | OOM | ~4-5s/iter | N/A |
| 总时间 | N/A | 7-10小时 | N/A |
| 有效batch | 4 | 8 | 2× |

---

## ✅ 检查清单

训练启动后检查：

- [ ] 4个GPU都在使用（nvidia-smi）
- [ ] 显存占用<20GB/GPU
- [ ] 日志中有"iter: X, loss: Y"
- [ ] 没有OOM错误
- [ ] Loss在下降
- [ ] Checkpoint正常保存

---

## 🎯 成功标志

```bash
# 训练正常的日志示例
iter: 10, loss_mask: 1.8, loss_dice: 0.45, llm_loss: 0.9, total: 3.15
iter: 20, loss_mask: 1.6, loss_dice: 0.42, llm_loss: 0.85, total: 2.87
iter: 30, loss_mask: 1.5, loss_dice: 0.40, llm_loss: 0.80, total: 2.70
...

# GPU使用正常
GPU 0: 14GB / 24GB
GPU 1: 14GB / 24GB
GPU 2: 14GB / 24GB
GPU 3: 14GB / 24GB
```

---

## 📁 生成的文件

```
/home/ubuntu/Sa2VA/
├── vessel_lora_training_4gpu.log       # 训练日志
├── deepspeed_zero2_offload.json        # DeepSpeed配置
├── work_dirs/
│   └── sa2va_vessel_lora_finetune/
│       ├── iter_500.pth                 # Checkpoint
│       ├── iter_1000.pth
│       └── ...
└── projects/sa2va/configs/
    └── sa2va_vessel_lora_finetune.py   # 训练配置
```

---

## 🚀 下一步

1. **等待初始化完成** (~2-3分钟)
   - 模型加载到所有GPU
   - DeepSpeed初始化
   - 数据加载器准备

2. **监控前100步**
   - 确认loss开始下降
   - 检查显存稳定
   - 验证速度符合预期

3. **长期监控**
   - 每小时检查一次进度
   - 预计7-10小时完成
   - 关注Val Dice指标

---

**当前状态**: 🟢 训练已启动  
**日志位置**: `vessel_lora_training_4gpu.log`  
**预计完成**: ~10小时后
