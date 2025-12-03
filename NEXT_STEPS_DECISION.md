# 🎯 下一步决策指南

**当前状态**: 所有实验已完成训练  
**最佳结果**: LoRA+PPO Full模式 - Dice 0.7889, Recall 0.7617  
**与目标差距**: Dice -7.2%, Recall -10.4%

---

## 🚦 立即行动：评估实验一

**实验一目前状态**:
- ✅ 训练完成（5000 timesteps）
- ❓ 未评估Dice/Recall指标
- 💡 可能已经有不错的效果

**评估命令**:
```bash
cd /home/ubuntu/Sa2VA/rl_prompt_optimization

# 评估最优策略
python evaluate_rl_prompt.py \
    --model_path outputs/rl_prompt_20251129_154906/final_model \
    --sa2va_model /home/ubuntu/Sa2VA/models/sa2va_vessel_hf \
    --test_data /home/ubuntu/Sa2VA/data/merged_vessel_data \
    --num_samples 100 \
    --output_dir evaluation_results
```

**预期结果**:
- 如果Dice ≥ 0.82: 实验一有效，可能结合使用
- 如果Dice < 0.80: 效果不佳，专注LoRA+PPO优化

---

## 🔄 方案A: 优化LoRA+PPO（推荐） ⭐⭐⭐⭐⭐

### 配置改进

创建优化配置文件 `/home/ubuntu/Sa2VA/lora_ppo_training/run_lora_ppo_v2.sh`:

```bash
#!/bin/bash

echo "========================================"
echo "LoRA + PPO 优化版训练"
echo "版本2: 更多数据 + 更高学习率 + 更大rank"
echo "========================================"

MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT="/home/ubuntu/Sa2VA/data/merged_vessel_data"
OUTPUT_DIR="/home/ubuntu/Sa2VA/lora_ppo_training/output_v2"
GPU=1

# 优化后的参数
LORA_RANK=64          # 32 → 64
LORA_ALPHA=128        # 64 → 128
MAX_TRAIN_SAMPLES=1220  # 1000 → 1220（全部数据）
MAX_VAL_SAMPLES=100
NUM_EPOCHS=10         # 3 → 10
LEARNING_RATE=1e-4    # 5e-5 → 1e-4

# 调整奖励权重（强化Recall）
REWARD_TYPE="multi_objective"
DICE_WEIGHT=0.4       # 0.5 → 0.4
RECALL_WEIGHT=0.4     # 0.2 → 0.4
TOPOLOGY_WEIGHT=0.15  # 0.2 → 0.15
LENGTH_WEIGHT=0.05    # 0.1 → 0.05
RECALL_TARGET=0.85

echo "优化配置:"
echo "  LoRA Rank: $LORA_RANK (提升)"
echo "  学习率: $LEARNING_RATE (提升)"
echo "  训练样本: $MAX_TRAIN_SAMPLES (增加)"
echo "  训练轮数: $NUM_EPOCHS (增加)"
echo "  Recall权重: $RECALL_WEIGHT (强化)"
echo ""

mkdir -p $OUTPUT_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/train_v2_${TIMESTAMP}.log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU python3 /home/ubuntu/Sa2VA/lora_ppo_training/train_lora_ppo.py \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_val_samples $MAX_VAL_SAMPLES \
    --reward_type $REWARD_TYPE \
    --dice_weight $DICE_WEIGHT \
    --recall_weight $RECALL_WEIGHT \
    --topology_weight $TOPOLOGY_WEIGHT \
    --length_weight $LENGTH_WEIGHT \
    --recall_target $RECALL_TARGET \
    --prompt "Please segment the blood vessel." \
    --gpu 0 \
    --num_workers 4 \
    --val_freq 100 \
    --save_freq 2 \
    --log_freq 10 \
    2>&1 | tee $LOG_FILE

echo ""
echo "训练完成！"
echo "日志文件: $LOG_FILE"
```

### 启动命令

```bash
cd /home/ubuntu/Sa2VA/lora_ppo_training

# 创建v2脚本
cat > run_lora_ppo_v2.sh << 'EOF'
# (上面的脚本内容)
EOF

chmod +x run_lora_ppo_v2.sh

# 后台运行
nohup bash run_lora_ppo_v2.sh > lora_ppo_v2.log 2>&1 &

# 查看PID
echo $! > lora_ppo_v2.pid
```

### 监控命令

```bash
# 实时日志
tail -f lora_ppo_v2.log

# GPU状态
watch -n 2 nvidia-smi

# 进程状态
ps aux | grep train_lora_ppo

# 停止训练（如果需要）
kill $(cat lora_ppo_v2.pid)
```

### 预期效果

```yaml
训练时间: 10-15小时
预期Dice: 0.84-0.86
预期Recall: 0.82-0.84
成功概率: 85%
```

---

## 🔄 方案B: Curriculum Learning

### 实现步骤

1. **样本排序**
```python
# 创建 prepare_curriculum_data.py
import json
import numpy as np
from PIL import Image

# 按血管面积排序
samples_with_area = []
for sample in annotations:
    mask = generate_mask(sample)
    area = mask.sum()
    samples_with_area.append((sample, area))

# 从大到小排序（简单到困难）
samples_with_area.sort(key=lambda x: x[1], reverse=True)

# 分三个阶段
easy = samples_with_area[:400]      # 大血管
medium = samples_with_area[400:800] # 中等
hard = samples_with_area[800:]      # 细小血管
```

2. **分阶段训练**
```bash
# Stage 1: 大血管（400张，3 epochs）
python train_lora_ppo.py --max_train_samples 400 --num_epochs 3

# Stage 2: 中等血管（800张，3 epochs）
python train_lora_ppo.py --max_train_samples 800 --num_epochs 3 \
    --resume_from stage1_best_model

# Stage 3: 所有血管（1220张，4 epochs）
python train_lora_ppo.py --max_train_samples 1220 --num_epochs 4 \
    --resume_from stage2_best_model
```

### 预期效果

```yaml
优势: 逐步增加难度，更稳定收敛
预期Dice: 0.85-0.87
预期Recall: 0.83-0.85
成功概率: 70%
训练时间: 15-20小时
```

---

## 🔄 方案C: 动态奖励权重

### 实现方式

修改`reward_functions.py`，根据当前性能动态调整权重：

```python
class AdaptiveMultiObjectiveReward:
    def __init__(self, target_recall=0.85):
        self.target_recall = target_recall
        self.current_avg_recall = 0.75  # 初始值
        
    def __call__(self, pred_mask, gt_mask):
        # ... 计算各项指标
        
        # 动态调整权重
        if self.current_avg_recall < self.target_recall - 0.05:
            # Recall太低，大幅增加权重
            recall_weight = 0.5
            dice_weight = 0.3
        elif self.current_avg_recall < self.target_recall:
            # 接近目标，适度增加
            recall_weight = 0.4
            dice_weight = 0.4
        else:
            # 已达标，恢复平衡
            recall_weight = 0.2
            dice_weight = 0.5
        
        # 计算总奖励
        total_reward = (
            dice_weight * dice_reward +
            recall_weight * recall_reward +
            0.2 * topology_reward +
            0.1 * length_penalty
        )
        
        return total_reward, reward_dict
```

---

## 📊 决策树

```
开始
  │
  ├─→ 评估实验一
  │     │
  │     ├─→ Dice ≥ 0.82? 
  │     │     ├─→ YES: 结合Prompt+LoRA
  │     │     └─→ NO: 继续LoRA优化
  │     │
  │     └─→ 时间: 1小时
  │
  ├─→ 方案A: 优化LoRA+PPO ⭐⭐⭐⭐⭐
  │     ├─→ 调整超参数
  │     ├─→ 增加数据和轮数
  │     ├─→ 强化Recall权重
  │     └─→ 时间: 10-15小时
  │          │
  │          ├─→ 成功（Dice 0.84+）: 完成
  │          └─→ 未达标: 方案B或C
  │
  ├─→ 方案B: Curriculum Learning
  │     ├─→ 实现分阶段训练
  │     ├─→ 逐步增加难度
  │     └─→ 时间: 15-20小时
  │
  └─→ 方案C: 动态奖励权重
        ├─→ 修改奖励函数
        ├─→ 自适应权重调整
        └─→ 时间: 12-18小时
```

---

## 🎯 推荐执行顺序

### 今天（2025-11-29晚上）

#### 1. 评估实验一（1小时）
```bash
cd /home/ubuntu/Sa2VA/rl_prompt_optimization
# 运行评估脚本（需要先创建）
```

#### 2. 启动方案A（立即后台运行）
```bash
cd /home/ubuntu/Sa2VA/lora_ppo_training
nohup bash run_lora_ppo_v2.sh > lora_ppo_v2.log 2>&1 &
```

### 明天（2025-11-30）

#### 上午
- 检查方案A训练进度
- 查看TensorBoard曲线
- 评估前几个epoch的效果

#### 下午
- 如果效果好，等待完成
- 如果效果不佳，准备方案B或C

### 后天（2025-12-01）

- 评估最终结果
- 撰写完整报告
- 部署最优模型

---

## 💡 关键技巧

### 1. 如何判断训练是否有效？

观察前3个epoch：
```yaml
有效的信号:
  - Dice持续上升（至少+0.01/epoch）
  - Recall上升幅度 > Dice
  - Loss稳定下降

无效的信号:
  - Dice几乎不变（<0.005/epoch）
  - Recall不上升或下降
  - Loss震荡或不下降

如果无效:
  - 停止训练
  - 调整学习率或奖励权重
  - 重新启动
```

### 2. 如何快速测试超参数？

使用Mini验证：
```bash
# 100张图像，1 epoch，快速验证超参数
python train_lora_ppo.py \
    --max_train_samples 100 \
    --num_epochs 1 \
    --learning_rate 1e-4  # 测试新学习率
```

如果1 epoch后Dice提升明显 → 超参数有效 → 完整训练

### 3. 如何避免过拟合？

```yaml
策略:
  - 使用数据增强
  - Early stopping（验证Dice不再提升时停止）
  - 定期在验证集上评估
  - 保存多个checkpoint对比
```

---

## 📞 最终建议

### 立即执行 ✅

**Option 1: 保守策略（推荐给时间紧的情况）**
```bash
1. 评估实验一（1小时）
2. 如果达标，完成项目
3. 如果未达标，执行方案A
```

**Option 2: 激进策略（推荐给追求极致的情况）**
```bash
1. 立即启动方案A（后台运行10-15小时）
2. 同时评估实验一
3. 对比两者结果，选择最优
```

### 我的推荐 ⭐⭐⭐⭐⭐

**立即执行Option 2（激进策略）**

原因：
1. 方案A训练时间长，越早开始越好
2. 评估实验一只需1小时，可并行进行
3. 两个结果都有，选择余地大
4. 成功概率最高

具体命令：
```bash
# 终端1: 启动方案A
cd /home/ubuntu/Sa2VA/lora_ppo_training
nohup bash run_lora_ppo_v2.sh > lora_ppo_v2.log 2>&1 &

# 终端2: 评估实验一（需要先创建评估脚本）
cd /home/ubuntu/Sa2VA/rl_prompt_optimization
# python evaluate_rl_prompt.py ...

# 监控
tail -f /home/ubuntu/Sa2VA/lora_ppo_training/lora_ppo_v2.log
```

---

**决策指南生成时间**: 2025-11-29 17:48  
**当前状态**: 等待下一步决策  
**推荐行动**: 立即启动方案A（优化LoRA+PPO） 🚀
