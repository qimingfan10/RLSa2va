# 🎯 Sa2VA DPO 训练指南

## 概述

DPO (Direct Preference Optimization) 是一种无需Critic网络的强化学习方法，特别适合大模型微调。

### PPO vs DPO 对比

| 特性 | PPO | DPO |
|------|-----|-----|
| 需要Critic | ✅ 需要训练UNet | ❌ 不需要 |
| 显存需求 | ~48GB | ~24GB |
| 训练稳定性 | 较差 | 较好 |
| BFloat16兼容 | ❌ 有问题 | ✅ 完美兼容 |
| 数据格式 | (state, action, reward) | (chosen, rejected) |

---

## 🔧 文件结构

```
/home/ubuntu/Sa2VA/
├── projects/sa2va/
│   ├── configs/
│   │   └── sa2va_dpo_vessel.py      # DPO训练配置
│   ├── datasets/
│   │   └── dpo_vessel_dataset.py    # DPO数据集类
│   └── models/
│       └── sa2va_dpo_model.py       # DPO模型wrapper
├── scripts/
│   └── generate_dpo_dataset.py      # 偏好对生成脚本
├── train_dpo_vessel.sh              # 训练启动脚本
└── data/
    └── dpo_vessel/                  # DPO数据集
        ├── dpo_annotations.json     # 偏好对标注
        └── masks/                   # 生成的mask
```

---

## 📊 数据集格式

### Annotations JSON格式

```json
[
  {
    "image": "images/image_001.jpg",
    "chosen_mask": "masks/image_001_chosen_0_1.png",
    "rejected_mask": "masks/image_001_rejected_0_1.png",
    "chosen_iou": 0.85,
    "rejected_iou": 0.62,
    "iou_gap": 0.23,
    "prompt": "<image>Please segment the blood vessels."
  },
  ...
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | str | 原始图像路径 |
| `chosen_mask` | str | 胜者mask路径（IoU更高）|
| `rejected_mask` | str | 败者mask路径（IoU更低）|
| `chosen_iou` | float | 胜者IoU值 |
| `rejected_iou` | float | 败者IoU值 |
| `iou_gap` | float | IoU差距 |
| `prompt` | str | 输入prompt |

---

## 🚀 使用步骤

### Step 1: 生成DPO数据集

#### 方式A: 使用模型生成多样化预测

```bash
cd /home/ubuntu/Sa2VA

python scripts/generate_dpo_dataset.py \
    --mode generate \
    --images_dir /home/ubuntu/Sa2VA/data/merged_vessel_data/images \
    --gt_dir /home/ubuntu/Sa2VA/data/merged_vessel_data/masks \
    --output_dir /home/ubuntu/Sa2VA/data/dpo_vessel \
    --model_path /home/ubuntu/Sa2VA/models/sa2va_vessel_hf \
    --num_samples 5 \
    --min_iou_gap 0.05
```

#### 方式B: 从已有的多种预测结果生成

如果您已经有多个模型/方法的预测结果：

```bash
python scripts/generate_dpo_dataset.py \
    --mode from_predictions \
    --images_dir /path/to/predictions_dir \
    --gt_dir /path/to/gt_masks \
    --output_dir /home/ubuntu/Sa2VA/data/dpo_vessel \
    --min_iou_gap 0.05
```

### Step 2: 启动DPO训练

```bash
cd /home/ubuntu/Sa2VA
bash train_dpo_vessel.sh
```

或手动运行：

```bash
# 单GPU
python tools/train.py projects/sa2va/configs/sa2va_dpo_vessel.py \
    --work-dir work_dirs/dpo_vessel_training

# 多GPU (4卡)
torchrun --nproc_per_node=4 \
    tools/train.py projects/sa2va/configs/sa2va_dpo_vessel.py \
    --work-dir work_dirs/dpo_vessel_training \
    --launcher pytorch
```

### Step 3: 评估结果

训练完成后，模型会保存在 `work_dirs/dpo_vessel_training/`

```bash
# 推理测试
python tools/test.py \
    projects/sa2va/configs/sa2va_dpo_vessel.py \
    work_dirs/dpo_vessel_training/iter_XXX.pth \
    --work-dir work_dirs/dpo_vessel_eval
```

---

## ⚙️ 关键配置参数

### DPO超参数 (sa2va_dpo_vessel.py)

```python
# DPO核心参数
beta = 0.1              # 温度参数，控制偏好强度
                        # 小 → 更激进地偏好chosen
                        # 大 → 更保守

label_smoothing = 0.0   # 标签平滑，防止过拟合

# 学习率
lr = 5e-6               # DPO通常使用更小的学习率

# Epoch
max_epochs = 2          # DPO收敛快，通常1-3个epoch足够
```

### 数据集参数

```python
min_iou_gap = 0.05      # 最小IoU差距阈值
                        # 太小 → 偏好信号弱
                        # 太大 → 数据量少
```

### DeepSpeed配置

```python
strategy = dict(
    type='DeepSpeedStrategy',
    zero_optimization=dict(
        stage=2,        # DPO可用ZeRO-2（无需Critic，模型更小）
        offload_optimizer=dict(device='cpu'),
    ),
    bf16=dict(enabled=True),  # ✅ BFloat16兼容
)
```

---

## 📈 DPO Loss详解

### 数学公式

**完整版（有reference model）:**
```
L_DPO = -E[log σ(β * ((log π(y_w|x) - log π_ref(y_w|x)) - 
                       (log π(y_l|x) - log π_ref(y_l|x))))]
```

**简化版（LoRA模式，无reference model）:**
```
L_DPO = -E[log σ(β * (log π(y_w|x) - log π(y_l|x)))]
```

其中：
- `π`: 当前策略（正在训练的模型）
- `π_ref`: 参考策略（冻结的模型）
- `y_w`: chosen（胜者）
- `y_l`: rejected（败者）
- `β`: 温度参数

### 对于分割任务的适配

```python
# 计算mask的log概率
log_prob = sum(y_i * log(p_i) + (1-y_i) * log(1-p_i)) / N

# y_i: GT mask的第i个像素
# p_i: 预测的第i个像素概率
# N: 像素总数
```

---

## 🔄 训练监控

### 关键指标

| 指标 | 含义 | 理想趋势 |
|------|------|----------|
| `dpo_loss` | DPO损失 | 下降 |
| `accuracy` | 模型偏好chosen的准确率 | 上升到~0.7-0.9 |
| `margin` | chosen和rejected reward差距 | 上升 |
| `chosen_rewards` | chosen的隐式奖励 | 上升 |
| `rejected_rewards` | rejected的隐式奖励 | 稳定或下降 |

### TensorBoard查看

```bash
tensorboard --logdir work_dirs/dpo_vessel_training
```

---

## 💡 技巧与建议

### 1. 数据质量比数量重要

- IoU差距要足够大（推荐 > 0.1）
- chosen确实要比rejected好（人工检查）
- 避免噪声标签

### 2. Beta调整

```python
# 如果模型不学习偏好 → 增大beta
beta = 0.2

# 如果模型过度偏好chosen（崩溃）→ 减小beta
beta = 0.05
```

### 3. 学习率

DPO对学习率敏感，推荐从小开始：
```python
lr = 1e-6  # 开始
lr = 5e-6  # 正常
lr = 1e-5  # 较大
```

### 4. LoRA vs Full Fine-tuning

- **LoRA（推荐）**: 显存少，不需要reference model
- **Full**: 需要reference model，显存翻倍

---

## ❓ 常见问题

### Q1: 为什么不需要Critic？

DPO直接从偏好对学习，不需要显式估计value/reward。数学上证明DPO等价于带约束的reward maximization。

### Q2: 如何选择beta？

- 默认0.1是个好起点
- 如果训练不稳定，尝试0.05
- 如果偏好学习太慢，尝试0.2

### Q3: 需要多少偏好对？

- 最少：100-500对（可以开始训练）
- 推荐：1000-5000对
- 更多：如果IoU差距较小

### Q4: DPO会过拟合吗？

DPO比PPO更不容易过拟合，但仍需注意：
- 使用early stopping
- 监控validation loss
- 不要训练太多epoch

---

## 📁 输出文件

```
work_dirs/dpo_vessel_training/
├── iter_200.pth         # 检查点
├── iter_400.pth
├── iter_XXX.pth         # 最终模型
├── 20231201_XXXXXX/     # 日志目录
│   ├── vis_data/
│   └── XXXXXX.log
└── config.py            # 保存的配置
```

---

## ✅ 总结

DPO是Sa2VA强化学习微调的最佳选择：

1. ✅ **不需要Critic** - 显存减半
2. ✅ **BFloat16兼容** - 复用MMEngine框架
3. ✅ **训练稳定** - 无需复杂的PPO调参
4. ✅ **数据高效** - 直接从IoU比较学习

**下一步**: 运行 `bash train_dpo_vessel.sh` 开始训练！
