# 🚀 LoRA + PPO快速启动指南

**创建时间**: 2025-11-29 15:13  
**状态**: ✅ 代码已就绪，可以开始训练

---

## 📋 已完成的工作

### ✅ 核心代码实现
1. **`reward_functions.py`** - 多目标奖励函数 ⭐⭐⭐⭐⭐
   - MultiObjectiveReward: Dice + Recall + 拓扑 + 长度
   - SimpleDiceReward: 仅Dice
   - RecallFocusedReward: 专注Recall

2. **`lora_config.py`** - LoRA配置
   - 预设配置 (small/medium/large)
   - 自动计算可训练参数
   - 保存/加载/合并功能

3. **`data_loader.py`** - 数据加载
   - 自动数据划分 (train/val/test)
   - 支持数据增强
   - 高效DataLoader

4. **`train_lora_ppo.py`** - 主训练脚本
   - Sa2VA + LoRA集成
   - 简化版PPO训练循环
   - 完整的验证和保存逻辑

5. **`run_lora_ppo.sh`** - 运行脚本
   - Quick模式 (50张, 1 epoch, ~30分钟)
   - Full模式 (1000张, 3 epochs, ~24-48小时)

6. **`install_dependencies.sh`** - 依赖安装
7. **`README.md`** - 完整文档

---

## 🎯 立即开始

### 第1步：安装依赖 (5分钟)

```bash
cd /home/ubuntu/Sa2VA/lora_ppo_training
bash install_dependencies.sh
```

**安装内容**:
- PEFT (LoRA实现)
- scikit-image (拓扑分析)
- OpenCV (图像处理)
- wandb (可选，实验追踪)
- accelerate (训练加速)

### 第2步：快速测试 (30分钟)

```bash
bash run_lora_ppo.sh quick
```

**配置**:
```yaml
训练样本: 50张
验证样本: 20张
训练轮数: 1 epoch
LoRA Rank: 32
学习率: 5e-5
GPU: GPU1
```

**目的**:
- ✅ 验证代码正常运行
- ✅ 检查GPU内存是否充足
- ✅ 确认无bug和错误
- ⚠️ 性能提升有限（样本太少）

### 第3步：查看结果

```bash
# 查看日志
tail -100 output/train_quick_*.log

# 查看训练信息
cat output/sa2va_lora_ppo_*/training_info.json

# 检查最佳模型
ls -lh output/sa2va_lora_ppo_*/best_lora/
```

### 第4步：完整训练 (24-48小时)

```bash
bash run_lora_ppo.sh full
```

**配置**:
```yaml
训练样本: 1000张
验证样本: 100张
训练轮数: 3 epochs
预期Dice: 0.85-0.87+
预期Recall: 0.83-0.85+
```

---

## 📊 监控训练

### 实时监控

```bash
# 查看日志
tail -f output/train_full_*.log

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep train_lora_ppo
```

### 关键指标

训练过程中关注：
- ✅ **train/dice**: 应该逐渐上升
- ✅ **train/recall**: 目标0.85+
- ✅ **val/dice**: 最重要的指标
- ⚠️ **loss/reward**: 应该稳定或上升

---

## 🎯 成功标准

### Quick模式
```
运行成功: 无错误，正常完成
GPU内存: 不超过80GB
训练速度: 每个epoch ~30分钟
```

### Full模式
```
Dice: ≥ 0.87
Recall: ≥ 0.85
Precision: ≥ 0.85
训练稳定: 无NaN或崩溃
```

---

## 🔧 故障排除

### 问题1: 依赖安装失败

```bash
# 使用清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft scikit-image opencv-python
```

### 问题2: GPU内存不足

```bash
# 修改run_lora_ppo.sh
LORA_RANK=16  # 降低rank（从32降到16）
```

### 问题3: 训练太慢

```bash
# 减少验证频率
VAL_FREQ=200  # 从100增加到200
```

### 问题4: Sa2VA加载失败

```bash
# 检查模型路径
ls -lh /home/ubuntu/Sa2VA/models/sa2va_vessel_hf/

# 确认tokenizer正常
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('/home/ubuntu/Sa2VA/models/sa2va_vessel_hf', trust_remote_code=True)"
```

---

## 📈 预期时间线

### Quick模式
```
0:00  ▶ 启动脚本
0:05  ⏳ 安装依赖（如果需要）
0:10  ⏳ 加载Sa2VA模型
0:15  ⏳ 应用LoRA
0:20  ⏳ 加载数据
0:25  🔄 开始训练
0:55  ✅ 训练完成
```

### Full模式
```
0:00   ▶ 启动脚本
0:15   ⏳ 初始化
0:30   🔄 Epoch 1/3
8:00   ⏳ Epoch 1 完成
16:00  ⏳ Epoch 2 完成
24:00  ⏳ Epoch 3 完成
24:30  ✅ 训练完成
```

---

## 🎓 关键技术点

### 1. 奖励函数设计 (最重要)

**当前设计**:
```python
reward = 0.5 * dice_reward +      # 主要优化目标
         0.2 * recall_bonus +     # 针对Recall低的问题
         0.2 * topology_reward +  # 保证血管连续性
         0.1 * length_penalty     # 约束总长度
```

**为什么有效**:
- Dice直接优化分割质量
- Recall bonus专门提升敏感度
- 拓扑奖励减少血管断裂
- 长度约束防止过度预测

### 2. LoRA原理

**核心思想**: 不修改预训练权重，只添加低秩矩阵
```
W = W_frozen + ΔW
ΔW = A × B  # A: (d, r), B: (r, d), r << d
```

**优势**:
- 参数少 (0.5% vs 100%)
- 训练快
- 易于切换和合并
- 保持预训练知识

### 3. RL优化Dice

**为什么RL能做到而监督学习做不到**:
```python
# 监督学习
loss = CrossEntropy(pred, gt)  # 像素级准确率
# 问题：优化的不是Dice

# 强化学习
reward = Dice(pred, gt)  # 直接优化Dice！
loss = -reward
```

---

## 💡 优化建议

### 如果Dice不达标

1. **调整奖励权重**
```bash
# 专注Dice
--dice_weight 0.7 --recall_weight 0.1

# 专注Recall
--dice_weight 0.3 --recall_weight 0.5
```

2. **增加训练数据**
```bash
MAX_TRAIN_SAMPLES=2000  # 使用更多数据
```

3. **提高LoRA rank**
```bash
LORA_RANK=64  # 更多参数
LORA_ALPHA=128
```

4. **Curriculum Learning**
```python
# 修改data_loader.py
# 先训练简单样本（大血管）
# 再训练困难样本（细小血管）
```

### 如果Recall提升但Precision下降

```bash
# 添加Precision约束
# 修改reward_functions.py
if precision < 0.85:
    precision_penalty = (0.85 - precision) * 10.0
```

---

## 📁 输出文件

训练完成后，检查以下文件：

```
output/
└── sa2va_lora_ppo_20251129_xxxxxx/
    ├── best_lora/                    # 最佳模型（验证集）
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── final_lora/                   # 最终模型
    ├── checkpoint_epoch_1/           # Epoch 1 checkpoint
    ├── checkpoint_epoch_2/           # Epoch 2 checkpoint
    ├── checkpoint_epoch_3/           # Epoch 3 checkpoint
    └── training_info.json            # 训练信息
```

**使用最佳模型**:
```bash
# 评估
python evaluate_lora.py \
    --base_model /path/to/sa2va \
    --lora_weights output/sa2va_lora_ppo_*/best_lora

# 推理
python inference_with_lora.py \
    --base_model /path/to/sa2va \
    --lora_weights output/sa2va_lora_ppo_*/best_lora \
    --image /path/to/test_image.jpg
```

---

## 🎉 期待结果

### Quick模式（验证可行性）
```
运行时间: ✅ ~30分钟
代码正常: ✅ 无错误
GPU内存: ✅ 不OOM
Dice提升: ⚠️ 有限（样本少）
```

### Full模式（达成目标）
```
Dice:      0.78 → 0.87+ ✅ (+11.5%)
Recall:    0.74 → 0.85+ ✅ (+14.9%)
Precision: 0.84 → 0.85+ ✅ (+1.2%)
拓扑:      显著改善 ✅
```

---

## 🚀 下一步行动

### 立即执行（今天）

```bash
# 1. 进入目录
cd /home/ubuntu/Sa2VA/lora_ppo_training

# 2. 安装依赖
bash install_dependencies.sh

# 3. 快速测试
bash run_lora_ppo.sh quick

# 4. 检查结果（30分钟后）
tail -100 output/train_quick_*.log
```

### 验证通过后（明天）

```bash
# 启动完整训练
bash run_lora_ppo.sh full

# 后台运行（推荐）
nohup bash run_lora_ppo.sh full > lora_ppo_full.log 2>&1 &

# 查看进程
ps aux | grep train_lora_ppo
```

### 训练完成后（2-3天后）

```bash
# 1. 评估性能
python evaluate_lora_model.py

# 2. 对比所有实验
python compare_all_experiments.py

# 3. 撰写最终报告
# 4. 选择最优方案部署
```

---

## ✅ 检查清单

训练前确认：
- [ ] 依赖已安装 (`bash install_dependencies.sh`)
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 数据集存在 (`ls /home/ubuntu/Sa2VA/data/merged_vessel_data`)
- [ ] 模型存在 (`ls /home/ubuntu/Sa2VA/models/sa2va_vessel_hf`)
- [ ] 磁盘空间充足 (`df -h`)

训练中监控：
- [ ] 进程正常运行 (`ps aux | grep train`)
- [ ] GPU使用率合理 (`nvidia-smi`)
- [ ] 日志无错误 (`tail -f output/train_*.log`)
- [ ] Dice逐渐上升

训练后检查：
- [ ] 最佳模型已保存 (`ls output/sa2va_lora_ppo_*/best_lora`)
- [ ] 训练信息完整 (`cat output/sa2va_lora_ppo_*/training_info.json`)
- [ ] 验证Dice达标 (≥ 0.87)
- [ ] Recall达标 (≥ 0.85)

---

**状态**: ✅ 准备就绪  
**建议**: 立即运行Quick模式验证  
**预期**: Full模式达到Dice 0.87+, Recall 0.85+

**开始命令**: `bash run_lora_ppo.sh quick` 🚀
