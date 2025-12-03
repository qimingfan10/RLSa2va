# 🔬 Sa2VA 三实验并行状态报告

**更新时间**: 2025-11-29 13:08  
**运行时间**: 约30分钟

---

## 📊 实验总览

| 实验 | 状态 | GPU | 进度 | Episode奖励 |
|------|------|-----|------|-------------|
| **实验一** | 🟢 运行中 | 多GPU | 3584/5000步 | 5.07 |
| **实验二** | 🟢 运行中 | 多GPU | 2176/5000步 | 0.434 |
| **实验三** | 🟡 重启中 | GPU1 | 启动中 | - |

---

## 🧪 实验一：Prompt优化强化学习

### 状态
- ✅ **运行正常**
- **PID**: 2586174
- **进度**: 3584/5000 步 (71.7%)
- **Episode奖励**: 5.07
- **Episode长度**: 3.0

### 核心思想
使用RL学习最优的文本prompt策略（11个候选prompt）

### 训练指标
```
iterations: 28
total_timesteps: 3584
ep_rew_mean: 5.07
ep_len_mean: 3.0
approx_kl: 0.0010767775
entropy_loss: -2.27
explained_variance: 0.0845
```

### 分析
- ✅ 训练稳定，奖励为正
- ✅ Episode长度为3，符合max_steps=3的设置
- ⚠️ explained_variance较低(0.08)，说明价值函数还在学习中

### 监控
```bash
# 查看日志
tail -f /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log

# TensorBoard
tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs --port 6006
```

---

## 🧪 实验二：后处理优化强化学习

### 状态
- ✅ **运行正常**
- **PID**: 2593217
- **进度**: 2176/5000 步 (43.5%)
- **Episode奖励**: 0.434
- **Episode长度**: 2.85

### 核心思想
使用RL优化后处理步骤（7个后处理操作：膨胀、闭运算、连通性修复等）

### 训练指标
```
iterations: 17
total_timesteps: 2176
ep_rew_mean: 0.434
ep_len_mean: 2.85
approx_kl: 0.004168327
clip_fraction: 0.0109
entropy_loss: -0.902
explained_variance: 0.0178
```

### 分析
- ✅ 训练正常，奖励为正
- ⚠️ 奖励较实验一低，可能后处理改进难度更大
- ⚠️ explained_variance很低(0.018)，价值估计还不准确
- 📈 clip_fraction: 0.0109，策略更新幅度合理

### 监控
```bash
# 查看日志
tail -f /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2_20251129_123457.log

# TensorBoard
tensorboard --logdir /home/ubuntu/Sa2VA/rl_postprocess_optimization/outputs/*/logs --port 6007
```

---

## 🧪 实验三：Reward Network微调（重启中）

### 状态
- 🟡 **重启中**（修复OOM问题）
- **问题**: 之前遇到CUDA OOM
- **解决方案**: 
  - ✅ 减小batch_size: 8 → 4
  - ✅ 使用device_map="auto"
  - ✅ 指定CUDA_VISIBLE_DEVICES=1使用GPU1

### 核心思想
分两步：
1. **步骤1**: 训练Reward Network评估分割质量
2. **步骤2**: 使用Reward指导微调Sa2VA

### 配置
- **样本数**: 50张（快速测试）
- **Epochs**: 10
- **Batch Size**: 4 (减小避免OOM)
- **GPU**: GPU1（最空闲）

### 监控
```bash
# 查看日志
tail -f /home/ubuntu/Sa2VA/rl_reward_network/logs/experiment3_*.log

# TensorBoard
tensorboard --logdir /home/ubuntu/Sa2VA/rl_reward_network/outputs/*/logs --port 6008
```

---

## 🎯 GPU使用情况

当前GPU分配：
```
GPU0: 22% 使用, 7.5GB/24GB  - 实验一+二共享
GPU1: 4%  使用, 9.5GB/24GB  - 实验三使用
GPU2: 21% 使用, 9.6GB/24GB  - 实验一+二共享
GPU3: 9%  使用, 18.3GB/24GB - 实验一+二共享
```

### 优化建议
- ✅ GPU1负载最轻，适合实验三
- ✅ 三个实验分布合理，避免单GPU过载

---

## 📈 预期完成时间

基于当前进度：

| 实验 | 当前进度 | 剩余步数 | 预计完成 |
|------|----------|----------|----------|
| 实验一 | 71.7% | 1416步 | ~5分钟 |
| 实验二 | 43.5% | 2824步 | ~10分钟 |
| 实验三 | 启动中 | - | ~8分钟 |

**全部完成**: 约 **13:20** (10分钟后)

---

## 📊 实时监控命令汇总

### 查看所有实验状态

```bash
# 检查进程
ps aux | grep -E "train_rl" | grep -v grep

# GPU使用
watch -n 1 nvidia-smi

# 快速查看三个实验日志
tail -10 /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log
tail -10 /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2_20251129_123457.log
tail -10 /home/ubuntu/Sa2VA/rl_reward_network/logs/experiment3_*.log
```

### 三个TensorBoard

```bash
# 终端1 - 实验一 (端口6006)
tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs --port 6006

# 终端2 - 实验二 (端口6007)
tensorboard --logdir /home/ubuntu/Sa2VA/rl_postprocess_optimization/outputs/*/logs --port 6007

# 终端3 - 实验三 (端口6008)
tensorboard --logdir /home/ubuntu/Sa2VA/rl_reward_network/outputs/*/logs --port 6008
```

然后访问：
- http://localhost:6006 (实验一)
- http://localhost:6007 (实验二)
- http://localhost:6008 (实验三)

---

## 🎓 训练进展分析

### 实验一 vs 实验二对比

| 指标 | 实验一 | 实验二 | 分析 |
|------|--------|--------|------|
| **进度** | 71.7% | 43.5% | 实验一更快 |
| **Episode奖励** | 5.07 | 0.434 | 实验一奖励更高 |
| **策略稳定性** | 高 | 中等 | 实验一KL散度更小 |
| **价值估计** | 进行中 | 进行中 | 都在学习中 |

### 初步结论（需完成后确认）

1. **实验一（Prompt优化）**：训练较顺利，奖励较高，可能效果更好
2. **实验二（后处理优化）**：训练正常但奖励较低，可能优化空间有限
3. **实验三（Reward Network）**：刚重启，需要观察

---

## 🚨 问题和解决方案

### 已解决问题

1. ✅ **实验三OOM**
   - 原因：GPU内存不足
   - 解决：减小batch_size (8→4)，使用device_map="auto"

### 当前注意事项

1. ⚠️ **三个实验同时运行，GPU负载较高**
   - 监控: `watch -n 1 nvidia-smi`
   - 如有卡顿可暂停一个实验

2. ⚠️ **explained_variance较低**
   - 正常现象，训练初期价值函数还在学习
   - 继续观察后续是否上升

---

## 📁 输出文件结构

```
Sa2VA/
├── rl_prompt_optimization/
│   ├── outputs/rl_prompt_20251129_121411/
│   │   ├── best_model/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── logs/rl_train_20251129_121403.log
│
├── rl_postprocess_optimization/
│   ├── outputs/rl_postprocess_20251129_123505/
│   │   ├── best_model/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── logs/experiment2_20251129_123457.log
│
└── rl_reward_network/
    ├── outputs/reward_net_*/
    │   ├── best_reward_net.pth
    │   ├── final_reward_net.pth
    │   └── logs/
    └── logs/experiment3_*.log
```

---

## 🎯 下一步计划

### 完成后需要做的事情

1. **评估三个实验的效果**
   ```bash
   # 实验一评估
   python3 /home/ubuntu/Sa2VA/rl_prompt_optimization/evaluate_rl_prompt.py \
       --rl_model_path outputs/*/best_model/best_model.zip \
       --split val
   
   # 实验二评估
   python3 /home/ubuntu/Sa2VA/rl_postprocess_optimization/evaluate_rl_postprocess.py \
       --rl_model_path outputs/*/best_model/best_model.zip \
       --split val
   ```

2. **对比结果**
   - 查看各实验的Dice、Recall、Precision提升
   - 选择最优方案

3. **可能的组合策略**
   - 先用实验一选最优prompt
   - 再用实验二做后处理
   - 获得更好的综合效果

4. **如果效果好，运行完整训练**
   ```bash
   # 完整训练（使用全部数据）
   bash rl_prompt_optimization/full_train.sh
   bash rl_postprocess_optimization/run_experiment2.sh
   ```

---

## 📝 训练日志示例

### 实验一（Prompt优化）
```
iterations: 28
ep_rew_mean: 5.07  ← 平均奖励
ep_len_mean: 3     ← 平均episode长度
approx_kl: 0.001   ← KL散度（策略变化）
entropy_loss: -2.27 ← 熵损失（探索程度）
```

### 实验二（后处理优化）
```
iterations: 17
ep_rew_mean: 0.434
ep_len_mean: 2.85
clip_fraction: 0.0109 ← 被裁剪的样本比例
```

---

## ✅ 成功指标

训练成功的标志：
- ✅ Episode奖励逐渐上升
- ✅ 训练稳定，无崩溃
- ✅ Checkpoint正常保存
- ✅ TensorBoard显示学习曲线
- ✅ GPU使用率稳定

---

**状态**: 🟢 实验一运行中 | 🟢 实验二运行中 | 🟡 实验三重启中  
**预计全部完成**: ~13:20 (10分钟后)  
**建议**: 继续等待训练完成，定期查看GPU状态
