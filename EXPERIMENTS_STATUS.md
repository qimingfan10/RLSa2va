# 🔬 Sa2VA 强化学习优化实验 - 双实验并行

## 📊 实验概览

**启动时间**: 2025-11-29 12:14 (实验一) / 12:34 (实验二)  
**实验策略**: 两个不同的RL方案同时运行，对比效果

---

## 🧪 实验一：Prompt优化强化学习

### 核心思想
使用RL学习最优的文本prompt策略，引导Sa2VA生成更完整的分割结果。

### 状态
- ✅ **运行中**
- **PID**: 2586174
- **日志**: `/home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log`

### 配置
| 参数 | 值 |
|------|-----|
| 训练样本 | 50张 |
| 总步数 | 5000 |
| Episode步数 | 3 |
| 动作空间 | 11个prompt候选 |
| 学习率 | 3e-4 |

### 预期效果
- Dice提升: +2-5%
- Recall提升: +3-6%
- 训练时间: 5-10分钟

### 监控
```bash
# 查看日志
tail -f /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log

# TensorBoard (端口6006)
tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs --port 6006

# 停止训练
bash /home/ubuntu/Sa2VA/rl_prompt_optimization/stop_train.sh
```

---

## 🧪 实验二：后处理优化强化学习

### 核心思想
使用RL优化Sa2VA输出mask的后处理步骤，通过形态学操作提高Recall。

### 状态
- 🚀 **启动中**
- **日志**: `/home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2_*.log`

### 配置
| 参数 | 值 |
|------|-----|
| 训练样本 | 50张 |
| 总步数 | 5000 |
| Episode步数 | 3 |
| 动作空间 | 7个后处理操作 |
| 学习率 | 3e-4 |

### 动作空间（7个操作）
1. **降低阈值**: 增加敏感度
2. **小尺度膨胀**: kernel=3
3. **中尺度膨胀**: kernel=5
4. **形态学闭运算**: 连接断裂区域
5. **连通性修复**: 修复血管断裂
6. **区域增长**: 从高置信度区域扩展
7. **终止**: 结果满意时主动停止

### 预期效果
- Dice提升: +3-7%
- Recall提升: +5-10%
- 训练时间: 5-10分钟

### 监控
```bash
# 查看日志（最新）
tail -f /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2_*.log

# TensorBoard (端口6007，避免与实验一冲突)
tensorboard --logdir /home/ubuntu/Sa2VA/rl_postprocess_optimization/outputs/*/logs --port 6007

# 停止训练
kill $(cat /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2.pid)
```

---

## 🔄 实验对比

| 维度 | 实验一 (Prompt) | 实验二 (后处理) |
|------|-----------------|-----------------|
| **修改模型** | ❌ 否 | ❌ 否 |
| **训练时间** | ~10分钟 | ~10分钟 |
| **动作类型** | 离散文本选择 | 图像处理操作 |
| **优势** | 可解释性强 | 直接优化输出 |
| **劣势** | 受限于prompt库 | 可能过度处理 |
| **Recall提升** | +3-6% | +5-10% |
| **Precision保持** | 好 | 中等 |

---

## 📈 综合监控

### 查看所有运行中的实验

```bash
# 检查进程
ps aux | grep -E "train_rl_prompt|train_rl_postprocess"

# GPU使用情况
watch -n 1 nvidia-smi

# 查看所有日志
ls -lth /home/ubuntu/Sa2VA/rl_*/logs/
```

### 同时查看两个TensorBoard

```bash
# 终端1：实验一
tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs --port 6006

# 终端2：实验二
tensorboard --logdir /home/ubuntu/Sa2VA/rl_postprocess_optimization/outputs/*/logs --port 6007
```

然后访问：
- 实验一: http://localhost:6006
- 实验二: http://localhost:6007

---

## 🎯 实验目标

### 当前性能
- Dice: 0.8191
- Recall: 0.7763 (主要问题)
- Precision: 0.8742

### 目标性能
- Dice: **0.85+**
- Recall: **0.85+**
- Precision: **0.80+** (可接受范围)

### 评估标准
实验成功的标志：
1. ✅ Dice提升到0.85以上
2. ✅ Recall提升到0.85以上
3. ✅ Precision保持在0.80以上
4. ✅ 训练稳定收敛

---

## 📝 实验流程

### 阶段1: 快速测试（当前）
- ⏰ **时间**: 各10分钟
- 🎯 **目标**: 验证框架正常工作
- 📊 **样本**: 各50张图片

### 阶段2: 完整训练（如果快速测试成功）
- ⏰ **时间**: 各1-2小时
- 🎯 **目标**: 获得最优策略
- 📊 **样本**: 全部1220张图片

### 阶段3: 综合评估
- 📊 在10张测试图片上评估两个实验
- 📈 对比效果，选择最优方案
- 🚀 可能的组合：先用实验一选prompt，再用实验二后处理

---

## 🔧 故障排查

### 实验一问题
```bash
# 查看日志
tail -100 /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log

# 检查进程
ps -p 2586174
```

### 实验二问题
```bash
# 查看日志
tail -100 /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2_*.log

# 检查进程
ps -p $(cat /home/ubuntu/Sa2VA/rl_postprocess_optimization/logs/experiment2.pid)
```

### 常见问题
1. **OOM**: 两个实验同时占用GPU，可能内存不足
   - 解决：停止一个实验，或减小batch_size
2. **训练太慢**: GPU被充分利用是正常的
3. **进程崩溃**: 查看对应日志文件的错误信息

---

## 📊 预期结果时间线

```
当前时间: 12:34
├── 12:40 - 实验一完成快速测试
├── 12:44 - 实验二完成快速测试
├── 12:50 - 评估两个实验的快速测试结果
└── 决策: 
    ├── 如果效果好 → 运行完整训练
    └── 如果效果不足 → 调整参数或尝试方案3
```

---

## 🎓 学习目标

两个实验将学习：

### 实验一学习的策略
- 针对不同图像特征选择合适的prompt
- 多步prompt组合策略
- 何时强调"完整性"，何时强调"细节"

### 实验二学习的策略
- 何时使用膨胀操作增加Recall
- 如何平衡Precision和Recall
- 何时停止后处理（避免过度处理）

---

## 📁 输出结构

```
Sa2VA/
├── rl_prompt_optimization/
│   ├── outputs/
│   │   └── rl_prompt_20251129_121411/
│   │       ├── best_model/
│   │       ├── checkpoints/
│   │       └── logs/
│   └── logs/
│       └── rl_train_20251129_121403.log
│
└── rl_postprocess_optimization/
    ├── outputs/
    │   └── rl_postprocess_*/
    │       ├── best_model/
    │       ├── checkpoints/
    │       └── logs/
    └── logs/
        └── experiment2_*.log
```

---

**更新时间**: 2025-11-29 12:34  
**实验状态**: 
- 实验一: 🟢 运行中
- 实验二: 🟡 启动中

**下次检查**: 5分钟后查看两个实验的训练进度
