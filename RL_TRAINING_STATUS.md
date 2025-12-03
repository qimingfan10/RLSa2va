# 🚀 Sa2VA RL训练状态

## ✅ 训练已启动

**启动时间**: 2025-11-29 12:14:06  
**训练模式**: 快速测试（Quick Test）  
**进程PID**: 2586174

---

## 📊 训练配置

| 参数 | 值 |
|------|-----|
| **训练样本数** | 50张图片 |
| **总训练步数** | 5000 steps |
| **最大Episode步数** | 3 steps |
| **保存频率** | 每1000步 |
| **评估频率** | 每1000步 |
| **学习率** | 3e-4 |
| **批次大小** | 64 |
| **GPU** | 4个GPU可用 |

---

## 📁 重要文件

```bash
# 训练日志
/home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log

# 进程PID文件
/home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train.pid

# 输出目录
/home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/
```

---

## 🔍 监控训练

### 1. 实时查看日志

```bash
tail -f /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log
```

### 2. 查看训练进程

```bash
bash /home/ubuntu/Sa2VA/rl_prompt_optimization/monitor_train.sh
```

### 3. 查看TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/*/logs

# 然后在浏览器打开
http://localhost:6006
```

### 4. 检查GPU使用

```bash
watch -n 1 nvidia-smi
```

---

## 🎯 预期训练时长

- **快速测试**: 约5-10分钟
- **预期效果**: 验证RL框架是否正常工作
- **后续步骤**: 如果成功，运行完整训练

---

## ⏸️ 停止训练

如需停止训练：

```bash
# 方法1：使用停止脚本
bash /home/ubuntu/Sa2VA/rl_prompt_optimization/stop_train.sh

# 方法2：直接kill进程
kill 2586174

# 方法3：使用PID文件
kill $(cat /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train.pid)
```

---

## 📈 训练完成后

### 1. 查看训练结果

```bash
# 查看输出目录
ls -lh /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/rl_prompt_*/

# 查看最佳模型
ls -lh /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/rl_prompt_*/best_model/
```

### 2. 评估训练好的策略

```bash
python3 /home/ubuntu/Sa2VA/rl_prompt_optimization/evaluate_rl_prompt.py \
    --rl_model_path /home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/rl_prompt_*/best_model/best_model.zip \
    --split val
```

### 3. 运行完整训练

如果快速测试成功：

```bash
bash /home/ubuntu/Sa2VA/rl_prompt_optimization/full_train.sh
```

---

## 🎓 训练指标说明

### TensorBoard中的关键指标

- **ep_rew_mean**: Episode平均奖励（应该上升）
- **ep_len_mean**: Episode平均长度
- **policy_loss**: PPO策略损失
- **value_loss**: PPO价值损失
- **explained_variance**: 解释方差（越接近1越好）

### 自定义指标（如果实现）

- **dice_score**: 平均Dice分数
- **recall_score**: 平均Recall
- **precision_score**: 平均Precision

---

## 📝 训练日志示例

训练日志会显示：
- 模型加载进度
- 数据集加载信息
- PPO训练进度条
- Episode奖励和长度
- 保存checkpoint的提示

---

## 🔧 调试技巧

### 如果训练失败

1. **查看完整日志**:
   ```bash
   cat /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log
   ```

2. **检查错误信息**:
   ```bash
   grep -i "error\|failed\|exception" /home/ubuntu/Sa2VA/rl_prompt_optimization/logs/rl_train_20251129_121403.log
   ```

3. **检查进程状态**:
   ```bash
   ps aux | grep train_rl_prompt
   ```

4. **检查GPU内存**:
   ```bash
   nvidia-smi
   ```

### 常见问题

1. **OOM (内存不足)**: 减少batch_size
2. **训练太慢**: 减少max_samples或total_timesteps
3. **依赖缺失**: pip3 install -r requirements.txt

---

## 🎯 成功标志

训练成功的标志：
- ✅ 进程正常运行（ps -p PID显示进程存在）
- ✅ 日志正常输出（无ERROR或Exception）
- ✅ TensorBoard显示训练曲线
- ✅ 定期保存checkpoint
- ✅ Episode奖励逐渐上升

---

**更新时间**: 2025-11-29 12:14  
**状态**: 🟢 训练中  
**下次检查**: 5分钟后查看训练进度
