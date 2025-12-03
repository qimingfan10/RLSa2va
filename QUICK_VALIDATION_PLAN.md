# 🚀 快速验证计划：阈值扫描实验

**启动时间**: 2025-11-29 15:06  
**验证目标**: 确定是否只需调整阈值就能达到Dice 0.85+

---

## 🎯 验证思路

根据您在 `思路.md` 中提到的建议：

> 在动手写复杂的 RL 代码前，建议先做一个**快速验证（Sanity Check）**：
> 编写一个脚本，在验证集上遍历 `Threshold` 从 0.1 到 0.9（步长 0.05）。
> * 如果在 Threshold = 0.3 时，Dice 能从 0.819 提升到 0.83，**那么您根本不需要训练复杂的 RL 模型**，只需要一个动态阈值算法。
> * 如果 Threshold 调整无效，再考虑 **路径二 (LoRA + PPO)**。

---

## 📋 实验设计

### 核心问题
**当前瓶颈**: Recall (0.77) < Precision (0.87)  
**假设**: 模型预测的概率图中，很多血管像素的置信度在0.3-0.5之间，被默认的0.5阈值过滤掉了。

### 验证方法
1. 使用Sa2VA生成概率图（保持原始概率值，不做二值化）
2. 扫描阈值范围：0.1 → 0.9（步长0.05）
3. 对每个阈值，计算所有样本的平均Dice/Recall/Precision
4. 找到最佳阈值，评估提升效果

### 实验配置
```yaml
模型: Sa2VA (sa2va_vessel_hf)
数据集: merged_vessel_data
样本数: 50张图像
Prompt: "Please segment the blood vessel."
阈值范围: [0.1, 0.9)
阈值步长: 0.05
GPU: GPU1
```

---

## 📊 预期结果

### 情况A: 阈值调整有效 ✅
**现象**: 最佳阈值 ≈ 0.3-0.4，Dice提升到0.83-0.85+

**结论**: 
- ✅ **无需复杂的RL训练**
- ✅ 直接使用动态阈值算法
- ✅ 成本低、效果好、易部署

**下一步**:
1. 实现动态阈值选择算法（基于图像特征）
2. 可选：用简单的回归模型预测最优阈值
3. 部署上线

### 情况B: 阈值调整效果有限 ⚠️
**现象**: 最佳阈值下Dice仅提升0.01-0.02

**结论**:
- ⚠️ 单纯调整阈值不足以解决问题
- ⚠️ 需要更深层次的优化

**下一步**:
1. **路径二**: LoRA + PPO微调（直接优化Dice指标）
2. **路径三**: 结合阈值 + 形态学后处理的RL Agent

### 情况C: 阈值调整负面影响 ❌
**现象**: 降低阈值后Recall提升但Precision大幅下降，Dice反而降低

**结论**:
- ❌ 模型预测质量问题
- ❌ 需要从模型层面优化

**下一步**:
1. **路径二**: LoRA + PPO微调（必选）
2. 引入拓扑连通性奖励

---

## 🔍 关键指标观察

### 1. Dice Score曲线
- **目标**: 找到峰值对应的阈值
- **期望**: 峰值 ≥ 0.85

### 2. Recall曲线
- **观察**: 阈值降低时Recall是否显著提升
- **期望**: Recall达到0.85+

### 3. Precision曲线
- **观察**: Recall提升时Precision下降程度
- **平衡点**: 找到Dice最优的平衡点

### 4. Dice vs Baseline
```
Baseline (threshold=0.5): Dice ≈ 0.82
目标: 提升到 0.85+
增量: +0.03 (3.7%提升)
```

---

## ⏱️ 预计时间

```
0:00  ▶ 启动脚本
0:30  ⏳ 加载模型
1:00  ⏳ 加载数据集
2:00  ⏳ Sa2VA预测 (50张 × 30s ≈ 25分钟)
27:00 ⏳ 阈值扫描 (17个阈值 × 10s ≈ 3分钟)
30:00 ✅ 生成结果和曲线图
```

**预计总时长**: 约30分钟

---

## 📁 输出文件

### 结果文件
```
threshold_validation_output/
├── threshold_scan_results.json  # 详细结果
└── threshold_scan_curves.png    # 可视化曲线
```

### JSON结构
```json
{
  "best_threshold": {
    "threshold": 0.35,
    "dice": 0.8567,
    "recall": 0.8821,
    "precision": 0.8334,
    "iou": 0.7502
  },
  "baseline_threshold": {
    "threshold": 0.5,
    "dice": 0.8191,
    "recall": 0.7763,
    "precision": 0.8742,
    "iou": 0.6940
  },
  "all_results": [...]
}
```

---

## 📊 可视化曲线

### 四个子图
1. **Dice vs Threshold**: 主要关注
2. **Recall vs Threshold**: 观察Recall提升
3. **Precision vs Threshold**: 观察Precision权衡
4. **综合对比**: 三条曲线叠加

### 关键标记
- 🔴 红色虚线: 目标线 (0.85)
- ⭐ 红色星号: 最佳阈值点

---

## 🎯 决策树

```
运行阈值扫描
    │
    ├─ 最佳Dice ≥ 0.85?
    │   ├─ YES → ✅ 使用动态阈值，无需RL
    │   │         实现简单回归模型预测阈值
    │   │
    │   └─ NO → Dice提升 > 0.02?
    │           ├─ YES → ⚠️ 结合动态阈值 + RL
    │           │         先部署阈值优化，再考虑RL微调
    │           │
    │           └─ NO → ❌ 阈值调整无效
    │                     必须进行LoRA + PPO微调
```

---

## 💡 技术洞察

### 为什么先验证阈值？

1. **成本效益**: 
   - 阈值调整: 0成本（纯推理时调整）
   - RL训练: 高成本（GPU训练+复杂实现）

2. **工程实用性**:
   - 阈值调整: 易部署、实时调整
   - RL模型: 部署复杂、需要额外模型

3. **问题诊断**:
   - 如果阈值有效 → 模型预测质量OK，只是后处理问题
   - 如果阈值无效 → 模型预测质量问题，需深层优化

### 动态阈值算法设计

如果阈值扫描验证有效，可以实现：

```python
# 简单版本：基于图像特征的阈值回归
def predict_optimal_threshold(image_features):
    """
    输入: 图像特征 (亮度、对比度、血管密度估计等)
    输出: 最优阈值
    """
    # 训练一个简单的线性回归或决策树
    threshold = model.predict(image_features)
    return threshold

# 高级版本：基于概率图统计的自适应阈值
def adaptive_threshold(prob_map):
    """
    输入: 概率图
    输出: 自适应阈值
    """
    # 分析概率直方图
    hist = np.histogram(prob_map, bins=100)
    
    # 找到双峰分布的谷底（Otsu方法的思想）
    threshold = otsu_method(hist)
    
    return threshold
```

---

## 🚀 实验运行状态

**当前状态**: 🟡 运行中

**监控命令**:
```bash
# 查看进程
ps aux | grep quick_threshold_validation

# 实时监控（如果有日志）
tail -f threshold_validation_output/validation.log
```

**预计完成时间**: 2025-11-29 15:36 (约30分钟后)

---

## 📝 验证完成后的行动

### 立即分析
1. 查看 `threshold_scan_results.json`
2. 检查可视化曲线图
3. 对比最佳阈值 vs baseline

### 撰写报告
1. 总结阈值扫描结果
2. 评估三条路径的适用性
3. 给出明确的技术方案推荐

### 下一步实施
根据验证结果，选择以下方案之一：
- **方案A**: 实现动态阈值算法（如果验证通过）
- **方案B**: 进行LoRA + PPO微调（如果阈值无效）
- **方案C**: 结合方案A+B（如果阈值部分有效）

---

**验证脚本**: `/home/ubuntu/Sa2VA/quick_threshold_validation.py`  
**运行脚本**: `/home/ubuntu/Sa2VA/run_quick_validation.sh`  
**输出目录**: `/home/ubuntu/Sa2VA/threshold_validation_output/`

🎯 **验证目标**: 用最小成本确定最优优化路径
