# Sa2VA训练模型推理指南

## 📊 当前状态

### ✅ 已完成
- 训练完成 (3672步，Loss: 13.76 → 1.08)
- 模型权重已保存 (`iter_3672.pth`, 2.5GB)
- Ground Truth可视化已生成 (5个样本)

### 📁 可视化结果位置
```bash
/home/ubuntu/Sa2VA/inference_results/predictions/
├── sample_1_Chen_Fang_0000103366__1-4_1_04B2D3CF_frame_000034.jpg
├── sample_2_Bai_Hui_Min_0000202318__1-3_1_04DB6FD9_frame_000045.jpg
├── sample_3_Gong_Chao_0000838952__1-2_1_0487E196_frame_000033.jpg
├── sample_4_Feng_Wan_Chang_0000889954__1-3_1_04CE6CAA_frame_000009.jpg
└── sample_5_Fang_Kun__0000470101__1-3_1_04A2C7DE_frame_000059.jpg
```

每张图片包含4个子图：
1. **Original** - 原始图像
2. **Ground Truth** - 标注的真实mask (红色)
3. **Prediction** - 模型预测结果 (绿色)
4. **Overlay** - 叠加对比 (红色=GT, 绿色=预测)

---

## 🚀 使用训练权重进行实际推理

由于Sa2VA是mmengine格式的模型，有以下几种推理方案：

### 方案1: 转换为HuggingFace格式 (推荐) ⭐

#### 步骤1: 转换模型

```bash
cd /home/ubuntu/Sa2VA

# 使用官方转换脚本
python tools/convert_to_hf.py \
    --model_path projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    --ckpt_path work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save_path models/sa2va_vessel_hf
```

**注意**: 此步骤需要在`topo-sarl`环境中运行，因为需要mmengine。

#### 步骤2: 使用HuggingFace模型推理

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# 加载模型
model = AutoModel.from_pretrained(
    "models/sa2va_vessel_hf", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "models/sa2va_vessel_hf",
    trust_remote_code=True
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 加载图片
image = Image.open("path/to/image.jpg").convert('RGB')

# 推理
with torch.no_grad():
    result = model.predict_forward(
        image=image,
        text="blood vessel",
        tokenizer=tokenizer
    )
    
    # 获取预测mask
    pred_masks = result['prediction_masks']
```

---

### 方案2: 使用mmengine Runner

如果在`topo-sarl`环境中，可以直接使用mmengine：

```python
from mmengine.config import Config
from mmengine.runner import Runner
import torch

# 加载配置
cfg = Config.fromfile('projects/sa2va/configs/sa2va_merged_vessel_finetune.py')

# 设置checkpoint
cfg.load_from = 'work_dirs/merged_vessel_segmentation/iter_3672.pth'

# 创建runner
runner = Runner.from_cfg(cfg)

# 进行推理
# (需要根据Sa2VA的具体API实现)
```

---

### 方案3: 使用官方评估脚本

Sa2VA提供了评估脚本，可以直接使用：

```bash
cd /home/ubuntu/Sa2VA

# 使用官方评估脚本
python projects/sa2va/evaluation/sa2va_eval_refcoco.py \
    --model_path models/sa2va_vessel_hf \
    --data_path data/merged_vessel_data \
    --output_dir evaluation_results
```

---

## 🔧 环境要求

### 转换模型需要
```bash
# 在topo-sarl环境中
- mmengine
- torch
- transformers
- 完整的Sa2VA依赖
```

### HuggingFace推理需要
```bash
# 可以在普通Python环境
- transformers
- torch
- PIL
- numpy
```

---

## 📝 快速开始脚本

### 一键转换和推理

```bash
chmod +x /home/ubuntu/Sa2VA/convert_and_inference.sh
/home/ubuntu/Sa2VA/convert_and_inference.sh
```

这个脚本会：
1. 检查HuggingFace模型是否存在
2. 如果不存在，转换mmengine模型
3. 使用转换后的模型进行推理
4. 生成可视化结果

---

## 🎯 当前可视化说明

当前`inference_results/predictions/`中的图片是**Ground Truth可视化**，因为：

1. **环境限制**: 系统Python环境没有mmengine
2. **临时方案**: 先可视化GT作为参考
3. **下一步**: 需要转换模型或在正确环境中运行

### 可视化格式

每张图片包含4个面板：
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Original   │ Ground Truth│ Prediction  │   Overlay   │
│             │   (红色)     │  (绿色)      │  (红+绿)     │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

当前Prediction面板显示的是Ground Truth（因为模型未加载）。

---

## 🔍 验证模型权重

### 检查checkpoint内容

```python
import torch

ckpt = torch.load(
    'work_dirs/merged_vessel_segmentation/iter_3672.pth',
    map_location='cpu',
    weights_only=False
)

print("Checkpoint keys:", ckpt.keys())
print("Meta info:", ckpt.get('meta', {}))
print("State dict keys:", list(ckpt['state_dict'].keys())[:10])
```

### 预期输出
```
Checkpoint keys: dict_keys(['state_dict', 'meta', 'optimizer', ...])
Meta info: {'iter': 3672, 'epoch': 3, ...}
State dict keys: ['mllm.model.embed_tokens.weight', ...]
```

---

## 📊 推理性能预估

基于训练配置：

| 指标 | 值 |
|------|-----|
| 模型大小 | 2.5GB |
| 推荐显存 | ≥24GB (单GPU) |
| FP16推理 | ~12GB显存 |
| 推理速度 | ~1-2秒/图 (512×512) |
| Batch推理 | 支持 (根据显存) |

---

## ⚠️ 常见问题

### Q1: 为什么当前可视化显示的是Ground Truth？

**A**: 因为系统Python环境缺少mmengine，模型无法加载。需要：
- 在`topo-sarl`环境中运行，或
- 转换为HuggingFace格式后推理

### Q2: 如何在topo-sarl环境中运行？

**A**: 
```bash
# 方法1: 使用micromamba
~/micromamba/bin/micromamba run -n topo-sarl python inference_with_trained_model.py

# 方法2: 激活环境后运行
eval "$(~/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl
python inference_with_trained_model.py
```

### Q3: 转换失败怎么办？

**A**: 检查：
1. 是否在topo-sarl环境中
2. mmengine是否正确安装
3. checkpoint路径是否正确
4. 配置文件是否存在

### Q4: 如何评估模型性能？

**A**: 
```python
from sklearn.metrics import jaccard_score, f1_score

# 计算IoU
iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten())

# 计算Dice Score
dice = f1_score(gt_mask.flatten(), pred_mask.flatten())

print(f"IoU: {iou:.4f}")
print(f"Dice: {dice:.4f}")
```

---

## 📚 相关文档

- `TRAINING_ANALYSIS_REPORT.md` - 训练详细分析
- `TRAINING_COMPLETE_SUMMARY.md` - 训练完整总结
- `inference_results/README.md` - 推理结果说明
- `tools/convert_to_hf.py` - 模型转换脚本

---

## 🎓 推荐工作流程

### 完整推理流程

```bash
# 1. 转换模型 (在topo-sarl环境)
python tools/convert_to_hf.py \
    --model_path projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    --ckpt_path work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save_path models/sa2va_vessel_hf

# 2. 推理 (可以在任何环境)
python hf_inference_script.py

# 3. 评估
python evaluate_predictions.py

# 4. 可视化
python visualize_results.py
```

---

## 💡 下一步建议

1. **立即可做**:
   - 查看当前GT可视化: `ls -lh inference_results/predictions/`
   - 阅读训练分析报告

2. **需要环境**:
   - 转换模型为HuggingFace格式
   - 进行实际模型推理

3. **进阶任务**:
   - 在测试集上全面评估
   - 计算IoU、Dice等指标
   - 与baseline模型对比

---

**文档更新时间**: 2025-11-25  
**模型版本**: iter_3672.pth  
**训练Loss**: 13.76 → 1.08 (↓92.2%)
