# 🎓 Sa2VA官方LoRA微调方法分析

**发现时间**: 2025-11-30 13:15  
**状态**: ✅ **找到正确的训练方法！**

---

## 🔍 关键发现

### 1. 训练脚本结构

```bash
# 官方训练命令
bash tools/dist.sh train projects/sa2va/configs/sa2va_finetune.py 8
```

**工作原理**：
- 使用`tools/train.py` (实际调用`xtuner.tools.train`)
- 配置文件：`projects/sa2va/configs/sa2va_finetune.py`
- 支持分布式训练（8 GPU）

---

## 🏗️ 模型架构（Sa2VAModel）

### 关键组件

```python
class Sa2VAModel(BaseModel):
    def __init__(self):
        # 1. MLLM (多模态大语言模型)
        self.mllm = InternVLMLLM(
            freeze_llm=True,           # ❄️ 冻结LLM
            freeze_visual_encoder=True, # ❄️ 冻结视觉编码器
            llm_lora=LoraConfig(        # ✅ 只训练LoRA
                r=128,
                lora_alpha=256,
                lora_dropout=0.05,
                task_type='CAUSAL_LM',
                modules_to_save=["embed_tokens", "lm_head"]
            )
        )
        
        # 2. SAM2 Grounding Encoder
        self.grounding_encoder = SAM2TrainRunner()
        self.grounding_encoder.requires_grad_(False)  # ❄️ 默认冻结
        
        # 3. SAM2 Mask Decoder（可选训练）
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
        
        # 4. 文本到视觉映射
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )  # ✅ 可训练
        
        # 5. Loss函数
        self.loss_mask = CrossEntropyLoss(loss_weight=2.0)
        self.loss_dice = DiceLoss(loss_weight=0.5)
```

---

## 🎯 训练配置（sa2va_finetune.py）

### LoRA参数

```python
llm_lora=dict(
    type=LoraConfig,
    r=128,              # LoRA rank
    lora_alpha=256,     # LoRA alpha
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    modules_to_save=["embed_tokens", "lm_head"]  # 额外训练的模块
)
```

### 训练超参数

```python
batch_size = 2              # per device
accumulative_counts = 16    # 8 GPUs × 2 = 实际batch=32
max_epochs = 1
lr = 4e-5
weight_decay = 0.05
warmup_ratio = 0.05
max_length = 8192
```

### Loss配置

```python
loss_mask = dict(
    type=CrossEntropyLoss,
    use_sigmoid=True,
    reduction='mean',
    loss_weight=2.0
)

loss_dice = dict(
    type=DiceLoss,
    use_sigmoid=True,
    activate=True,
    reduction='mean',
    naive_dice=True,
    eps=1.0,
    loss_weight=0.5
)
```

---

## 📊 数据格式（annotations.json）

```json
[
    {
        "image": "image001.jpg",
        "text": ["blood vessel", "artery"],
        "mask": [
            [[x1,y1,x2,y2,...], [...]],  // polygon for object 1
            [[x1,y1,x2,y2,...]]           // polygon for object 2
        ]
    }
]
```

**处理流程**：
1. 读取image和mask
2. 将polygon转换为binary mask
3. 创建对话格式：
   ```python
   "<image>\nPlease segment the blood vessel. [SEG]"
   "Sure, [SEG]."
   ```
4. Tokenize并编码

---

## 🔄 训练流程（forward函数）

### 关键代码片段

```python
def forward(self, data_samples):
    # 1. 前向传播MLLM
    llm_output = self.mllm(
        input_ids=data_samples['input_ids'],
        pixel_values=data_samples['pixel_values'],
        labels=data_samples['labels'],  # 用于计算language loss
    )
    
    # 2. 提取[SEG] token的hidden states
    seg_hidden_states = extract_seg_hidden_states(
        llm_output.hidden_states,
        output_ids,
        seg_token_idx
    )
    
    # 3. 通过text_hidden_fcs映射
    seg_embeddings = self.text_hidden_fcs(seg_hidden_states)
    
    # 4. SAM2编码器生成特征
    sam_states = self.grounding_encoder.get_sam2_embeddings(
        data_samples['extra_pixel_values']
    )
    
    # 5. 注入language embedding并生成mask
    pred_masks = self.grounding_encoder.inject_language_embd(
        sam_states, seg_embeddings
    )
    
    # 6. 计算mask loss
    loss_mask = self.loss_mask(pred_masks, gt_masks)
    loss_dice = self.loss_dice(pred_masks, gt_masks)
    
    # 7. 总loss = language_loss + mask_loss + dice_loss
    total_loss = llm_output.loss + loss_mask + loss_dice
    
    return {'loss': total_loss}
```

---

## ⚡ 关键区别：训练 vs 推理

| 方面 | 训练（forward） | 推理（predict_forward） |
|------|----------------|----------------------|
| 函数 | `forward()` | `predict_forward()` |
| 模式 | `model.train()` | `model.eval()` |
| 梯度 | ✅ 有 | ❌ 无（@torch.no_grad） |
| 输入 | 完整训练数据 | 单张图像+文本 |
| 输出 | Loss | 分割mask |
| Token生成 | 使用labels（teacher forcing） | 使用generate()采样 |
| SAM2 | 直接计算loss | 推理生成mask |

---

## 🆚 为什么我们的训练失败了

### 我们的方法 ❌

```python
# 使用predict_forward（推理函数）
result = model.predict_forward(image, text, tokenizer, return_tensors=True)
pred = result['probability_maps'][0][0]
loss = criterion(pred, gt_mask)
loss.backward()  # ❌ 梯度无法回传
```

**问题**：
1. `predict_forward`是推理流程
2. 内部使用`generate()`（即使移除@no_grad，也是离散采样）
3. SAM2部分没有梯度
4. 无法真正优化参数

### 官方方法 ✅

```python
# 使用forward（训练函数）
data = {
    'input_ids': ...,
    'pixel_values': ...,
    'extra_pixel_values': ...,
    'labels': ...,
    'masks': ...
}
outputs = model.forward(data)
loss = outputs['loss']  # 内部已计算好
loss.backward()  # ✅ 梯度正确回传
optimizer.step()
```

**优势**：
1. 使用teacher forcing（labels直接指导）
2. SAM2 decoder可以训练
3. text_hidden_fcs可以训练
4. 完整的梯度路径

---

## 📝 可训练参数

```python
✅ LoRA参数（LLM）         ~41M
✅ embed_tokens（token嵌入）
✅ lm_head（输出层）
✅ text_hidden_fcs          ~2M
✅ SAM2 mask decoder       ~4M（可选）

❄️ LLM backbone（冻结）
❄️ Vision encoder（冻结）
❄️ SAM2 encoder（冻结）

总可训练：~45-50M参数（约占总参数的1-2%）
```

---

## 🎨 数据准备示例

### 转换我们的数据格式

```python
import json
import glob
import numpy as np
from PIL import Image

annotations = []
for img_path in glob.glob('Segment_DATA_Merged_512/images/*.jpg'):
    img_name = os.path.basename(img_path)
    mask_path = img_path.replace('images', 'masks').replace('.jpg', '_mask.png')
    
    # 读取mask
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # 转换为polygon（简化版）
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygon = contours[0].flatten().tolist()
    
    annotations.append({
        'image': img_name,
        'text': ['blood vessel'],
        'mask': [[polygon]]
    })

with open('annotations.json', 'w') as f:
    json.dump(annotations, f)
```

---

## 🚀 使用官方方法训练的步骤

### 1. 准备数据

```bash
data/my_data/
  ├── images/
  │   ├── image001.jpg
  │   └── ...
  └── annotations.json
```

### 2. 修改配置

编辑`projects/sa2va/configs/sa2va_finetune.py`：
```python
path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"  # 你的模型路径
pretrained_pth = None  # 或指向预训练权重
RES_ROOT = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/'
```

### 3. 运行训练

```bash
cd /home/ubuntu/Sa2VA
bash tools/dist.sh train projects/sa2va/configs/sa2va_finetune.py 1  # 单GPU
# 或
bash tools/dist.sh train projects/sa2va/configs/sa2va_finetune.py 4  # 4 GPU
```

---

## ⚠️ 注意事项

1. **内存要求**：
   - 单GPU batch_size=2 需要~24GB显存
   - 建议使用梯度累积（accumulative_counts）

2. **数据格式**：
   - annotations.json必须严格按照格式
   - polygon坐标必须是valid的

3. **预训练权重**：
   - 最好从官方预训练模型开始
   - 或设置`pretrained_pth=None`从头训练

4. **训练时间**：
   - 1000样本 × 100 repeats = 100k steps
   - 单GPU约需2-3天

---

## 🎯 总结

| 方面 | 我们的方法 | 官方方法 |
|------|----------|---------|
| 函数 | predict_forward | forward |
| 配置 | 手动脚本 | mmengine配置 |
| 数据 | 简单加载 | 完整pipeline |
| Loss | 手动计算 | 内置计算 |
| LoRA | PEFT库 | XTuner集成 |
| 训练器 | PyTorch原生 | XTuner Trainer |
| 结果 | ❌ 失败 | ✅ 可行 |

---

## 💡 下一步建议

### 选项A: 使用官方训练方法 ⭐

**时间**: 1-2天  
**难度**: 中等  
**收益**: 可能达到目标性能

**步骤**：
1. 准备annotations.json
2. 修改配置文件
3. 运行官方训练脚本

### 选项B: 使用阈值优化

**时间**: 立即  
**难度**: 简单  
**收益**: Val Dice 0.7849（已验证）

---

**推荐**: 如果时间紧，使用**阈值优化**。如果要真正训练，使用**官方训练方法**。
