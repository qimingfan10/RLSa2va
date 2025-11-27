# 🔍 HuggingFace转换过程问题分析

## 🚨 **问题发现**

通过深入检查，我们发现了为什么两个不同的训练checkpoint（权重相差96%）转换成HF模型后推理性能完全相同。

---

## 📊 **证据链**

### 1️⃣ 原始Checkpoint确实不同

```bash
检查结果:
├── 共同参数: 536个
├── 完全相同: 20个 (3.73%)
└── 不同参数: 516个 (96.27%) ✅

关键参数差异:
- lm_head.weight: 平均差异 1.84e-04, 最大差异 9.22e-03
- sam2_decoder:   平均差异 1.48e-03, 最大差异 1.15e-02
```

**结论**: 两个checkpoint权重明显不同！

### 2️⃣ HF模型权重却完全相同

```bash
HF模型第一个safetensors文件对比:
- embed_tokens.weight:        差异 0.000000e+00 ❌
- layers.0.input_layernorm:   差异 0.000000e+00 ❌  
- layers.0.mlp.down_proj:     差异 0.000000e+00 ❌
- layers.0.mlp.gate_proj:     差异 0.000000e+00 ❌
- layers.0.mlp.up_proj:       差异 0.000000e+00 ❌
```

**结论**: 转换后的HF模型权重完全相同！

### 3️⃣ 配置文件相同

```bash
config.json对比:
✅ llm_config: 相同
✅ vision_config: 相同
✅ template: 相同
✅ architectures: 相同
```

---

## 🔎 **根本原因分析**

### 转换脚本执行流程

文件：`tools/convert_to_hf.py`

```python
# 步骤1: 加载配置
cfg = Config.fromfile(args.config)  # 读取配置文件

# 步骤2: 构建模型 ⚠️ 关键点
model = BUILDER.build(cfg.model)
# 这会调用 Sa2VAModel.__init__()
# 在__init__中会执行:
#   if pretrained_pth is not None:
#       pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
#       self.load_state_dict(filtered_state_dict, strict=False)
#       print(f'Load pretrained weight from {pretrained_pth}')

# 步骤3: 加载训练checkpoint
state_dict = torch.load(args.pth_model, map_location='cpu')['state_dict']

# 步骤4: 再次load_state_dict ⚠️ 关键点
model.load_state_dict(state_dict, strict=False)
print(f'Load PTH model from {args.pth_model}')
```

### 问题所在

**步骤2的详细过程** (from `sa2va.py`):

```python
def __init__(self, ..., pretrained_pth=None):
    # ... 初始化各种模块 ...
    
    if pretrained_pth is not None:  # ← 这里！
        pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v  # ← 加载预训练权重
        
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f'Load pretrained weight from {pretrained_pth}')
```

### 配置文件内容

**两个配置都指向同一个预训练权重**:

```python
# sa2va_vessel_finetune.py (旧模型配置)
pretrained_pth = "/home/ubuntu/Sa2VA-26B.pth"

# sa2va_merged_vessel_finetune.py (新模型配置)
pretrained_pth = "/home/ubuntu/Sa2VA-26B.pth"
```

---

## 🎯 **问题总结**

### 转换流程

```
旧模型转换 (iter_12192.pth):
├── 1. 读取 sa2va_vessel_finetune.py
├── 2. 构建模型 → 加载 Sa2VA-26B.pth ✅
├── 3. 加载 iter_12192.pth (strict=False)
│      └── 只更新训练过的参数
└── 4. 保存为 HF格式

新模型转换 (iter_3672.pth):
├── 1. 读取 sa2va_merged_vessel_finetune.py
├── 2. 构建模型 → 加载 Sa2VA-26B.pth ✅ (同一个！)
├── 3. 加载 iter_3672.pth (strict=False)
│      └── 只更新训练过的参数
└── 4. 保存为 HF格式
```

### 为什么权重相同？

**两个HF模型的大部分权重来自同一个`Sa2VA-26B.pth`！**

1. **基础权重相同**: 两次转换都先加载了`Sa2VA-26B.pth`
2. **训练权重覆盖有限**: `strict=False`只更新训练过的参数
3. **如果训练使用了LoRA**: 大部分backbone参数没有被训练更新
4. **结果**: 两个HF模型的主要参数（如embed_tokens, layer weights）保持相同

---

## 📈 **转换日志验证**

### 旧模型转换日志 (convert_to_hf.log)

```
11/22 14:09:00 - INFO - Loads checkpoint from: pretrained/sam2/sam2_hiera_large.pt
11/22 14:09:01 - INFO - Loaded checkpoint successfully

Load pretrained weight from /home/ubuntu/Sa2VA-26B.pth  ← 先加载这个！

Skipped 3 mismatched keys:
- text_hidden_fcs.0.weight: checkpoint shape [6144, 6144] vs model shape [3584, 3584]
- text_hidden_fcs.0.bias: checkpoint shape [6144] vs model shape [3584]
- text_hidden_fcs.2.weight: checkpoint shape [256, 6144] vs model shape [256, 3584]

Load PTH model from work_dirs/vessel_segmentation/iter_12192.pth  ← 然后加载训练checkpoint
```

**分析**:
- 明确显示先加载了`Sa2VA-26B.pth`
- 然后才加载训练checkpoint
- 训练checkpoint只更新了匹配的参数

---

## 🤔 **为什么Checkpoint不同但HF相同？**

### 可能的解释

#### 1️⃣ **使用了LoRA训练** (最可能)

如果训练使用了LoRA适配器:
- 只训练LoRA参数（小模块）
- backbone参数冻结不训练
- checkpoint中的差异可能主要在LoRA参数
- 但转换HF时可能合并了LoRA或只保存了backbone

#### 2️⃣ **不同训练阶段的同一模型**

- iter_12192 和 iter_3672可能是同一个训练的不同阶段
- 配置文件名不同但实际训练过程相同
- 只是在不同epoch保存的checkpoint

#### 3️⃣ **训练权重被覆盖**

```python
# 伪代码说明问题
model.load_state_dict(Sa2VA_26B, strict=False)  # 先加载基础权重
model.load_state_dict(training_ckpt, strict=False)  # 再加载训练权重

# 如果 training_ckpt 只包含少量参数更新
# 大部分权重仍然是 Sa2VA_26B 的值
```

---

## ✅ **验证方法**

### 当前正在进行

**100张图片大规模评估**:
- 目的: 确认是否有微小差异未被10张样本检测到
- 预计时间: 20-30分钟
- 状态: 进行中...

### 建议的额外验证

1. **检查训练是否使用LoRA**
   ```bash
   grep -r "lora" /home/ubuntu/Sa2VA/projects/sa2va/configs/
   ```

2. **对比HF模型的所有权重文件**
   ```python
   # 加载并对比所有7个safetensors文件
   # 查看是否有任何参数不同
   ```

3. **查看训练配置中的冻结参数**
   ```python
   # 检查哪些参数被冻结
   # 哪些参数实际被训练
   ```

4. **直接用checkpoint推理**
   ```python
   # 绕过HF转换
   # 直接加载checkpoint进行推理
   # 看是否有差异
   ```

---

## 🎯 **结论**

### 确认的事实

1. ✅ **原始checkpoint确实不同** (96%参数有差异)
2. ✅ **HF转换后的模型权重相同** (前5个参数差异为0)
3. ✅ **两次转换都加载了同一个Sa2VA-26B.pth**
4. ✅ **转换过程使用strict=False允许部分覆盖**

### 最可能的原因

**转换过程没有正确保留训练checkpoint的差异**

- 由于先加载`Sa2VA-26B.pth`，后加载训练checkpoint
- 且使用`strict=False`
- 如果训练只更新了部分参数（如LoRA）
- 大部分参数保持了`Sa2VA-26B.pth`的值
- 导致两个HF模型实质上是相同的

### 等待100张评估结果

如果100张图片评估结果仍然相同，则基本确认：
**两个HF模型实际上是相同的，转换过程存在问题。**

---

## 🛠️ **建议的修复方案**

### 方案1: 不加载预训练权重

修改配置文件，在转换时不加载`Sa2VA-26B.pth`:

```python
# 临时配置用于转换
pretrained_pth = None  # 不加载预训练权重
```

### 方案2: 修改转换脚本

```python
# 在 convert_to_hf.py 中
# 在构建模型之前临时移除 pretrained_pth
cfg.model.pretrained_pth = None
model = BUILDER.build(cfg.model)

# 然后加载完整的训练checkpoint
model.load_state_dict(state_dict, strict=False)
```

### 方案3: 强制加载训练checkpoint

```python
# 使用更严格的加载策略
model.load_state_dict(state_dict, strict=True)
# 或者确保训练checkpoint完全覆盖模型参数
```

---

**更新时间**: 2025-11-25 19:00  
**状态**: 100张图片评估进行中...  
**下一步**: 等待评估结果确认假设
