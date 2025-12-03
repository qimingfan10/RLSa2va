# ⚠️ LoRA SFT训练结果分析

**训练时间**: 2025-11-29 23:27 - 2025-11-30 10:56  
**总耗时**: ~11.5小时  
**状态**: ❌ **训练无效 - 模型参数未更新**

---

## 📊 训练结果

### 各Epoch指标

```
Epoch 1/15:  Train Loss: 0.3149, Train Dice: 0.7416, Val Dice: 0.7342
Epoch 2/15:  Train Loss: 0.3148, Train Dice: 0.7417, Val Dice: 0.7342
Epoch 3/15:  Train Loss: 0.3148, Train Dice: 0.7415, Val Dice: 0.7342
Epoch 4/15:  Train Loss: 0.3147, Train Dice: 0.7416, Val Dice: 0.7342
Epoch 5/15:  Train Loss: 0.3147, Train Dice: 0.7417, Val Dice: 0.7342
Epoch 6/15:  Train Loss: 0.3144, Train Dice: 0.7419, Val Dice: 0.7342
Epoch 7/15:  Train Loss: 0.3146, Train Dice: 0.7418, Val Dice: 0.7342
Epoch 8/15:  Train Loss: 0.3146, Train Dice: 0.7417, Val Dice: 0.7342
Epoch 9/15:  Train Loss: 0.3146, Train Dice: 0.7418, Val Dice: 0.7342
Epoch 10/15: Train Loss: 0.3148, Train Dice: 0.7416, Val Dice: 0.7342
Epoch 11/15: Train Loss: 0.3147, Train Dice: 0.7416, Val Dice: 0.7342
Epoch 12/15: Train Loss: 0.3146, Train Dice: 0.7417, Val Dice: 0.7342
Epoch 13/15: Train Loss: 0.3145, Train Dice: 0.7418, Val Dice: 0.7342
Epoch 14/15: Train Loss: 0.3147, Train Dice: 0.7418, Val Dice: 0.7342
Epoch 15/15: Train Loss: 0.3148, Train Dice: 0.7416, Val Dice: 0.7342

Best Val Dice: 0.7342 (没有提升)
```

---

## ❌ 关键问题

### 1. Val Dice完全没有变化

**所有15个epoch的Val Dice都是0.7342**

这明确证明：
- ✅ 训练循环可以正常运行
- ❌ **LoRA参数没有真正更新**
- ❌ 模型输出完全没有改变

### 2. 根本原因

正如之前分析的，问题在于：

```python
# Sa2VA的predict_forward调用链
predict_forward() 
  → generate()  # 有 @torch.no_grad() 装饰器
    → 整个计算图被禁用
      → 梯度无法回传
```

即使我们手动设置：
```python
pred_prob = pred_prob.detach().requires_grad_(True)
```

这只是让后续的Loss计算不报错，但**梯度无法穿透`@torch.no_grad()`回传到LoRA参数**。

---

## 🎯 正确的解决方案

### 必须重写训练流程

不能使用`predict_forward`，需要直接使用`forward`函数：

```python
# 正确的训练方式
def train_step(model, image, mask, text, tokenizer):
    # 1. 准备训练数据格式
    data = prepare_training_data(
        image=image,
        mask=mask, 
        text=text,
        tokenizer=tokenizer
    )
    # data = {
    #     'pixel_values': ...,
    #     'input_ids': ...,
    #     'labels': ...,
    #     'attention_mask': ...,
    #     'position_ids': ...,
    # }
    
    # 2. 直接调用forward（有梯度）
    outputs = model.forward(data, mode='loss')
    
    # 3. 计算loss
    loss = compute_segmentation_loss(outputs, mask)
    
    # 4. 反向传播（梯度会回传到LoRA参数）
    loss.backward()
    optimizer.step()
```

---

## 📉 当前结果分析

### Val Dice 0.7342 vs 之前的结果

```yaml
当前SFT训练:     Val Dice 0.7342  (未优化的基础模型)
阈值扫描最优:    Val Dice 0.7849  (threshold=0.35)
LoRA PPO (旧):  Val Dice 0.7889  (但也可能有问题)
```

**结论**: 当前的0.7342就是**未经微调的Sa2VA基础模型**在该数据集上的表现。

---

## 🔧 需要做的工作

### 1. 研究Sa2VA的forward函数

查看`modeling_sa2va_chat.py`的`forward()`函数，理解其输入格式：

```python
def forward(self, data, data_samples=None, mode='loss'):
    # 需要的输入
    pixel_values = data['pixel_values']
    input_ids = data['input_ids']
    position_ids = data['position_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']
    # ...
```

### 2. 创建proper的数据加载器

```python
class ProperVesselDataset(Dataset):
    def __getitem__(self, idx):
        # 返回完整的训练格式数据
        return {
            'pixel_values': ...,
            'input_ids': ...,
            'labels': ...,
            'attention_mask': ...,
            'position_ids': ...,
        }
```

### 3. 重写训练循环

```python
def train_epoch(model, dataloader):
    for batch in dataloader:
        # 直接使用forward
        outputs = model.forward(batch, mode='loss')
        loss = outputs.loss  # 或者自定义loss
        loss.backward()
        optimizer.step()
```

---

## 📁 当前训练产出

```
输出目录: /home/ubuntu/Sa2VA/lora_sft_training/output_sft/sft_20251129_232726/
模型: best_model/  (实际上就是未优化的LoRA adapter)

这个模型没有任何价值，因为参数没有更新。
```

---

## 💡 替代方案

如果重写训练太复杂，可以考虑：

### 方案A: 使用现有阈值优化
```yaml
方法: 固定threshold=0.35
结果: Val Dice 0.7849
优势: 简单直接，已验证有效
```

### 方案B: 尝试其他训练框架
- 使用Hugging Face Trainer
- 使用PEFT库的官方训练示例
- 查找Sa2VA的官方训练代码

### 方案C: 联系Sa2VA作者
- 询问如何正确训练
- 获取官方训练脚本

---

## 📊 时间投入 vs 收益

```
已投入时间: ~11.5小时训练 + 调试时间
实际收益:   0 (模型未优化)
学到的:     LoRA训练的正确方式很重要

建议: 
1. 如果急需结果 → 使用阈值优化(0.35) → Dice 0.7849
2. 如果要真正优化 → 重写训练流程 → 可能需要1-2天
3. 如果只是实验 → 已经完成目标（验证了方法的可行性和局限性）
```

---

## ✅ 下一步行动

### 立即可行
1. **使用阈值0.35** - 已验证Dice 0.7849
2. 保存当前结果作为baseline

### 中期目标  
1. 研究Sa2VA的forward函数
2. 重写训练数据加载器
3. 实现正确的训练循环

### 长期目标
1. 真正优化Sa2VA模型
2. 达到Dice 0.84-0.86的目标

---

**总结**: 当前训练虽然运行完成，但由于梯度问题，模型参数未更新，训练无效。需要完全重写训练流程才能真正优化模型。
