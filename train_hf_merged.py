#!/usr/bin/env python3
"""
使用HuggingFace Trainer在Merged数据集上训练Sa2VA模型
利用4×24GB GPU，更好的显存管理
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import cv2

print("=" * 80)
print("Sa2VA HuggingFace Trainer - Merged Dataset")
print("=" * 80)

# 配置
model_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf'
data_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data/'
output_dir = '/home/ubuntu/Sa2VA/work_dirs/hf_merged_training/'

print(f"\n模型路径: {model_path}")
print(f"数据路径: {data_root}")
print(f"输出目录: {output_dir}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 检查GPU
num_gpus = torch.cuda.device_count()
print(f"\n✅ 检测到 {num_gpus} 个GPU")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 自定义数据集
class VesselSegmentationDataset(Dataset):
    def __init__(self, data_root, tokenizer, max_length=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载annotations
        with open(os.path.join(data_root, 'annotations.json')) as f:
            self.annotations = json.load(f)
        
        print(f"  加载了 {len(self.annotations)} 个样本")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        sample = self.annotations[idx]
        
        # 加载图像
        img_path = os.path.join(self.data_root, 'images', sample['image'])
        image = Image.open(img_path).convert('RGB')
        
        # 创建文本输入
        text = "<image>Please segment the blood vessel in this image. [SEG]"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建ground truth mask
        image_np = np.array(image)
        gt_mask = self.polygon_to_mask(sample['mask'][0] if sample['mask'] else [], image_np.shape)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'image': image,
            'gt_mask': torch.from_numpy(gt_mask).float(),
            'image_path': sample['image']
        }
    
    def polygon_to_mask(self, polygon_coords, image_shape):
        """将多边形坐标转换为掩码"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if len(polygon_coords) == 0:
            return mask
        
        points = []
        for i in range(0, len(polygon_coords), 2):
            if i + 1 < len(polygon_coords):
                points.append([polygon_coords[i], polygon_coords[i+1]])
        
        if len(points) > 0:
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        
        return mask

# 加载模型和tokenizer
print("\n加载模型...")
print("这可能需要几分钟...")

try:
    # 使用device_map="auto"自动分配到多个GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 平衡分配到所有GPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"✅ 模型加载成功")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer加载成功")
    
    # 显示显存使用
    print("\n当前显存使用:")
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: 已分配 {mem_allocated:.2f} GB, 已保留 {mem_reserved:.2f} GB")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 创建数据集
print("\n创建数据集...")
train_dataset = VesselSegmentationDataset(data_root, tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # 有效batch size = 4 * 1 * 16 = 64
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_dir=os.path.join(output_dir, 'logs'),
    logging_steps=10,
    save_steps=500,
    save_total_limit=5,
    bf16=True,  # 使用bfloat16
    dataloader_num_workers=2,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,  # 启用梯度检查点以节省显存
    optim="adamw_torch",
    report_to="none",  # 不使用wandb等
)

print("\n训练配置:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size per GPU: {training_args.per_device_train_batch_size}")
print(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
print(f"  有效batch size: {num_gpus * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  学习率: {training_args.learning_rate}")
print(f"  梯度检查点: {training_args.gradient_checkpointing}")

# 创建Trainer
print("\n创建Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始训练
print("\n" + "=" * 80)
print("开始训练...")
print("=" * 80)

try:
    trainer.train()
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")
    
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n最终显存使用:")
for i in range(num_gpus):
    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"  GPU {i}: 已分配 {mem_allocated:.2f} GB, 已保留 {mem_reserved:.2f} GB")
