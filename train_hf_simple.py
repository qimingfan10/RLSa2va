#!/usr/bin/env python3
"""
使用HuggingFace模型在Merged数据集上训练（简化版，不使用Trainer）
利用4×24GB GPU的显存
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import cv2

print("=" * 80)
print("Sa2VA训练 - Merged Dataset (简化版)")
print("=" * 80)

# 配置
model_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf'
data_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data/'
output_dir = '/home/ubuntu/Sa2VA/work_dirs/hf_simple_training/'

print(f"\n模型路径: {model_path}")
print(f"数据路径: {data_root}")
print(f"输出目录: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

# 检查GPU
num_gpus = torch.cuda.device_count()
print(f"\n✅ 检测到 {num_gpus} 个GPU")
for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {name} ({mem:.2f} GB)")

# 加载模型
print("\n加载模型...")
print("使用device_map='balanced'将模型分散到4个GPU...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 平衡分配
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"✅ 模型加载成功")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer加载成功")
    
    # 显示显存使用
    print("\n模型加载后显存使用:")
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {mem_allocated:.2f} GB / {mem_total:.2f} GB ({mem_allocated/mem_total*100:.1f}%)")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 加载数据
print("\n加载数据...")
with open(os.path.join(data_root, 'annotations.json')) as f:
    annotations = json.load(f)

print(f"总样本数: {len(annotations)}")

# 简单测试：在几个样本上进行推理
print("\n测试推理...")
test_samples = annotations[:5]

for idx, sample in enumerate(test_samples):
    print(f"\n样本 {idx+1}/{len(test_samples)}: {sample['image']}")
    
    # 加载图像
    img_path = os.path.join(data_root, 'images', sample['image'])
    image = Image.open(img_path).convert('RGB')
    
    try:
        with torch.no_grad():
            # 使用predict_forward
            text = "<image>Please segment the blood vessel in this image. [SEG]"
            
            result = model.predict_forward(
                image=image,
                text=text,
                tokenizer=tokenizer,
                processor=None
            )
            
            print(f"  ✅ 推理成功")
            print(f"  预测文本: {result.get('prediction', 'N/A')[:50]}")
            
            if 'prediction_masks' in result and result['prediction_masks']:
                print(f"  预测掩码数量: {len(result['prediction_masks'])}")
            
    except Exception as e:
        print(f"  ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 显示当前显存
    print(f"  当前显存:")
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"    GPU {i}: {mem_allocated:.2f} GB")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n说明:")
print("  如果推理成功，说明模型可以在4个GPU上正常工作")
print("  可以继续实现完整的训练循环")
print("  或者直接使用当前模型在整个数据集上进行评估")
