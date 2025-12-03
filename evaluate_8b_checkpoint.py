#!/usr/bin/env python3
"""直接用checkpoint评估8B模型"""
import os
import sys
sys.path.insert(0, '/home/ubuntu/Sa2VA')

import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from xtuner.registry import BUILDER

def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def main():
    # 加载配置
    cfg = Config.fromfile('projects/sa2va/configs/sa2va_vessel_lora_finetune_8b_extreme.py')
    
    # 设置预训练权重
    cfg.model.pretrained_pth = '/home/ubuntu/Sa2VA/work_dirs/sa2va_vessel_lora_finetune_8b_extreme/iter_15280.pth'
    
    print("加载模型...")
    model = BUILDER.build(cfg.model)
    model = model.cuda().eval()
    
    # 获取tokenizer
    tokenizer = BUILDER.build(cfg.tokenizer)
    
    # 数据路径
    data_dir = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
    with open(os.path.join(data_dir, "annotations.json")) as f:
        annotations = json.load(f)
    
    num_samples = min(50, len(annotations))
    print(f"评估 {num_samples} 个样本...")
    
    dice_scores = []
    
    for i, ann in enumerate(tqdm(annotations[:num_samples])):
        try:
            img_name = ann['image']
            img_path = os.path.join(data_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, "images", img_name)
            if not os.path.exists(img_path):
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(data_dir, "masks", f"{base_name}_mask.png")
            if not os.path.exists(mask_path):
                continue
            
            gt_mask = np.array(Image.open(mask_path).convert('L')) > 127
            
            # 推理
            text = "<image>\nPlease segment the blood vessel."
            with torch.no_grad():
                result = model.mllm.model.predict_forward(
                    image=image,
                    text=text,
                    tokenizer=tokenizer,
                )
            
            pred_masks = result.get('prediction_masks', [])
            if not pred_masks:
                continue
            
            pred_mask = pred_masks[0]
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]
            
            if pred_mask.shape != gt_mask.shape:
                pred_mask = np.array(Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)) > 127
            
            dice = dice_score(pred_mask.astype(float), gt_mask.astype(float))
            dice_scores.append(dice)
            
        except Exception as e:
            print(f"Error {i}: {e}")
            continue
    
    print("\n" + "="*50)
    print("8B模型评估结果")
    print("="*50)
    print(f"样本数: {len(dice_scores)}")
    print(f"Average Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Min: {np.min(dice_scores):.4f}, Max: {np.max(dice_scores):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
