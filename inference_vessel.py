#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
血管分割推理脚本
使用训练好的权重对数据集图片进行预测
"""

import os
import sys
import torch
import argparse
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Sa2VA Vessel Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='训练好的checkpoint路径')
    parser.add_argument('--image-dir', type=str, default='/home/ubuntu/Sa2VA/data/vessel_data/images', 
                        help='图片目录')
    parser.add_argument('--output-dir', type=str, default='./inference_results', 
                        help='输出目录')
    parser.add_argument('--num-images', type=int, default=10, 
                        help='要推理的图片数量')
    parser.add_argument('--base-model', type=str, 
                        default='/home/ubuntu/huggingface_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9',
                        help='基础模型路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='推理设备 (cpu 或 cuda)')
    return parser.parse_args()

def show_mask_on_image(image, masks, alpha=0.5):
    """
    在图像上叠加掩码
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    
    # 为每个掩码使用不同的颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 青色
    ]
    
    if masks is not None and len(masks) > 0:
        for idx, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # 确保mask是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 调整mask大小以匹配图像
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            color = colors[idx % len(colors)]
            mask_bool = mask > 0.5
            
            # 在掩码区域应用颜色
            for c in range(3):
                overlay[:, :, c] = np.where(mask_bool, 
                                           color[c], 
                                           overlay[:, :, c])
    
    # 混合原图和掩码
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return result

def load_model(checkpoint_path, base_model_path, device='cpu'):
    """
    加载模型和checkpoint
    """
    print(f"正在加载基础模型: {base_model_path}")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 根据设备选择dtype
    if device == 'cpu':
        torch_dtype = torch.float32  # CPU使用float32
        print("⚠️  使用CPU推理,速度会比较慢,请耐心等待...")
    else:
        torch_dtype = torch.bfloat16  # GPU使用bfloat16
    
    # 加载模型
    model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    
    # 加载训练好的权重
    print(f"正在加载checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 处理可能的state_dict格式
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 移除可能的'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    print("权重加载完成!")
    
    # 移动到指定设备
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("模型已移至GPU")
    else:
        print("模型在CPU上运行")
    
    return model, tokenizer

def inference_image(model, tokenizer, image_path, prompt="请分割图像中的血管。"):
    """
    对单张图片进行推理
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 准备输入
    text = f"<image>{prompt}"
    
    input_dict = {
        'image': image,
        'text': text,
        'past_text': '',
        'mask_prompts': None,
        'tokenizer': tokenizer,
    }
    
    # 推理
    with torch.no_grad():
        try:
            return_dict = model.predict_forward(**input_dict)
            return return_dict
        except Exception as e:
            print(f"推理出错: {e}")
            return None

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_model(args.checkpoint, args.base_model, args.device)
    
    # 获取图片列表
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))[:args.num_images]
    
    print(f"\n找到 {len(image_files)} 张图片进行推理")
    print(f"输出目录: {output_dir}\n")
    
    results = []
    
    for idx, image_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] 正在处理: {image_path.name}")
        
        # 推理
        result = inference_image(model, tokenizer, str(image_path))
        
        if result is None:
            print(f"  ⚠️  推理失败")
            continue
        
        # 获取预测结果
        prediction_text = result.get('prediction', '').strip()
        prediction_masks = result.get('prediction_masks', [])
        
        print(f"  📝 预测文本: {prediction_text}")
        print(f"  🎭 掩码数量: {len(prediction_masks) if prediction_masks else 0}")
        
        # 保存结果
        original_image = Image.open(image_path).convert('RGB')
        
        # 生成可视化结果
        if prediction_masks and len(prediction_masks) > 0:
            vis_image = show_mask_on_image(original_image, prediction_masks)
            vis_image = Image.fromarray(vis_image)
        else:
            vis_image = original_image
        
        # 保存图片
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        vis_image.save(output_path)
        print(f"  ✅ 已保存: {output_path}\n")
        
        # 记录结果
        results.append({
            'image': image_path.name,
            'prediction': prediction_text,
            'num_masks': len(prediction_masks) if prediction_masks else 0,
            'output': str(output_path)
        })
    
    # 保存JSON结果
    json_path = output_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 推理完成! 共处理 {len(results)} 张图片")
    print(f"📊 结果已保存至: {json_path}")

if __name__ == '__main__':
    main()
