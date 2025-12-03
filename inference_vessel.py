#!/usr/bin/env python3
"""
血管分割推理脚本 - 使用训练好的2B模型
"""
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class VesselSegmentor:
    def __init__(self, model_path=None):
        if model_path is None:
            # 默认使用2B模型
            model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_vessel_lora_finetune/iter_7640_hf"
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded!")
    
    def segment(self, image, prompt="Please segment the blood vessel."):
        """
        分割图像中的血管
        
        Args:
            image: PIL.Image 或 图像路径
            prompt: 分割提示词
            
        Returns:
            mask: numpy array, 二值mask
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        ori_size = image.size  # (w, h)
        
        text = f"<image>\n{prompt}"
        
        with torch.no_grad():
            result = self.model.predict_forward(
                image=image,
                text=text,
                tokenizer=self.tokenizer,
            )
        
        masks = result.get('prediction_masks', [])
        
        if masks and len(masks) > 0:
            mask = masks[0]
            if len(mask.shape) == 3:
                mask = mask[0]  # 取第一帧
            
            # Resize到原始大小
            if mask.shape != (ori_size[1], ori_size[0]):
                mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    ori_size, Image.NEAREST)) > 127
            
            return mask.astype(np.uint8)
        
        return None
    
    def segment_and_save(self, image_path, output_path=None):
        """分割并保存结果"""
        mask = self.segment(image_path)
        
        if mask is not None:
            if output_path is None:
                base = os.path.splitext(image_path)[0]
                output_path = f"{base}_pred_mask.png"
            
            Image.fromarray(mask * 255).save(output_path)
            print(f"Saved mask to {output_path}")
            return output_path
        else:
            print("Failed to generate mask")
            return None

def dice_score(pred, target):
    pred = pred.flatten().astype(float)
    target = target.flatten().astype(float)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def main():
    import json
    from tqdm import tqdm
    
    # 初始化分割器
    segmentor = VesselSegmentor()
    
    # 数据路径
    data_dir = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
    
    # 加载annotations
    with open(os.path.join(data_dir, "annotations.json")) as f:
        annotations = json.load(f)
    
    # 评估
    num_samples = min(50, len(annotations))
    print(f"\n评估 {num_samples} 个样本...")
    
    dice_scores = []
    
    for ann in tqdm(annotations[:num_samples]):
        img_path = os.path.join(data_dir, "images", ann['image'])
        if not os.path.exists(img_path):
            continue
        
        # 获取GT mask
        base_name = os.path.splitext(ann['image'])[0]
        mask_path = os.path.join(data_dir, "masks", f"{base_name}_mask.png")
        if not os.path.exists(mask_path):
            continue
        
        gt_mask = np.array(Image.open(mask_path).convert('L')) > 127
        
        # 预测
        pred_mask = segmentor.segment(img_path)
        
        if pred_mask is not None:
            dice = dice_score(pred_mask, gt_mask)
            dice_scores.append(dice)
    
    # 输出结果
    print("\n" + "="*50)
    print("2B模型评估结果")
    print("="*50)
    print(f"样本数: {len(dice_scores)}")
    print(f"Average Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Min: {np.min(dice_scores):.4f}, Max: {np.max(dice_scores):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
