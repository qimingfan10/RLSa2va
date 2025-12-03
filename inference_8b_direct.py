#!/usr/bin/env python3
"""
直接使用8B模型内部组件进行推理 - 绕过文本生成
核心：直接用固定的[SEG] embedding + SAM2 decoder生成mask
"""
import os
import sys
sys.path.insert(0, '/home/ubuntu/Sa2VA')

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def dice_score(pred, target):
    pred = pred.flatten().astype(float)
    target = target.flatten().astype(float)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

class Direct8BInference:
    def __init__(self, model_path):
        print("Loading 8B model...")
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 获取[SEG] token id和embedding
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids('[SEG]')
        print(f"[SEG] token id: {self.seg_token_id}")
        
    def predict_mask(self, image_path):
        """直接生成mask，使用强制[SEG]方法"""
        image = Image.open(image_path).convert('RGB')
        ori_size = image.size  # (w, h)
        
        # 使用包含[SEG]的固定prompt进行推理
        # 关键：在assistant回复中明确要求输出[SEG]
        prompt = "<image>\nSegment the blood vessel and output [SEG] token."
        
        with torch.no_grad():
            result = self.model.predict_forward(
                image=image,
                text=prompt,
                tokenizer=self.tokenizer,
            )
        
        masks = result.get('prediction_masks', [])
        pred_text = result.get('prediction', '')
        
        print(f"  Output: {pred_text[:100]}")
        
        if masks and len(masks) > 0:
            mask = masks[0]
            if len(mask.shape) == 3:
                mask = mask[0]
            return mask, ori_size
        
        return None, ori_size

def main():
    import json
    from tqdm import tqdm
    
    model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_vessel_lora_finetune_8b_extreme/iter_15280_hf"
    data_dir = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
    
    inferencer = Direct8BInference(model_path)
    
    # 加载annotations
    with open(os.path.join(data_dir, "annotations.json")) as f:
        annotations = json.load(f)
    
    # 测试多个不同的prompt
    test_prompts = [
        "<image>\nPlease segment the blood vessel.",
        "<image>\nSegment blood vessel. Output: [SEG]",
        "<image>\nIdentify and segment the blood vessel region. Respond with [SEG].",
        "<image>\n请分割血管区域。",
    ]
    
    test_img = os.path.join(data_dir, "images", annotations[0]['image'])
    print(f"\n测试图片: {test_img}")
    
    for prompt in test_prompts:
        print(f"\n--- Prompt: {prompt[:50]}... ---")
        with torch.no_grad():
            result = inferencer.model.predict_forward(
                image=Image.open(test_img).convert('RGB'),
                text=prompt,
                tokenizer=inferencer.tokenizer,
            )
        pred = result.get('prediction', 'N/A')
        masks = result.get('prediction_masks', [])
        print(f"Output: {pred}")
        print(f"Has mask: {len(masks) > 0}")
        if '[SEG]' in str(pred):
            print(">>> SUCCESS: Found [SEG]! <<<")
            break

if __name__ == "__main__":
    main()
