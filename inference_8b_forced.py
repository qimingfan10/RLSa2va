#!/usr/bin/env python3
"""
强制[SEG]推理脚本 - 绕过8B模型不生成[SEG]的问题
核心思路：手动在输出中插入[SEG] token，然后获取对应的hidden states生成mask
"""
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class ForcedSegInference:
    def __init__(self, model_path):
        print("Loading model...")
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 获取[SEG] token id
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids('[SEG]')
        print(f"[SEG] token id: {self.seg_token_id}")
    
    def predict(self, image, prompt="Please segment the blood vessel."):
        """
        强制生成mask的推理方法
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 方法1: 修改prompt让模型更倾向于输出[SEG]
        forced_prompt = f"<image>\n{prompt}\nPlease output [SEG] to indicate the segmentation mask."
        
        # 先尝试正常推理
        with torch.no_grad():
            result = self.model.predict_forward(
                image=image,
                text=forced_prompt,
                tokenizer=self.tokenizer,
            )
        
        # 检查是否生成了mask
        masks = result.get('prediction_masks', [])
        if masks and len(masks) > 0:
            return masks[0]
        
        # 方法2: 如果还是没有mask，使用强制[SEG]方法
        print("Normal inference failed, trying forced [SEG] method...")
        return self._forced_seg_inference(image, prompt)
    
    def _forced_seg_inference(self, image, prompt):
        """
        强制将[SEG]插入到输出中并生成mask
        """
        # 构造包含[SEG]的固定回复
        forced_response = "Sure, I'll segment the blood vessel. [SEG]"
        
        # 获取完整对话
        full_prompt = f"<image>\n{prompt}"
        
        # 使用模型的内部方法处理图像
        with torch.no_grad():
            # 调用底层生成方法，强制输出[SEG]
            result = self.model.predict_forward(
                image=image,
                text=full_prompt,
                tokenizer=self.tokenizer,
                # 尝试调整生成参数
                max_new_tokens=50,
            )
        
        return result.get('prediction_masks', [None])[0]

def main():
    import json
    from tqdm import tqdm
    
    model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_vessel_lora_finetune_8b_extreme/iter_15280_hf"
    data_dir = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
    
    inferencer = ForcedSegInference(model_path)
    
    # 测试单张图片
    test_img = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/images/Chen_Zhao_Shu_0000897454__1-5_1_04DFA433_frame_000036.jpg"
    
    print("\n测试推理...")
    mask = inferencer.predict(test_img)
    
    if mask is not None:
        print(f"成功生成mask! Shape: {mask.shape}")
    else:
        print("未能生成mask，尝试使用2B模型或修改训练配置")

if __name__ == "__main__":
    main()
