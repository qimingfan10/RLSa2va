"""
快速阈值测试 - 直接修改模型predict_forward返回概率图
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')

from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_model_for_probs(model):
    """直接修改模型predict_forward方法，让它返回概率图"""
    original_predict = model.predict_forward
    
    # 创建一个wrapper保存概率图
    model._last_prob_maps = []
    
    def predict_with_prob_capture(*args, **kwargs):
        # 清空之前的概率图
        model._last_prob_maps = []
        
        # 调用原始方法
        result = original_predict(*args, **kwargs)
        
        # 返回结果（概率图已经保存在model._last_prob_maps中）
        return result
    
    # 替换方法
    model.predict_forward = predict_with_prob_capture
    
    # 同时需要hook模型内部的mask生成部分
    # 找到text_hidden_fcs的forward，在那里拦截
    return model


def calculate_metrics(pred_mask, gt_mask):
    """计算指标"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    
    return {'dice': dice, 'recall': recall, 'precision': precision}


def main():
    print("="*80)
    print("快速阈值测试")
    print("="*80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # 加载模型
    print("\n加载模型...")
    model_path = '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    print("✅ 模型加载完成")
    
    # 加载Baseline测试的10张图像
    print("\n使用Baseline的10张图像进行测试...")
    baseline_images = [
        "Chen_Xiao_Hui_0000820236__1-3_1_02F7DFB7_frame_000010.jpg",
        "Lu_Jia_Xi_0000800516__1-3_1_03DC5E36_frame_000001.jpg", 
        "Lv_Shao_You_0000788036__1-3_1_052F13B8_frame_000041.jpg",
        "Tang_Hong_0000856316__1-4_1_04AA8BF3_frame_000025.jpg",
        "Wang_Gui_Lian_0000779356__1-3_1_001E8C68_frame_000001.jpg",
        "Wang_Gui_Lian_0000779356__1-3_1_001E8C68_frame_000017.jpg",
        "Wu_Yin_E_0000769116__1-4_1_02CE5B73_frame_000001.jpg",
        "Xu_Xiu_Lan_0000772236__1-3_1_01E06B60_frame_000013.jpg",
        "Zhao_Chun_Hua_0000821376__1-3_1_02E37CFD_frame_000017.jpg",
        "Zhao_Chun_Hua_0000821376__1-3_1_02E37CFD_frame_000033.jpg"
    ]
    
    data_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data'
    
    # 收集概率图
    print("\n步骤1: 推理获取概率图...")
    prob_data = []
    
    for img_name in tqdm(baseline_images):
        image_path = os.path.join(data_root, 'images', img_name)
        if not os.path.exists(image_path):
            print(f"⚠️ 图像不存在: {img_name}")
            continue
            
        # 找对应的mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(data_root, 'masks', mask_name)
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask不存在: {mask_name}")
            continue
        
        # 加载图像和mask
        image = Image.open(image_path).convert('RGB')
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # 推理
        try:
            result = model.predict_forward(
                image=image,
                text='<image>\nPlease segment the blood vessel.',
                tokenizer=tokenizer
            )
            
            if len(result['prediction_masks']) > 0:
                # 使用默认0.5阈值的mask来反推概率图
                # 但这不准确...我们需要修改源码
                binary_mask = result['prediction_masks'][0][0]
                
                # 临时方案：假设接近边界的地方是0.5
                # 实际上我们需要修改源码才能获取真实概率图
                print(f"  {img_name}: 推理成功（但无法获取概率图）")
                
        except Exception as e:
            print(f"⚠️ {img_name} 失败: {e}")
            continue
    
    print("\n❌ 当前方法无法获取概率图！")
    print("\n需要直接修改 modeling_sa2va_chat.py 源码:")
    print("  1. 找到第768行: masks = masks.sigmoid() > 0.5")
    print("  2. 在这行之前添加: prob_maps = masks.sigmoid().cpu().numpy()")
    print("  3. 返回: return {'prediction': predict, 'prediction_masks': ret_masks, 'probability_maps': prob_maps}")
    
    print("\n或者使用更简单的方法：")
    print("  直接在推理时手动测试不同阈值（修改源码中的0.5为其他值）")


if __name__ == '__main__':
    main()
