#!/usr/bin/env python3
"""
使用Segment_DATA_Merged_512数据集进行预测
随机选择10张图片
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
from pathlib import Path

print("=" * 80)
print("Sa2VA血管分割 - Segment_DATA_Merged_512数据集预测")
print("=" * 80)

# 配置
model_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf'
data_root = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/'
output_dir = '/home/ubuntu/Sa2VA/merged_dataset_predictions/'

print(f"\n模型路径: {model_path}")
print(f"数据路径: {data_root}")
print(f"输出目录: {output_dir}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# 检查GPU
print("\n检查GPU...")
if not torch.cuda.is_available():
    print("❌ CUDA不可用")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"✅ 检测到 {num_gpus} 个GPU")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 加载模型
print("\n加载模型...")
print("这可能需要几分钟，请耐心等待...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # 使用单GPU避免设备不匹配问题
        trust_remote_code=True
    ).eval()
    
    print(f"✅ 模型加载成功")
    
    print("  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer加载成功")
    
    # 显示显存使用
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"\n当前显存使用: {mem_allocated:.2f} GB")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 获取所有图片
print("\n加载图片列表...")
images_dir = os.path.join(data_root, 'images')
masks_dir = os.path.join(data_root, 'masks')

all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
print(f"总图片数: {len(all_images)}")

# 随机选择10张图片
random.seed(42)  # 设置随机种子以便复现
selected_images = random.sample(all_images, min(10, len(all_images)))
print(f"选择图片数: {len(selected_images)}")

# 辅助函数
def load_mask_from_file(mask_path, image_shape):
    """从mask文件加载掩码"""
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # 确保尺寸匹配
            if mask.shape != image_shape[:2]:
                mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
            # 二值化
            mask = (mask > 127).astype(np.uint8)
            return mask
    # 如果没有mask文件，返回空掩码
    return np.zeros(image_shape[:2], dtype=np.uint8)

def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """计算分割评价指标"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)
    
    TP = np.sum((pred_binary == 1) & (gt_binary == 1))
    FP = np.sum((pred_binary == 1) & (gt_binary == 0))
    FN = np.sum((pred_binary == 0) & (gt_binary == 1))
    TN = np.sum((pred_binary == 0) & (gt_binary == 0))
    
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy
    }

# 开始预测
print("\n开始预测...")
print("-" * 80)

all_metrics = {
    'dice': [],
    'iou': [],
    'precision': [],
    'recall': [],
    'specificity': [],
    'accuracy': []
}

results = []

for idx, img_name in enumerate(tqdm(selected_images, desc="预测进度")):
    try:
        # 加载图像
        img_path = os.path.join(images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # 加载ground truth mask
        mask_name = img_name.replace('.jpg', '.png').replace('.png', '.png')
        mask_path = os.path.join(masks_dir, mask_name)
        gt_mask = load_mask_from_file(mask_path, image_np.shape)
        
        print(f"\n样本 {idx+1}: {img_name}")
        print(f"  图像尺寸: {image_np.shape}")
        print(f"  GT掩码: {'存在' if os.path.exists(mask_path) else '不存在'}")
        
        with torch.no_grad():
            try:
                # 使用官方demo的调用方式
                text = "<image>Please segment the blood vessel in this image. [SEG]"
                
                result = model.predict_forward(
                    image=image,
                    text=text,
                    tokenizer=tokenizer,
                    processor=None
                )
                
                print("  ✅ 预测成功！")
                
                # 提取预测掩码
                if 'prediction_masks' in result and result['prediction_masks'] is not None and len(result['prediction_masks']) > 0:
                    pred_masks_list = result['prediction_masks'][0]
                    
                    if len(pred_masks_list) > 0:
                        pred_mask = pred_masks_list[0]
                        
                        if torch.is_tensor(pred_mask):
                            pred_mask = pred_mask.cpu().numpy()
                        
                        if pred_mask.shape != gt_mask.shape:
                            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                        
                        if pred_mask.max() > 1:
                            pred_mask = pred_mask / 255.0
                        
                        print(f"  预测掩码形状: {pred_mask.shape}")
                        print(f"  预测值范围: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
                    else:
                        print("  ⚠️  预测掩码列表为空")
                        pred_mask = np.zeros_like(gt_mask, dtype=float)
                else:
                    print("  ⚠️  模型未返回预测掩码")
                    pred_mask = np.zeros_like(gt_mask, dtype=float)
                    
            except Exception as e:
                print(f"  ❌ 预测失败: {e}")
                pred_mask = np.zeros_like(gt_mask, dtype=float)
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        # 记录指标
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # 保存结果
        results.append({
            'image': img_name,
            'metrics': metrics
        })
        
        # 可视化
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title(f'Model Prediction')
        axes[2].axis('off')
        
        axes[3].imshow(image_np)
        axes[3].imshow(pred_mask, alpha=0.5, cmap='Greens')
        axes[3].imshow(gt_mask, alpha=0.3, cmap='Reds')
        axes[3].set_title(f'Overlay\nDice: {metrics["dice"]:.4f}, IoU: {metrics["iou"]:.4f}')
        axes[3].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, 'visualizations', f'pred_{idx:02d}_{img_name}')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 可视化已保存")
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        continue

# 汇总结果
print("\n" + "=" * 80)
print("预测结果汇总")
print("=" * 80)

for key in all_metrics:
    if len(all_metrics[key]) > 0:
        mean_val = np.mean(all_metrics[key])
        std_val = np.std(all_metrics[key])
        print(f"{key.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# 保存结果
results_json = {
    'dataset': 'Segment_DATA_Merged_512',
    'model_path': model_path,
    'num_samples': len(selected_images),
    'summary': {key: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} 
                for key, vals in all_metrics.items() if len(vals) > 0},
    'details': results
}

with open(os.path.join(output_dir, 'prediction_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n详细结果已保存到: {os.path.join(output_dir, 'prediction_results.json')}")
print(f"可视化结果已保存到: {os.path.join(output_dir, 'visualizations/')}")

print("\n" + "=" * 80)
print("🎉 预测完成！")
print("=" * 80)
