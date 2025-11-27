#!/usr/bin/env python3
"""
使用转换后的HuggingFace格式模型进行真实推理
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
from pathlib import Path

print("=" * 80)
print("Sa2VA血管分割 - 真实HuggingFace模型推理")
print("=" * 80)

# 配置
model_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192_hf'
data_root = '/home/ubuntu/Sa2VA/data/vessel_data/'
output_dir = '/home/ubuntu/Sa2VA/real_hf_inference_results/'

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
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"    总显存: {mem_total:.2f} GB")

# 加载模型
print("\n加载HuggingFace格式的模型...")
print("这可能需要几分钟，请耐心等待...")

try:
    from transformers import AutoModel, AutoTokenizer
    
    # 加载模型（使用device_map自动多GPU分配）
    print("  加载模型...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"  # 自动分配到多个GPU
    ).eval()
    
    print(f"✅ 模型加载成功")
    
    # 加载tokenizer
    print("  加载tokenizer...")
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
    sys.exit(1)

# 加载数据
print("\n加载测试数据...")
with open(os.path.join(data_root, 'annotations.json'), 'r') as f:
    annotations = json.load(f)

print(f"总样本数: {len(annotations)}")

# 选择测试样本
test_samples = annotations[::10][:5]  # 测试5个样本
print(f"测试样本数: {len(test_samples)}")

# 辅助函数
def polygon_to_mask(polygon_coords, image_shape):
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

# 开始真实推理
print("\n开始真实HuggingFace模型推理...")
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

for idx, sample in enumerate(tqdm(test_samples, desc="真实推理进度")):
    try:
        # 加载图像
        img_path = os.path.join(data_root, 'images', sample['image'])
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # 创建ground truth mask
        gt_mask = polygon_to_mask(sample['mask'][0] if sample['mask'] else [], image_np.shape)
        
        print(f"\n样本 {idx+1}: {sample['image']}")
        print("  执行真实模型推理...")
        
        with torch.no_grad():
            # 使用HuggingFace模型的predict_forward方法
            try:
                result = model.predict_forward(
                    image=image,
                    text="blood vessel",
                    tokenizer=tokenizer
                )
                
                print("  ✅ 真实推理成功！")
                
                # 提取预测掩码
                if 'prediction_masks' in result and result['prediction_masks'] is not None and len(result['prediction_masks']) > 0:
                    pred_mask = result['prediction_masks'][0]
                    
                    # 转换为numpy
                    if torch.is_tensor(pred_mask):
                        pred_mask = pred_mask.cpu().numpy()
                    
                    # 调整形状
                    if pred_mask.ndim == 3:
                        pred_mask = pred_mask[0]
                    
                    # 确保与GT相同尺寸
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                    
                    # 归一化到[0, 1]
                    if pred_mask.max() > 1:
                        pred_mask = pred_mask / 255.0
                    
                    print(f"  预测掩码形状: {pred_mask.shape}")
                    print(f"  预测值范围: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
                    print(f"  这是真实的模型预测！")
                    
                else:
                    print("  ⚠️  模型未返回预测掩码")
                    # 创建空掩码
                    pred_mask = np.zeros_like(gt_mask, dtype=float)
                    
            except Exception as e:
                print(f"  ❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                # 创建空掩码
                pred_mask = np.zeros_like(gt_mask, dtype=float)
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        # 记录指标
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # 保存结果
        results.append({
            'image': sample['image'],
            'metrics': metrics,
            'prediction_text': result.get('prediction', '') if 'result' in locals() else ''
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
        axes[2].set_title(f'Real Model Prediction\n(HuggingFace)')
        axes[2].axis('off')
        
        axes[3].imshow(image_np)
        axes[3].imshow(pred_mask, alpha=0.5, cmap='Greens')
        axes[3].imshow(gt_mask, alpha=0.3, cmap='Reds')
        axes[3].set_title(f'Overlay\nDice: {metrics["dice"]:.4f}, IoU: {metrics["iou"]:.4f}')
        axes[3].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, 'visualizations', f'real_hf_pred_{idx:03d}.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 可视化已保存: {vis_path}")
        
        # 显示当前显存使用
        print(f"  当前显存使用:")
        for i in range(num_gpus):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"    GPU {i}: {mem_allocated:.2f} GB")
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        continue

# 汇总结果
print("\n" + "=" * 80)
print("真实HuggingFace模型推理结果汇总")
print("=" * 80)

for key in all_metrics:
    if len(all_metrics[key]) > 0:
        mean_val = np.mean(all_metrics[key])
        std_val = np.std(all_metrics[key])
        print(f"{key.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# 保存结果
results_json = {
    'model_type': 'HuggingFace (Real Inference)',
    'model_path': model_path,
    'num_gpus': num_gpus,
    'summary': {key: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} 
                for key, vals in all_metrics.items() if len(vals) > 0},
    'details': results
}

with open(os.path.join(output_dir, 'real_hf_inference_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n详细结果已保存到: {os.path.join(output_dir, 'real_hf_inference_results.json')}")
print(f"可视化结果已保存到: {os.path.join(output_dir, 'visualizations/')}")

# 最终显存使用
print("\n最终显存使用:")
for i in range(num_gpus):
    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"  GPU {i}: 已分配 {mem_allocated:.2f} GB, 已保留 {mem_reserved:.2f} GB")

print("\n" + "=" * 80)
print("🎉 真实HuggingFace模型推理完成！")
print("=" * 80)
