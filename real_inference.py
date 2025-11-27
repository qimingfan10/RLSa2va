#!/usr/bin/env python3
"""
Sa2VA血管分割模型真实推理脚本
使用训练好的权重进行真实的模型预测
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')

print("=" * 80)
print("Sa2VA血管分割模型真实推理")
print("=" * 80)

# 配置
config_path = '/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_vessel_finetune.py'
checkpoint_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192.pth'
data_root = '/home/ubuntu/Sa2VA/data/vessel_data/'
output_dir = '/home/ubuntu/Sa2VA/real_inference_results/'

print(f"\n配置文件: {config_path}")
print(f"权重文件: {checkpoint_path}")
print(f"数据路径: {data_root}")
print(f"输出目录: {output_dir}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# 检查文件是否存在
print("\n检查文件...")
if not os.path.exists(checkpoint_path):
    print(f"❌ 权重文件不存在: {checkpoint_path}")
    sys.exit(1)
print(f"✅ 权重文件存在")

if not os.path.exists(config_path):
    print(f"❌ 配置文件不存在: {config_path}")
    sys.exit(1)
print(f"✅ 配置文件存在")

# 加载配置
print("\n加载配置...")
try:
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)
    print(f"✅ 配置加载成功")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    sys.exit(1)

# 加载模型
print("\n加载模型...")
print("这可能需要几分钟，请耐心等待...")

try:
    # 导入模型类
    from projects.sa2va.models import Sa2VAModel
    from mmengine.registry import MODELS
    
    # 创建模型
    print("创建模型实例...")
    # 使用mmengine的build方式
    model = MODELS.build(cfg.model)
    
    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查checkpoint结构
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 加载state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  缺失的keys: {len(missing_keys)}个")
        if len(missing_keys) < 10:
            for key in missing_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"⚠️  多余的keys: {len(unexpected_keys)}个")
        if len(unexpected_keys) < 10:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    # 设置为评估模式
    model.eval()
    
    # 由于模型太大（9.2B参数），使用CPU推理
    device = 'cpu'
    print(f"使用设备: {device} (模型太大，GPU显存不足)")
    print(f"注意：CPU推理会比较慢，请耐心等待...")
    
    print(f"✅ 模型加载成功")
    
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

# 选择测试样本（CPU推理很慢，只测试1个样本）
test_samples = annotations[::10][:1]  # 只测试1个样本
print(f"测试样本数: {len(test_samples)}")

# 辅助函数
def polygon_to_mask(polygon_coords, image_shape):
    """将多边形坐标转换为掩码"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(polygon_coords) == 0:
        return mask
    
    # 转换为(x,y)点对
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

# 开始推理
print("\n开始真实推理...")
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

for idx, sample in enumerate(tqdm(test_samples, desc="推理进度")):
    try:
        # 加载图像
        img_path = os.path.join(data_root, 'images', sample['image'])
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # 创建ground truth mask
        gt_mask = polygon_to_mask(sample['mask'][0] if sample['mask'] else [], image_np.shape)
        
        # 准备模型输入
        # 注意：这里需要根据Sa2VA的实际输入格式进行调整
        print(f"\n样本 {idx+1}: {sample['image']}")
        print("  准备输入数据...")
        
        # TODO: 根据Sa2VA的实际接口调整输入格式
        # 这里需要查看Sa2VAModel的forward方法签名
        
        with torch.no_grad():
            # 转换图像为tensor
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
            # CPU推理，不需要移动到GPU
            
            print("  执行模型推理...")
            
            # 调用模型
            # 注意：这里需要根据实际的模型接口调整
            try:
                # 尝试直接调用
                output = model(image_tensor, text_prompt="blood vessel")
            except Exception as e:
                print(f"  ⚠️  直接调用失败: {e}")
                print("  尝试其他调用方式...")
                
                # 尝试使用predict方法
                if hasattr(model, 'predict'):
                    output = model.predict(image_tensor, "blood vessel")
                elif hasattr(model, 'inference'):
                    output = model.inference(image_tensor, "blood vessel")
                else:
                    raise Exception("无法找到合适的推理方法")
            
            print("  ✅ 推理完成")
            
            # 提取预测掩码
            if isinstance(output, dict):
                if 'mask' in output:
                    pred_mask = output['mask']
                elif 'pred_mask' in output:
                    pred_mask = output['pred_mask']
                else:
                    print(f"  输出keys: {output.keys()}")
                    raise Exception("无法找到mask输出")
            else:
                pred_mask = output
            
            # 转换为numpy
            if torch.is_tensor(pred_mask):
                pred_mask = pred_mask.cpu().numpy()
            
            # 调整形状
            if pred_mask.ndim == 4:
                pred_mask = pred_mask[0, 0]  # [B, C, H, W] -> [H, W]
            elif pred_mask.ndim == 3:
                pred_mask = pred_mask[0]  # [B, H, W] -> [H, W]
            
            # 确保与GT相同尺寸
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            
            print(f"  预测掩码形状: {pred_mask.shape}")
            print(f"  预测值范围: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        # 记录指标
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # 保存结果
        results.append({
            'image': sample['image'],
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
        axes[2].set_title(f'Prediction\n(Real Model Output)')
        axes[2].axis('off')
        
        axes[3].imshow(image_np)
        axes[3].imshow(pred_mask, alpha=0.5, cmap='Greens')
        axes[3].imshow(gt_mask, alpha=0.3, cmap='Reds')
        axes[3].set_title(f'Overlay\nDice: {metrics["dice"]:.4f}, IoU: {metrics["iou"]:.4f}')
        axes[3].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, 'visualizations', f'real_pred_{idx:03d}.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 可视化已保存: {vis_path}")
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        continue

# 汇总结果
print("\n" + "=" * 80)
print("真实推理结果汇总")
print("=" * 80)

for key in all_metrics:
    if len(all_metrics[key]) > 0:
        mean_val = np.mean(all_metrics[key])
        std_val = np.std(all_metrics[key])
        print(f"{key.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# 保存结果
results_json = {
    'summary': {key: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} 
                for key, vals in all_metrics.items() if len(vals) > 0},
    'details': results
}

with open(os.path.join(output_dir, 'real_inference_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n详细结果已保存到: {os.path.join(output_dir, 'real_inference_results.json')}")
print(f"可视化结果已保存到: {os.path.join(output_dir, 'visualizations/')}")

print("\n" + "=" * 80)
print("真实推理完成！")
print("=" * 80)
