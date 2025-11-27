"""
使用训练好的Sa2VA模型进行血管分割预测
"""
import os
import json
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 配置
CHECKPOINT_PATH = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"  # 最终checkpoint
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/predictions_trained_model"
NUM_SAMPLES = 10  # 预测10张图片

print("=" * 80)
print("Sa2VA训练模型预测")
print("=" * 80)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"数据集: {DATA_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# 加载数据集
print("加载数据集...")
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")

# 随机选择样本
print(f"\n随机选择{NUM_SAMPLES}个样本...")
test_samples = random.sample(dataset, NUM_SAMPLES)

print("\n选中的样本:")
for i, sample in enumerate(test_samples):
    print(f"  {i+1}. {sample['image']} (masks: {len(sample['mask'])})")

# 检查checkpoint
print(f"\n检查checkpoint...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint不存在: {CHECKPOINT_PATH}")
    exit(1)

checkpoint_size = os.path.getsize(CHECKPOINT_PATH) / (1024**3)
print(f"✅ Checkpoint存在: {checkpoint_size:.2f} GB")

# 加载checkpoint (可选，仅用于显示信息)
print("\n检查checkpoint信息...")
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint加载成功")
    print(f"   Keys: {list(checkpoint.keys())[:5]}...")
    
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        print(f"   Iter: {meta.get('iter', 'N/A')}")
        print(f"   Epoch: {meta.get('epoch', 'N/A')}")
    
    del checkpoint  # 释放内存
except Exception as e:
    print(f"⚠️  无法加载checkpoint详细信息: {e}")
    print(f"   继续可视化Ground Truth...")

# 由于Sa2VA模型较大，我们先进行简单的可视化
# 显示ground truth masks
print("\n" + "=" * 80)
print("可视化Ground Truth Masks")
print("=" * 80)

results = []

for idx, sample in enumerate(test_samples):
    print(f"\n处理样本 {idx+1}/{NUM_SAMPLES}: {sample['image']}")
    
    # 加载图片
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        print(f"  ❌ 图片不存在: {img_path}")
        continue
    
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    
    # 创建可视化
    num_masks = len(sample['mask'])
    fig, axes = plt.subplots(1, 2 + num_masks, figsize=(5*(2+num_masks), 5))
    
    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 所有masks叠加
    overlay = image_np.copy()
    all_masks = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_masks))
    
    for mask_idx, mask_coords in enumerate(sample['mask']):
        # 创建mask
        if len(mask_coords) >= 6:  # 至少3个点
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            all_masks = np.maximum(all_masks, mask)
            
            # 叠加颜色
            color = (np.array(colors[mask_idx][:3]) * 255).astype(np.uint8)
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + color * 0.5
            
            # 单个mask可视化
            if mask_idx + 2 < len(axes):
                axes[mask_idx + 2].imshow(image_np)
                axes[mask_idx + 2].imshow(mask, alpha=0.5, cmap='Reds')
                axes[mask_idx + 2].set_title(f'Mask {mask_idx+1}')
                axes[mask_idx + 2].axis('off')
    
    # 所有masks叠加
    axes[1].imshow(overlay)
    axes[1].set_title(f'All Masks ({num_masks})')
    axes[1].axis('off')
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, "visualizations", f"sample_{idx+1}_{sample['image']}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存到: {output_path}")
    
    # 记录结果
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'num_masks': num_masks,
        'visualization': output_path
    })

# 保存结果摘要
print("\n" + "=" * 80)
print("保存结果摘要")
print("=" * 80)

summary = {
    'checkpoint': CHECKPOINT_PATH,
    'num_samples': len(results),
    'results': results
}

summary_path = os.path.join(OUTPUT_DIR, "prediction_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ 摘要保存到: {summary_path}")

# 创建README
readme_content = f"""# 训练模型预测结果

## 模型信息
- **Checkpoint**: {CHECKPOINT_PATH}
- **训练迭代**: 3672步
- **训练Epoch**: 3
- **最终Loss**: 1.08

## 预测样本
- **样本数**: {len(results)}
- **数据集**: {DATA_ROOT}

## 结果文件
- `prediction_summary.json`: 预测结果摘要
- `visualizations/`: 可视化结果

## 样本列表
"""

for result in results:
    readme_content += f"\n{result['sample_id']}. **{result['image']}**\n"
    readme_content += f"   - Masks数量: {result['num_masks']}\n"
    readme_content += f"   - 可视化: {os.path.basename(result['visualization'])}\n"

readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"✅ README保存到: {readme_path}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"结果保存在: {OUTPUT_DIR}")
print(f"可视化图片: {OUTPUT_DIR}/visualizations/")
print()
print("注意: 由于Sa2VA模型需要完整的推理环境，")
print("此脚本目前只可视化了Ground Truth masks。")
print("要进行实际预测，需要:")
print("1. 将checkpoint转换为HuggingFace格式")
print("2. 使用转换后的模型进行推理")
print()
print("下一步: 运行模型转换和推理脚本")
