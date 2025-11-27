#!/usr/bin/env python3
"""
验证坐标缩放是否正确
"""

import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 读取annotations
with open('/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json') as f:
    annotations = json.load(f)

# 选择一个需要缩放的样本（800×800 -> 512×512）
sample = annotations[0]  # An Cong Xue...frame_000011.jpg

print("=" * 80)
print("验证坐标缩放")
print("=" * 80)
print(f"\n样本: {sample['image']}")
print(f"掩码数量: {len(sample['mask'])}")

# 加载图像
img_path = f"/home/ubuntu/Sa2VA/data/merged_vessel_data/images/{sample['image']}"
image = Image.open(img_path)
image_np = np.array(image)

print(f"图像尺寸: {image_np.shape}")

# 检查坐标范围
all_coords = []
for mask in sample['mask']:
    for i in range(0, len(mask), 2):
        x, y = mask[i], mask[i+1]
        all_coords.append((x, y))

x_coords = [c[0] for c in all_coords]
y_coords = [c[1] for c in all_coords]

print(f"\n坐标范围:")
print(f"  X: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
print(f"  Y: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
print(f"  图像宽度: {image_np.shape[1]}")
print(f"  图像高度: {image_np.shape[0]}")

# 检查是否有超出范围的坐标
out_of_bounds = []
for x, y in all_coords:
    if x < 0 or x > image_np.shape[1] or y < 0 or y > image_np.shape[0]:
        out_of_bounds.append((x, y))

if out_of_bounds:
    print(f"\n⚠️  发现 {len(out_of_bounds)} 个超出范围的坐标！")
    print(f"  前5个: {out_of_bounds[:5]}")
else:
    print(f"\n✅ 所有坐标都在图像范围内")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 原图
axes[0].imshow(image_np, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# 绘制多边形
axes[1].imshow(image_np, cmap='gray')
for mask in sample['mask']:
    points = []
    for i in range(0, len(mask), 2):
        points.append([mask[i], mask[i+1]])
    points = np.array(points, dtype=np.int32)
    axes[1].plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
    axes[1].plot(points[:, 0], points[:, 1], 'ro', markersize=3)

axes[1].set_title('Polygon Annotations (Scaled)')
axes[1].axis('off')

plt.tight_layout()
output_path = '/home/ubuntu/Sa2VA/coordinate_scaling_verification.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n可视化已保存: {output_path}")

print("\n" + "=" * 80)
if not out_of_bounds:
    print("✅ 坐标缩放正确！")
else:
    print("❌ 坐标缩放有问题！")
print("=" * 80)
