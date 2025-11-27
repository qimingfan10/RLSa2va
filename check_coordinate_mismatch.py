#!/usr/bin/env python3
"""
检查JSON坐标和实际图像尺寸的不匹配
"""

import json
import os
from PIL import Image

json_dir = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/json/'
images_dir = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512/images/'

# 检查所有样本
json_files = sorted(os.listdir(json_dir))

mismatches = []
size_stats = {}

for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)
    img_file = json_file.replace('.json', '.jpg')
    img_path = os.path.join(images_dir, img_file)
    
    if not os.path.exists(img_path):
        continue
    
    with open(json_path) as f:
        data = json.load(f)
    
    img = Image.open(img_path)
    
    json_size = (data.get('imageWidth'), data.get('imageHeight'))
    actual_size = img.size
    
    # 统计
    key = f"JSON:{json_size} -> 实际:{actual_size}"
    size_stats[key] = size_stats.get(key, 0) + 1
    
    if json_size != actual_size:
        mismatches.append({
            'file': img_file,
            'json_size': json_size,
            'actual_size': actual_size,
            'scale_x': actual_size[0] / json_size[0],
            'scale_y': actual_size[1] / json_size[1]
        })

print("=" * 80)
print("坐标尺寸不匹配检查")
print("=" * 80)
print(f"\n总样本数: {len(json_files)}")
print(f"不匹配数: {len(mismatches)}")
print(f"匹配率: {(len(json_files) - len(mismatches)) / len(json_files) * 100:.2f}%")

print("\n尺寸统计:")
for key, count in sorted(size_stats.items()):
    print(f"  {key}: {count}个")

if mismatches:
    print("\n前5个不匹配样本:")
    for item in mismatches[:5]:
        print(f"\n  文件: {item['file']}")
        print(f"  JSON尺寸: {item['json_size']}")
        print(f"  实际尺寸: {item['actual_size']}")
        print(f"  缩放比例: X={item['scale_x']:.4f}, Y={item['scale_y']:.4f}")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
if len(mismatches) > 0:
    print("⚠️  发现坐标不匹配！")
    print("   JSON中的坐标是基于原始尺寸的")
    print("   但图像已经被resize到512×512")
    print("   需要在prepare_merged_dataset.py中缩放坐标！")
else:
    print("✅ 所有坐标都匹配")
