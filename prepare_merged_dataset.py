#!/usr/bin/env python3
"""
将Segment_DATA_Merged_512数据集转换为Sa2VA训练格式
"""

import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# 路径配置
source_dir = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512'
target_dir = '/home/ubuntu/Sa2VA/data/merged_vessel_data'

images_source = os.path.join(source_dir, 'images')
json_source = os.path.join(source_dir, 'json')

images_target = os.path.join(target_dir, 'images')
annotations_file = os.path.join(target_dir, 'annotations.json')

print("=" * 80)
print("准备Segment_DATA_Merged_512数据集用于Sa2VA训练")
print("=" * 80)

# 创建目标目录
os.makedirs(images_target, exist_ok=True)

# 获取所有图片文件
image_files = sorted([f for f in os.listdir(images_source) if f.endswith('.jpg')])
print(f"\n找到 {len(image_files)} 张图片")

# 准备annotations
annotations = []
skipped = 0
scaled_count = 0
no_scale_count = 0

for img_file in image_files:
    # 对应的JSON文件
    json_file = img_file.replace('.jpg', '.json')
    json_path = os.path.join(json_source, json_file)
    
    if not os.path.exists(json_path):
        print(f"警告: 找不到JSON文件: {json_file}")
        skipped += 1
        continue
    
    # 读取JSON标注
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        # 获取JSON中记录的尺寸和实际图像尺寸
        json_width = label_data.get('imageWidth', 512)
        json_height = label_data.get('imageHeight', 512)
        
        # 读取实际图像尺寸
        from PIL import Image
        img_path_check = os.path.join(images_source, img_file)
        with Image.open(img_path_check) as img:
            actual_width, actual_height = img.size
        
        # 计算缩放比例
        scale_x = actual_width / json_width
        scale_y = actual_height / json_height
        
        # 统计缩放情况
        if scale_x != 1.0 or scale_y != 1.0:
            scaled_count += 1
            if scaled_count <= 3:  # 只打印前3个
                print(f"  缩放 {img_file}: {json_width}×{json_height} -> {actual_width}×{actual_height} (scale: {scale_x:.4f})")
        else:
            no_scale_count += 1
        
        # 提取多边形坐标并缩放
        masks = []
        if 'shapes' in label_data:
            for shape in label_data['shapes']:
                if 'points' in shape:
                    # 将points转换为扁平列表并缩放坐标
                    points = shape['points']
                    flat_coords = []
                    for point in points:
                        # 缩放坐标到实际图像尺寸
                        scaled_x = float(point[0]) * scale_x
                        scaled_y = float(point[1]) * scale_y
                        flat_coords.extend([scaled_x, scaled_y])
                    masks.append(flat_coords)
        
        if len(masks) == 0:
            print(f"警告: {json_file} 没有标注")
            skipped += 1
            continue
        
        # 创建annotation条目
        # text必须是列表，每个mask对应一个text
        texts = ["blood vessel"] * len(masks)
        
        annotation = {
            "image": img_file,
            "mask": masks,
            "text": texts,  # 必须是列表！
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nPlease segment the blood vessel in this image."
                },
                {
                    "from": "gpt",
                    "value": "Sure, [SEG]."
                }
            ]
        }
        
        annotations.append(annotation)
        
    except Exception as e:
        print(f"错误处理 {json_file}: {e}")
        skipped += 1
        continue

print(f"\n成功处理: {len(annotations)} 个样本")
print(f"跳过: {skipped} 个样本")
print(f"需要缩放坐标: {scaled_count} 个样本")
print(f"无需缩放: {no_scale_count} 个样本")

# 复制图片到目标目录
print("\n复制图片...")
for annotation in annotations:
    img_file = annotation['image']
    src = os.path.join(images_source, img_file)
    dst = os.path.join(images_target, img_file)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

print(f"✅ 已复制 {len(annotations)} 张图片")

# 保存annotations.json
print(f"\n保存annotations到: {annotations_file}")
with open(annotations_file, 'w') as f:
    json.dump(annotations, f, indent=2)

print(f"✅ 已保存 {len(annotations)} 个标注")

# 数据集统计
print("\n" + "=" * 80)
print("数据集统计")
print("=" * 80)
print(f"总样本数: {len(annotations)}")
print(f"图片目录: {images_target}")
print(f"标注文件: {annotations_file}")

# 显示一些样本
print("\n样本示例:")
for i, ann in enumerate(annotations[:3]):
    print(f"\n样本 {i+1}:")
    print(f"  图片: {ann['image']}")
    print(f"  掩码数量: {len(ann['mask'])}")
    print(f"  第一个掩码点数: {len(ann['mask'][0]) // 2}")

print("\n" + "=" * 80)
print("✅ 数据准备完成！")
print("=" * 80)
print(f"\n下一步: 修改配置文件指向新数据集")
print(f"  数据路径: {target_dir}")
