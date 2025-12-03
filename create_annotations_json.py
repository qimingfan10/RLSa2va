#!/usr/bin/env python3
"""
将Segment_DATA_Merged_512数据转换为Sa2VA训练所需的annotations.json格式

官方格式：
[
    {
        "image": "image001.jpg",
        "text": ["description1", "description2"],
        "mask": [
            [[x1,y1,x2,y2,...], [...]],  # polygon for object 1
            [[x1,y1,x2,y2,...]]           # polygon for object 2
        ]
    }
]
"""

import json
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def mask_to_polygon(mask, min_area=100):
    """
    将binary mask转换为polygon坐标列表
    
    Args:
        mask: numpy array, binary mask (0/255 or 0/1)
        min_area: 最小轮廓面积，过滤小噪声
        
    Returns:
        list of polygons, 每个polygon是[x1,y1,x2,y2,...]格式
    """
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # 过滤小轮廓
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 简化轮廓（可选，减少点数）
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为[x1,y1,x2,y2,...]格式
        polygon = approx.reshape(-1, 2).flatten().tolist()
        
        # 确保至少有3个点（6个坐标）
        if len(polygon) >= 6:
            polygons.append(polygon)
    
    return polygons


def create_annotations_json(
    data_root='/home/ubuntu/Sa2VA/Segment_DATA_Merged_512',
    output_path=None,
    text_description='blood vessel',
    min_area=100,
    validate=True
):
    """
    创建annotations.json文件
    
    Args:
        data_root: 数据根目录
        output_path: 输出annotations.json路径，默认为data_root/annotations.json
        text_description: 分割对象的文本描述
        min_area: 最小轮廓面积
        validate: 是否验证数据
    """
    if output_path is None:
        output_path = os.path.join(data_root, 'annotations.json')
    
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')
    
    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    annotations = []
    skipped = 0
    
    for img_path in tqdm(image_files, desc="处理图像"):
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(masks_dir, mask_name)
        
        # 检查mask是否存在
        if not os.path.exists(mask_path):
            print(f"警告: {mask_name} 不存在，跳过")
            skipped += 1
            continue
        
        try:
            # 读取mask
            mask = np.array(Image.open(mask_path).convert('L'))
            
            # 检查mask是否为空
            if mask.max() == 0:
                print(f"警告: {mask_name} 是空mask，跳过")
                skipped += 1
                continue
            
            # 转换为polygon
            polygons = mask_to_polygon(mask, min_area=min_area)
            
            if len(polygons) == 0:
                print(f"警告: {mask_name} 没有有效轮廓，跳过")
                skipped += 1
                continue
            
            # 创建annotation
            annotation = {
                'image': img_name,
                'text': [text_description],
                'mask': [polygons]  # 注意：外层list对应text，内层list对应该object的多个polygon
            }
            
            annotations.append(annotation)
            
        except Exception as e:
            print(f"错误: 处理 {img_name} 时出错: {e}")
            skipped += 1
            continue
    
    print(f"\n处理完成:")
    print(f"  成功: {len(annotations)} 个")
    print(f"  跳过: {skipped} 个")
    
    # 验证数据
    if validate and len(annotations) > 0:
        print("\n验证数据格式...")
        sample = annotations[0]
        print(f"  示例数据:")
        print(f"    image: {sample['image']}")
        print(f"    text: {sample['text']}")
        print(f"    mask数量: {len(sample['mask'])}")
        print(f"    第一个mask的polygon数量: {len(sample['mask'][0])}")
        print(f"    第一个polygon的点数: {len(sample['mask'][0][0]) // 2}")
    
    # 保存annotations.json
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ 已保存到: {output_path}")
    return output_path


def validate_annotations_json(json_path):
    """验证annotations.json格式是否正确"""
    print(f"\n验证 {json_path}...")
    
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"总样本数: {len(annotations)}")
    
    errors = []
    for i, ann in enumerate(annotations):
        # 检查必要字段
        if 'image' not in ann:
            errors.append(f"样本 {i}: 缺少 'image' 字段")
        if 'text' not in ann:
            errors.append(f"样本 {i}: 缺少 'text' 字段")
        if 'mask' not in ann:
            errors.append(f"样本 {i}: 缺少 'mask' 字段")
            continue
        
        # 检查数据类型
        if not isinstance(ann['text'], list):
            errors.append(f"样本 {i}: 'text' 应该是list")
        if not isinstance(ann['mask'], list):
            errors.append(f"样本 {i}: 'mask' 应该是list")
            continue
        
        # 检查text和mask长度是否一致
        if len(ann['text']) != len(ann['mask']):
            errors.append(f"样本 {i}: text长度({len(ann['text'])}) != mask长度({len(ann['mask'])})")
        
        # 检查polygon格式
        for j, polygons in enumerate(ann['mask']):
            if not isinstance(polygons, list):
                errors.append(f"样本 {i}, mask {j}: polygons应该是list")
                continue
            
            for k, polygon in enumerate(polygons):
                if not isinstance(polygon, list):
                    errors.append(f"样本 {i}, mask {j}, polygon {k}: polygon应该是list")
                    continue
                
                if len(polygon) < 6:  # 至少3个点
                    errors.append(f"样本 {i}, mask {j}, polygon {k}: 点数太少({len(polygon)//2})")
                
                if len(polygon) % 2 != 0:
                    errors.append(f"样本 {i}, mask {j}, polygon {k}: 坐标数必须是偶数")
    
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for err in errors[:10]:  # 只显示前10个
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors)-10} 个错误")
    else:
        print("\n✅ 格式验证通过！")
    
    return len(errors) == 0


def visualize_annotation(
    json_path,
    data_root,
    sample_idx=0,
    output_path='visualization.jpg'
):
    """可视化一个样本，检查polygon是否正确"""
    import cv2
    
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    ann = annotations[sample_idx]
    
    # 读取图像
    img_path = os.path.join(data_root, 'images', ann['image'])
    img = cv2.imread(img_path)
    
    # 绘制polygon
    for polygons in ann['mask']:
        for polygon in polygons:
            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    # 添加文本
    text = ', '.join(ann['text'])
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, img)
    print(f"\n✅ 可视化已保存到: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='创建Sa2VA训练所需的annotations.json')
    parser.add_argument('--data_root', default='/home/ubuntu/Sa2VA/Segment_DATA_Merged_512',
                        help='数据根目录')
    parser.add_argument('--output', default=None,
                        help='输出annotations.json路径')
    parser.add_argument('--text', default='blood vessel',
                        help='分割对象的文本描述')
    parser.add_argument('--min_area', type=int, default=100,
                        help='最小轮廓面积（过滤噪声）')
    parser.add_argument('--no_validate', action='store_true',
                        help='不验证数据')
    parser.add_argument('--visualize', type=int, default=None,
                        help='可视化指定样本索引')
    
    args = parser.parse_args()
    
    # 创建annotations.json
    output_path = create_annotations_json(
        data_root=args.data_root,
        output_path=args.output,
        text_description=args.text,
        min_area=args.min_area,
        validate=not args.no_validate
    )
    
    # 验证格式
    if not args.no_validate:
        validate_annotations_json(output_path)
    
    # 可视化
    if args.visualize is not None:
        visualize_annotation(
            output_path,
            args.data_root,
            sample_idx=args.visualize,
            output_path=f'visualization_{args.visualize}.jpg'
        )
