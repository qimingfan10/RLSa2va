#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的数据集可视化脚本
由于GPU显存被训练占用,我们先可视化数据集的图片和标注
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_dataset_samples(data_root, num_samples=10, output_dir='dataset_visualization'):
    """
    可视化数据集样本
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载annotations
    ann_file = Path(data_root) / 'annotations.json'
    with open(ann_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"数据集总样本数: {len(annotations)}")
    print(f"可视化前 {num_samples} 个样本\n")
    
    # 选择前N个样本
    samples = annotations[:num_samples]
    
    results = []
    
    for idx, sample in enumerate(samples):
        print(f"[{idx+1}/{num_samples}] 处理样本...")
        
        # 获取图片路径
        image_path = Path(data_root) / 'images' / sample['image']
        if not image_path.exists():
            print(f"  ⚠️  图片不存在: {image_path}")
            continue
        
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # 获取标注信息
        text_labels = sample.get('text', [])
        masks = sample.get('mask', [])
        
        # 构建问题和答案
        if text_labels:
            question = f"请分割图像中的{', '.join(text_labels)}"
            answer = f"图像中包含 {len(masks)} 个{text_labels[0]}区域"
        else:
            question = "请分割图像中的目标"
            answer = f"图像中包含 {len(masks)} 个目标区域"
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左侧: 原图
        axes[0].imshow(img_array)
        axes[0].set_title(f'原始图片\n{Path(image_path).name}', fontsize=10)
        axes[0].axis('off')
        
        # 右侧: 标注信息
        axes[1].axis('off')
        info_text = f"""
样本 #{idx+1}

图片: {Path(image_path).name}
尺寸: {img_array.shape[1]} x {img_array.shape[0]}

问题:
{question[:200]}...

答案:
{answer[:200]}...

掩码数量: {len(sample.get('masks', []))}
        """
        axes[1].text(0.1, 0.5, info_text, 
                    fontsize=11, 
                    verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        output_file = output_path / f'sample_{idx+1:03d}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 已保存: {output_file}")
        
        results.append({
            'sample_id': idx + 1,
            'image': str(image_path.name),
            'image_size': f"{img_array.shape[1]}x{img_array.shape[0]}",
            'num_masks': len(sample.get('masks', [])),
            'question_preview': question[:100],
            'answer_preview': answer[:100],
        })
    
    # 保存摘要
    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 可视化完成!")
    print(f"📊 输出目录: {output_path}")
    print(f"📄 摘要文件: {summary_file}")
    
    # 打印统计信息
    print(f"\n📊 数据集统计:")
    print(f"  - 总样本数: {len(annotations)}")
    print(f"  - 已可视化: {len(results)}")
    if results:
        avg_masks = sum(r['num_masks'] for r in results) / len(results)
        print(f"  - 平均掩码数: {avg_masks:.2f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化血管分割数据集')
    parser.add_argument('--data-root', type=str, 
                       default='/home/ubuntu/Sa2VA/data/vessel_data',
                       help='数据集根目录')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='可视化样本数量')
    parser.add_argument('--output-dir', type=str, default='dataset_visualization',
                       help='输出目录')
    
    args = parser.parse_args()
    
    visualize_dataset_samples(
        data_root=args.data_root,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
