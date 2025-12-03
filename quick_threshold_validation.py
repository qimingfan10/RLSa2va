"""
快速验证：阈值对Dice/Recall的影响
目标：验证是否只需要调整阈值就能提升Dice到0.85+
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(pred_mask, gt_mask):
    """计算评估指标"""
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    
    dice = (2.0 * intersection) / (union + 1e-8)
    iou = intersection / (pred.sum() + gt.sum() - intersection + 1e-8)
    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }


def load_dataset(data_root, max_samples=None):
    """加载数据集"""
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    if max_samples is not None:
        annotations = annotations[:max_samples]
    
    dataset = []
    images_dir = os.path.join(data_root, 'images')
    
    for ann in tqdm(annotations, desc="加载数据集"):
        try:
            image_path = os.path.join(images_dir, ann['image'])
            image = Image.open(image_path).convert('RGB')
            
            # 生成GT mask
            width, height = image.size
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            
            if 'mask' in ann and len(ann['mask']) > 0:
                polygons = ann['mask']
                for polygon in polygons:
                    if len(polygon) >= 6:
                        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                        draw.polygon(points, fill=255)
            
            mask = np.array(mask_img)
            
            dataset.append({
                'image': image,
                'mask': mask,
                'image_path': ann['image']
            })
            
        except Exception as e:
            print(f"加载失败 {ann.get('image', 'unknown')}: {e}")
            continue
    
    print(f"✅ 加载{len(dataset)}个样本")
    return dataset


def predict_with_sa2va(model, tokenizer, image, prompt, device):
    """使用Sa2VA进行预测，返回概率图"""
    # 确保prompt包含<image>标记
    if '<image>' not in prompt:
        text_with_image = f"<image>\n{prompt}"
    else:
        text_with_image = prompt
    
    with torch.no_grad():
        result = model.predict_forward(
            image=image,
            text=text_with_image,
            tokenizer=tokenizer
        )
    
    if isinstance(result, dict) and 'prediction_masks' in result:
        masks = result['prediction_masks']
        if len(masks) > 0:
            pred_mask = masks[0]
            if len(pred_mask.shape) > 2:
                pred_mask = pred_mask[0]
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.cpu().numpy()
            
            # 返回概率值，不做二值化
            return pred_mask.astype(np.float32)
    
    return None


def threshold_scan_experiment(args):
    """阈值扫描实验"""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载Sa2VA模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✅ Sa2VA模型加载成功")
    
    # 加载数据集
    print(f"\n加载数据集...")
    dataset = load_dataset(args.data_root, max_samples=args.max_samples)
    
    # 获取所有概率图
    print(f"\n生成预测概率图...")
    probability_maps = []
    gt_masks = []
    
    for sample in tqdm(dataset, desc="Sa2VA预测"):
        prob_map = predict_with_sa2va(
            model, tokenizer, sample['image'], args.prompt, device
        )
        if prob_map is not None:
            probability_maps.append(prob_map)
            gt_masks.append(sample['mask'])
    
    print(f"✅ 获得{len(probability_maps)}个概率图")
    
    # 阈值扫描
    print(f"\n开始阈值扫描...")
    thresholds = np.arange(args.min_threshold, args.max_threshold, args.threshold_step)
    results = []
    
    for threshold in tqdm(thresholds, desc="扫描阈值"):
        metrics_list = []
        
        for prob_map, gt_mask in zip(probability_maps, gt_masks):
            # 二值化
            pred_mask = (prob_map > threshold).astype(np.uint8) * 255
            
            # 调整大小匹配GT
            if pred_mask.shape != gt_mask.shape:
                from PIL import Image
                pred_mask = np.array(Image.fromarray(pred_mask).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                ))
            
            # 计算指标
            metrics = calculate_metrics(pred_mask, gt_mask)
            metrics_list.append(metrics)
        
        # 平均指标
        avg_metrics = {
            'threshold': float(threshold),
            'dice': np.mean([m['dice'] for m in metrics_list]),
            'iou': np.mean([m['iou'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list])
        }
        results.append(avg_metrics)
        
        print(f"  Threshold={threshold:.2f}: Dice={avg_metrics['dice']:.4f}, "
              f"Recall={avg_metrics['recall']:.4f}, Precision={avg_metrics['precision']:.4f}")
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x['dice'])
    print(f"\n{'='*60}")
    print(f"🎯 最佳阈值: {best_result['threshold']:.2f}")
    print(f"{'='*60}")
    print(f"Dice:      {best_result['dice']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"IoU:       {best_result['iou']:.4f}")
    print(f"{'='*60}")
    
    # 与baseline对比
    baseline_result = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]
    print(f"\n📊 与Baseline (threshold=0.5) 对比:")
    print(f"Dice:      {baseline_result['dice']:.4f} → {best_result['dice']:.4f} "
          f"({best_result['dice']-baseline_result['dice']:+.4f})")
    print(f"Recall:    {baseline_result['recall']:.4f} → {best_result['recall']:.4f} "
          f"({best_result['recall']-baseline_result['recall']:+.4f})")
    print(f"Precision: {baseline_result['precision']:.4f} → {best_result['precision']:.4f} "
          f"({best_result['precision']-baseline_result['precision']:+.4f})")
    
    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'threshold_scan_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'best_threshold': best_result,
            'baseline_threshold': baseline_result,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存至: {results_file}")
    
    # 绘制曲线
    plot_results(results, output_dir)
    
    return results, best_result


def plot_results(results, output_dir):
    """绘制阈值-指标曲线"""
    thresholds = [r['threshold'] for r in results]
    dice_scores = [r['dice'] for r in results]
    recall_scores = [r['recall'] for r in results]
    precision_scores = [r['precision'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: Dice
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, dice_scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
    best_idx = np.argmax(dice_scores)
    plt.plot(thresholds[best_idx], dice_scores[best_idx], 'r*', markersize=15, 
             label=f'Best: {dice_scores[best_idx]:.4f}@{thresholds[best_idx]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2: Recall
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, recall_scores, 'g-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
    best_idx = np.argmax(dice_scores)
    plt.plot(thresholds[best_idx], recall_scores[best_idx], 'r*', markersize=15,
             label=f'Best: {recall_scores[best_idx]:.4f}@{thresholds[best_idx]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图3: Precision
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, precision_scores, 'm-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
    best_idx = np.argmax(dice_scores)
    plt.plot(thresholds[best_idx], precision_scores[best_idx], 'r*', markersize=15,
             label=f'Best: {precision_scores[best_idx]:.4f}@{thresholds[best_idx]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图4: 综合对比
    plt.subplot(2, 2, 4)
    plt.plot(thresholds, dice_scores, 'b-', linewidth=2, label='Dice', marker='o', markersize=3)
    plt.plot(thresholds, recall_scores, 'g-', linewidth=2, label='Recall', marker='s', markersize=3)
    plt.plot(thresholds, precision_scores, 'm-', linewidth=2, label='Precision', marker='^', markersize=3)
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target: 0.85')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('All Metrics vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'threshold_scan_curves.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ 曲线图已保存至: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='快速验证：阈值扫描')
    
    parser.add_argument('--model_path', type=str,
                        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf',
                        help='Sa2VA模型路径')
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/Sa2VA/data/merged_vessel_data',
                        help='数据集根目录')
    parser.add_argument('--prompt', type=str,
                        default='Please segment the blood vessel.',
                        help='使用的prompt')
    parser.add_argument('--output_dir', type=str,
                        default='./threshold_validation_output',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='最大评估样本数')
    parser.add_argument('--min_threshold', type=float, default=0.1,
                        help='最小阈值')
    parser.add_argument('--max_threshold', type=float, default=0.9,
                        help='最大阈值')
    parser.add_argument('--threshold_step', type=float, default=0.05,
                        help='阈值步长')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("快速验证：阈值对Dice/Recall的影响")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.data_root}")
    print(f"Prompt: {args.prompt}")
    print(f"样本数: {args.max_samples}")
    print(f"阈值范围: [{args.min_threshold}, {args.max_threshold})")
    print(f"阈值步长: {args.threshold_step}")
    print("="*60 + "\n")
    
    results, best_result = threshold_scan_experiment(args)
    
    print("\n" + "="*60)
    print("🎯 结论:")
    print("="*60)
    
    # 判断是否需要RL
    baseline_dice = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]['dice']
    improvement = best_result['dice'] - baseline_dice
    
    if best_result['dice'] >= 0.85:
        print(f"✅ 通过调整阈值到{best_result['threshold']:.2f}，")
        print(f"   Dice已达到{best_result['dice']:.4f}，超过目标0.85！")
        print(f"   🎉 建议：直接使用动态阈值，无需复杂的RL训练！")
    elif improvement > 0.02:
        print(f"✅ 通过调整阈值到{best_result['threshold']:.2f}，")
        print(f"   Dice可提升{improvement:.4f} (从{baseline_dice:.4f}到{best_result['dice']:.4f})")
        print(f"   建议：结合动态阈值 + RL进一步优化")
    else:
        print(f"⚠️ 阈值调整效果有限（提升仅{improvement:.4f}）")
        print(f"   建议：直接进行RL微调（路径二：LoRA + PPO）")
    
    print("="*60)


if __name__ == '__main__':
    main()
