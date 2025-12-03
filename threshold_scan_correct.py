"""
正确的阈值扫描实验
使用原始概率图进行阈值测试，而非二值化mask
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')

from transformers import AutoModelForCausalLM, AutoTokenizer
from get_probability_maps import predict_forward_with_probs, apply_threshold


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    # Dice
    dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    
    # IoU
    iou = intersection / union if union > 0 else 0.0
    
    # Recall (Sensitivity)
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    
    # Precision
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    
    return {
        'dice': dice,
        'iou': iou,
        'recall': recall,
        'precision': precision
    }


def threshold_scanning_experiment(
    model,
    tokenizer,
    data_root,
    thresholds,
    max_samples=50,
    output_dir='./threshold_scan_results_correct'
):
    """
    正确的阈值扫描实验
    
    Args:
        model: Sa2VA模型
        tokenizer: tokenizer
        data_root: 数据集根目录
        thresholds: 要测试的阈值列表
        max_samples: 最大测试样本数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据集...")
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        all_data = json.load(f)
    
    # 使用验证集（后20%）
    n_train = int(len(all_data) * 0.8)
    val_data = all_data[n_train:n_train + max_samples]
    
    print(f"加载{len(val_data)}张图像进行测试")
    
    # 为每个样本推理一次，保存概率图
    print("\n步骤1: 推理并保存概率图...")
    prob_maps_data = []
    
    for idx, item in enumerate(tqdm(val_data, desc="推理")):
        image_path = os.path.join(data_root, 'images', item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # 加载GT mask
        mask_path = os.path.join(data_root, 'masks', item['mask'])
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # 推理获取概率图
        try:
            result = predict_forward_with_probs(
                model=model,
                image=image,
                text='<image>\nPlease segment the blood vessel.',
                tokenizer=tokenizer
            )
            
            if len(result['probability_maps']) > 0:
                prob_map = result['probability_maps'][0][0]  # shape: (H, W)
                
                prob_maps_data.append({
                    'image_name': item['image'],
                    'prob_map': prob_map,
                    'gt_mask': gt_mask
                })
            else:
                print(f"⚠️ 样本 {idx} 推理失败")
                
        except Exception as e:
            print(f"⚠️ 样本 {idx} 推理出错: {e}")
            continue
    
    print(f"\n✅ 成功推理 {len(prob_maps_data)} 个样本")
    
    # 对每个阈值进行测试
    print("\n步骤2: 测试不同阈值...")
    all_results = defaultdict(list)
    
    for threshold in tqdm(thresholds, desc="阈值扫描"):
        threshold_metrics = []
        
        for data in prob_maps_data:
            prob_map = data['prob_map']
            gt_mask = data['gt_mask']
            
            # 应用阈值
            pred_mask = apply_threshold(prob_map, threshold)
            
            # 计算指标
            metrics = calculate_metrics(pred_mask, gt_mask)
            threshold_metrics.append(metrics)
        
        # 计算平均指标
        avg_metrics = {
            'threshold': float(threshold),
            'avg_dice': np.mean([m['dice'] for m in threshold_metrics]),
            'avg_recall': np.mean([m['recall'] for m in threshold_metrics]),
            'avg_precision': np.mean([m['precision'] for m in threshold_metrics]),
            'avg_iou': np.mean([m['iou'] for m in threshold_metrics]),
            'std_dice': np.std([m['dice'] for m in threshold_metrics]),
            'std_recall': np.std([m['recall'] for m in threshold_metrics]),
            'std_precision': np.std([m['precision'] for m in threshold_metrics]),
        }
        
        all_results[threshold] = avg_metrics
        
        print(f"  阈值{threshold:.2f}: "
              f"Dice={avg_metrics['avg_dice']:.4f}, "
              f"Recall={avg_metrics['avg_recall']:.4f}, "
              f"Precision={avg_metrics['avg_precision']:.4f}")
    
    # 保存结果
    results_path = os.path.join(output_dir, 'threshold_scan_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'num_samples': len(prob_maps_data),
            'thresholds_tested': [float(t) for t in thresholds],
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n✅ 结果保存至: {results_path}")
    
    # 找到最优阈值
    print("\n" + "="*80)
    print("最优阈值分析")
    print("="*80)
    
    # 按Dice排序
    sorted_by_dice = sorted(all_results.items(), key=lambda x: x[1]['avg_dice'], reverse=True)
    print("\n按Dice Score排序:")
    for i, (thresh, metrics) in enumerate(sorted_by_dice[:5]):
        print(f"  #{i+1} 阈值{thresh:.2f}: "
              f"Dice={metrics['avg_dice']:.4f}, "
              f"Recall={metrics['avg_recall']:.4f}, "
              f"Precision={metrics['avg_precision']:.4f}")
    
    # 按Recall排序
    sorted_by_recall = sorted(all_results.items(), key=lambda x: x[1]['avg_recall'], reverse=True)
    print("\n按Recall排序:")
    for i, (thresh, metrics) in enumerate(sorted_by_recall[:5]):
        print(f"  #{i+1} 阈值{thresh:.2f}: "
              f"Dice={metrics['avg_dice']:.4f}, "
              f"Recall={metrics['avg_recall']:.4f}, "
              f"Precision={metrics['avg_precision']:.4f}")
    
    # F1-Score (平衡Recall和Precision)
    for thresh, metrics in all_results.items():
        r = metrics['avg_recall']
        p = metrics['avg_precision']
        metrics['f1_score'] = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    
    sorted_by_f1 = sorted(all_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    print("\n按F1-Score排序:")
    for i, (thresh, metrics) in enumerate(sorted_by_f1[:5]):
        print(f"  #{i+1} 阈值{thresh:.2f}: "
              f"F1={metrics['f1_score']:.4f}, "
              f"Dice={metrics['avg_dice']:.4f}, "
              f"Recall={metrics['avg_recall']:.4f}, "
              f"Precision={metrics['avg_precision']:.4f}")
    
    # 绘制曲线
    plot_threshold_curves(all_results, output_dir)
    
    return all_results, prob_maps_data


def plot_threshold_curves(results, output_dir):
    """绘制阈值-性能曲线"""
    import matplotlib.pyplot as plt
    
    thresholds = sorted(results.keys())
    dices = [results[t]['avg_dice'] for t in thresholds]
    recalls = [results[t]['avg_recall'] for t in thresholds]
    precisions = [results[t]['avg_precision'] for t in thresholds]
    f1_scores = [results[t]['f1_score'] for t in thresholds]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dice vs Threshold
    axes[0, 0].plot(thresholds, dices, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Target 0.85')
    axes[0, 0].set_xlabel('Threshold', fontsize=12)
    axes[0, 0].set_ylabel('Dice Score', fontsize=12)
    axes[0, 0].set_title('Dice vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    best_dice_idx = np.argmax(dices)
    axes[0, 0].plot(thresholds[best_dice_idx], dices[best_dice_idx], 'r*', markersize=15, 
                    label=f'Best: {thresholds[best_dice_idx]:.2f}')
    
    # Recall vs Threshold
    axes[0, 1].plot(thresholds, recalls, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Target 0.85')
    axes[0, 1].set_xlabel('Threshold', fontsize=12)
    axes[0, 1].set_ylabel('Recall', fontsize=12)
    axes[0, 1].set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    best_recall_idx = np.argmax(recalls)
    axes[0, 1].plot(thresholds[best_recall_idx], recalls[best_recall_idx], 'r*', markersize=15,
                    label=f'Best: {thresholds[best_recall_idx]:.2f}')
    
    # Precision vs Threshold
    axes[1, 0].plot(thresholds, precisions, 'm-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    best_precision_idx = np.argmax(precisions)
    axes[1, 0].plot(thresholds[best_precision_idx], precisions[best_precision_idx], 'r*', markersize=15,
                    label=f'Best: {thresholds[best_precision_idx]:.2f}')
    axes[1, 0].legend()
    
    # All metrics
    axes[1, 1].plot(thresholds, dices, 'b-o', label='Dice', linewidth=2, markersize=6)
    axes[1, 1].plot(thresholds, recalls, 'g-s', label='Recall', linewidth=2, markersize=6)
    axes[1, 1].plot(thresholds, precisions, 'm-^', label='Precision', linewidth=2, markersize=6)
    axes[1, 1].plot(thresholds, f1_scores, 'r-d', label='F1-Score', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Target 0.85')
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('All Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'threshold_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 曲线图保存至: {plot_path}")


def main():
    print("="*80)
    print("正确的阈值扫描实验")
    print("使用原始概率图 (而非二值化mask)")
    print("="*80)
    
    # 加载模型
    print("\n加载Sa2VA模型...")
    model_path = '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model.eval()
    print("✅ 模型加载成功")
    
    # 设置阈值范围
    thresholds = np.arange(0.1, 0.9, 0.05)  # 0.1 到 0.85，步长0.05
    
    # 运行实验
    results, prob_maps = threshold_scanning_experiment(
        model=model,
        tokenizer=tokenizer,
        data_root='/home/ubuntu/Sa2VA/data/merged_vessel_data',
        thresholds=thresholds,
        max_samples=50,
        output_dir='./threshold_scan_results_correct'
    )
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()
