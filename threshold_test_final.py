"""
最终阈值测试 - 使用修改后的模型
"""
import os
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_metrics(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    
    return {'dice': dice, 'recall': recall, 'precision': precision}


def main():
    print("="*80)
    print("阈值扫描测试（使用修改后的模型）")
    print("="*80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # 加载模型
    print("\n加载模型...")
    model_path = '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    print("✅ 模型加载完成")
    
    # 使用正确的数据集路径
    print("\n使用Segment_DATA_Merged_512数据集（前20张）...")
    data_root = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512'
    
    # 直接列出图像文件
    import glob
    image_files = sorted(glob.glob(os.path.join(data_root, 'images', '*.jpg')))[:20]
    print(f"找到{len(image_files)}张图像")
    
    # 步骤1: 推理获取概率图
    print("\n步骤1: 推理获取概率图...")
    prob_data = []
    
    for image_path in tqdm(image_files, desc="推理"):
        img_name = os.path.basename(image_path)
        
        # mask文件名 = 图像文件名.jpg -> _mask.png
        mask_name = img_name.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png')
        mask_path = os.path.join(data_root, 'masks', mask_name)
        
        if not os.path.exists(image_path):
            print(f"  图像不存在: {img_name}")
            continue
        if not os.path.exists(mask_path):
            print(f"  Mask不存在: {mask_name}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        try:
            result = model.predict_forward(
                image=image,
                text='<image>\nPlease segment the blood vessel.',
                tokenizer=tokenizer
            )
            
            # 检查是否有概率图
            if 'probability_maps' in result and len(result['probability_maps']) > 0:
                prob_map = result['probability_maps'][0][0]
                prob_data.append({
                    'image': img_name,
                    'prob_map': prob_map,
                    'gt_mask': gt_mask
                })
                print(f"  ✅ {img_name}: 概率图shape {prob_map.shape}, 范围[{prob_map.min():.3f}, {prob_map.max():.3f}]")
            else:
                print(f"  ❌ {img_name}: 没有返回概率图")
                
        except Exception as e:
            print(f"  ❌ {img_name}: 失败 - {e}")
    
    if len(prob_data) == 0:
        print("\n❌ 没有成功获取任何概率图！")
        return
    
    print(f"\n✅ 成功获取{len(prob_data)}个概率图")
    
    # 步骤2: 测试不同阈值
    print("\n步骤2: 测试不同阈值...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    print(f"\n{'阈值':<8} {'Dice':<8} {'Recall':<8} {'Precision':<10}")
    print("-" * 40)
    
    all_results = {}
    
    for threshold in thresholds:
        metrics_list = []
        
        for data in prob_data:
            pred_mask = (data['prob_map'] > threshold).astype(np.uint8)
            metrics = calculate_metrics(pred_mask, data['gt_mask'])
            metrics_list.append(metrics)
        
        avg_dice = np.mean([m['dice'] for m in metrics_list])
        avg_recall = np.mean([m['recall'] for m in metrics_list])
        avg_precision = np.mean([m['precision'] for m in metrics_list])
        
        all_results[threshold] = {
            'dice': avg_dice,
            'recall': avg_recall,
            'precision': avg_precision
        }
        
        print(f"{threshold:<8.2f} {avg_dice:<8.4f} {avg_recall:<8.4f} {avg_precision:<10.4f}")
    
    # 找最优
    print("\n" + "="*80)
    print("最优阈值")
    print("="*80)
    
    best_dice = max(all_results.items(), key=lambda x: x[1]['dice'])
    best_recall = max(all_results.items(), key=lambda x: x[1]['recall'])
    
    print(f"\n🎯 最高Dice: 阈值={best_dice[0]:.2f}")
    print(f"   Dice:      {best_dice[1]['dice']:.4f}")
    print(f"   Recall:    {best_dice[1]['recall']:.4f}")
    print(f"   Precision: {best_dice[1]['precision']:.4f}")
    
    print(f"\n🎯 最高Recall: 阈值={best_recall[0]:.2f}")
    print(f"   Dice:      {best_recall[1]['dice']:.4f}")
    print(f"   Recall:    {best_recall[1]['recall']:.4f}")
    print(f"   Precision: {best_recall[1]['precision']:.4f}")
    
    # Baseline (0.5)
    baseline = all_results[0.5]
    print(f"\n📊 Baseline (阈值0.5):")
    print(f"   Dice:      {baseline['dice']:.4f}")
    print(f"   Recall:    {baseline['recall']:.4f}")
    print(f"   Precision: {baseline['precision']:.4f}")
    
    # 保存结果
    output_dir = './threshold_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump({
            'num_samples': len(prob_data),
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n✅ 结果保存至: {output_dir}/results.json")
    
    # 绘图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    thresholds_list = sorted(all_results.keys())
    dices = [all_results[t]['dice'] for t in thresholds_list]
    recalls = [all_results[t]['recall'] for t in thresholds_list]
    precisions = [all_results[t]['precision'] for t in thresholds_list]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds_list, dices, 'b-o', label='Dice', linewidth=2, markersize=8)
    plt.plot(thresholds_list, recalls, 'g-s', label='Recall', linewidth=2, markersize=8)
    plt.plot(thresholds_list, precisions, 'm-^', label='Precision', linewidth=2, markersize=8)
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target 0.85')
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default 0.5')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Metrics vs Threshold (Corrected)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_curve.png', dpi=150)
    print(f"✅ 曲线图保存至: {output_dir}/threshold_curve.png")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
