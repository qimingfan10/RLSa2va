"""
使用新训练模型 (iter_3672) 进行推理评估
对比新旧模型性能
"""
import os
import sys
import json
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/Sa2VA')

print("=" * 80)
print("使用新训练模型 (iter_3672) 进行推理评估")
print("=" * 80)

# 配置 - 使用新模型
NEW_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"  # ✨ 新模型
OLD_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"  # 旧模型（对比用）
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/new_model_evaluation_results"
NUM_SAMPLES = 10

print(f"新模型路径: {NEW_MODEL_PATH}")
print(f"旧模型路径: {OLD_MODEL_PATH}")
print(f"数据路径: {DATA_ROOT}")
print(f"评估样本数: {NUM_SAMPLES}")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

# 评价指标计算
def calculate_metrics(pred_mask, gt_mask):
    """计算分割评价指标"""
    pred_flat = (pred_mask > 127).flatten().astype(int)
    gt_flat = (gt_mask > 127).flatten().astype(int)
    
    if len(np.unique(gt_flat)) == 1 and len(np.unique(pred_flat)) == 1:
        if gt_flat[0] == pred_flat[0]:
            return {
                'IoU': 1.0, 'Dice': 1.0, 'Precision': 1.0, 
                'Recall': 1.0, 'Accuracy': 1.0, 'Pixel_Accuracy': 1.0
            }
        else:
            return {
                'IoU': 0.0, 'Dice': 0.0, 'Precision': 0.0, 
                'Recall': 0.0, 'Accuracy': 0.0, 'Pixel_Accuracy': 0.0
            }
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    pixel_acc = np.sum(pred_flat == gt_flat) / len(gt_flat)
    
    return {
        'IoU': float(iou),
        'Dice': float(dice),
        'Precision': float(precision),
        'Recall': float(recall),
        'Accuracy': float(accuracy),
        'Pixel_Accuracy': float(pixel_acc)
    }

# 加载新模型
print("=" * 80)
print("步骤1: 加载新训练的HuggingFace模型 (iter_3672)")
print("=" * 80)

try:
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        NEW_MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载成功")
    
    print("\n加载新模型...")
    print(f"  模型来源: iter_3672.pth")
    print(f"  训练时间: Nov 23 21:41")
    print(f"  训练步数: 3672步 (3 epochs)")
    print(f"  训练Loss: 13.76 → 1.08")
    
    model = AutoModelForCausalLM.from_pretrained(
        NEW_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 新模型加载成功")
    print(f"设备分配: {model.hf_device_map}")
    
    model.eval()
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    MODEL_LOADED = False
    exit(1)

# 加载数据集 - 使用与旧模型相同的样本进行公平对比
print("\n" + "=" * 80)
print("步骤2: 加载数据集（与旧模型使用相同样本）")
print("=" * 80)

with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集总数: {len(dataset)}")

# 使用相同的随机种子，确保选择相同的样本
random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"已选择 {NUM_SAMPLES} 张图片进行评估（与旧模型相同）")
for i, sample in enumerate(test_samples, 1):
    print(f"  {i}. {sample['image']}")
print()

# 推理和评估
print("=" * 80)
print("步骤3: 使用新模型推理")
print("=" * 80)

all_metrics = []
results = []
successful_inferences = 0
failed_inferences = 0

for idx, sample in enumerate(test_samples):
    print(f"\n[{idx+1}/{NUM_SAMPLES}] 处理: {sample['image']}")
    print("-" * 80)
    
    # 加载图片
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        print(f"  ❌ 图片不存在: {img_path}")
        failed_inferences += 1
        continue
    
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    print(f"  图片尺寸: {w} x {h}")
    
    # 创建Ground Truth mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    gt_pixels = np.sum(gt_mask > 127)
    print(f"  GT像素数: {gt_pixels} ({gt_pixels/(h*w)*100:.2f}%)")
    
    # 使用新模型进行推理
    print(f"  🔄 使用新模型 (iter_3672) 推理...")
    
    try:
        text = "<image>Please segment the blood vessel."
        
        result = model.predict_forward(
            image=image,
            text=text,
            tokenizer=tokenizer,
            processor=None,
        )
        
        prediction_text = result.get('prediction', '')
        print(f"  📝 模型输出: {prediction_text}")
        
        # 检查是否有分割结果
        if '[SEG]' in prediction_text and 'prediction_masks' in result:
            pred_masks = result['prediction_masks']
            
            if len(pred_masks) > 0:
                pred_mask = pred_masks[0][0]
                
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                
                if pred_mask.shape != (h, w):
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                if pred_mask.max() <= 1.0:
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                else:
                    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
                
                pred_pixels = np.sum(pred_mask > 127)
                print(f"  ✅ 预测成功！预测像素数: {pred_pixels} ({pred_pixels/(h*w)*100:.2f}%)")
                successful_inferences += 1
            else:
                print(f"  ⚠️  没有分割结果，使用空mask")
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                failed_inferences += 1
        else:
            print(f"  ⚠️  输出中没有[SEG]标记，使用空mask")
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            failed_inferences += 1
    
    except Exception as e:
        print(f"  ❌ 推理失败: {e}")
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        failed_inferences += 1
    
    # 计算评价指标
    metrics = calculate_metrics(pred_mask, gt_mask)
    all_metrics.append(metrics)
    
    print(f"  📊 评价指标:")
    print(f"     IoU (Jaccard):    {metrics['IoU']:.4f}")
    print(f"     Dice Score:       {metrics['Dice']:.4f}")
    print(f"     Precision:        {metrics['Precision']:.4f}")
    print(f"     Recall:           {metrics['Recall']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('New Model Prediction\n(iter_3672, 3 epochs)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(image_np)
    axes[1, 0].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1, 0].set_title('GT Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_np)
    axes[1, 1].imshow(pred_mask, alpha=0.5, cmap='Greens')
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    diff = np.abs(pred_mask.astype(float) - gt_mask.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(
        f'Difference Map\nIoU={metrics["IoU"]:.3f}, Dice={metrics["Dice"]:.3f}', 
        fontsize=14, fontweight='bold'
    )
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_filename = f"new_model_{idx+1:02d}_{os.path.basename(sample['image'])}"
    output_path = os.path.join(OUTPUT_DIR, "predictions", output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 保存到: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'metrics': metrics,
        'output': output_path
    })

# 总体评估
print("\n" + "=" * 80)
print("步骤4: 新模型评估结果")
print("=" * 80)

if len(all_metrics) > 0:
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n推理统计:")
    print(f"  成功推理: {successful_inferences}/{NUM_SAMPLES}")
    print(f"  失败推理: {failed_inferences}/{NUM_SAMPLES}")
    print(f"  成功率: {successful_inferences/NUM_SAMPLES*100:.1f}%")
    
    print(f"\n新模型平均指标:")
    print(f"  IoU (Jaccard):      {avg_metrics['IoU']:.4f}")
    print(f"  Dice Score:         {avg_metrics['Dice']:.4f}")
    print(f"  Precision:          {avg_metrics['Precision']:.4f}")
    print(f"  Recall:             {avg_metrics['Recall']:.4f}")
    print(f"  Accuracy:           {avg_metrics['Accuracy']:.4f}")
    print(f"  Pixel Accuracy:     {avg_metrics['Pixel_Accuracy']:.4f}")
    
    # 保存结果
    detailed_results = {
        'model_info': {
            'model_path': NEW_MODEL_PATH,
            'model_type': 'HuggingFace Sa2VA-26B',
            'source_checkpoint': 'iter_3672.pth',
            'training_date': 'Nov 23 21:41',
            'training_steps': 3672,
            'training_epochs': 3,
            'training_loss': '13.76 → 1.08'
        },
        'inference_method': 'predict_forward (official)',
        'successful_inferences': successful_inferences,
        'failed_inferences': failed_inferences,
        'total_samples': NUM_SAMPLES,
        'success_rate': successful_inferences / NUM_SAMPLES,
        'average_metrics': avg_metrics,
        'per_sample_results': results
    }
    
    results_path = os.path.join(OUTPUT_DIR, "new_model_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {results_path}")
    
    # 生成Markdown报告
    report_path = os.path.join(OUTPUT_DIR, "new_model_evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Sa2VA新模型评估报告 (iter_3672)\n\n")
        f.write(f"## 模型信息\n\n")
        f.write(f"- **模型**: Sa2VA-26B (HuggingFace格式)\n")
        f.write(f"- **源checkpoint**: iter_3672.pth\n")
        f.write(f"- **训练时间**: Nov 23 21:41\n")
        f.write(f"- **训练步数**: 3672步 (3 epochs)\n")
        f.write(f"- **训练Loss**: 13.76 → 1.08\n")
        f.write(f"- **推理方法**: `predict_forward` (官方推荐)\n")
        f.write(f"- **评估样本数**: {NUM_SAMPLES}\n\n")
        
        f.write(f"## 推理统计\n\n")
        f.write(f"- **成功**: {successful_inferences}/{NUM_SAMPLES} ({successful_inferences/NUM_SAMPLES*100:.1f}%)\n")
        f.write(f"- **失败**: {failed_inferences}/{NUM_SAMPLES}\n\n")
        
        f.write(f"## 平均评价指标\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| IoU (Jaccard) | {avg_metrics['IoU']:.4f} |\n")
        f.write(f"| Dice Score | {avg_metrics['Dice']:.4f} |\n")
        f.write(f"| Precision | {avg_metrics['Precision']:.4f} |\n")
        f.write(f"| Recall | {avg_metrics['Recall']:.4f} |\n")
        f.write(f"| Accuracy | {avg_metrics['Accuracy']:.4f} |\n")
        f.write(f"| Pixel Accuracy | {avg_metrics['Pixel_Accuracy']:.4f} |\n\n")
        
        f.write(f"## 逐样本结果\n\n")
        f.write(f"| ID | 图片 | IoU | Dice | Precision | Recall |\n")
        f.write(f"|----|------|-----|------|-----------|--------|\n")
        for result in results:
            m = result['metrics']
            f.write(f"| {result['sample_id']} | {result['image']} | {m['IoU']:.4f} | {m['Dice']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} |\n")
    
    print(f"✅ Markdown报告已保存到: {report_path}")

print("\n" + "=" * 80)
print("🎉 新模型评估完成！")
print("=" * 80)
print(f"\n结果目录: {OUTPUT_DIR}")
print()
print("=" * 80)
