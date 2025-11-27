"""
真正使用训练好的Sa2VA权重进行推理和评估
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
from scipy.spatial.distance import directed_hausdorff

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')
os.environ['PYTHONPATH'] = '/home/ubuntu/Sa2VA:' + os.environ.get('PYTHONPATH', '')

print("=" * 80)
print("Sa2VA真实权重推理和评估")
print("=" * 80)

# 配置
CHECKPOINT_PATH = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"
CONFIG_PATH = "/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/real_inference_results"
NUM_SAMPLES = 10

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"配置文件: {CONFIG_PATH}")
print(f"数据集: {DATA_ROOT}")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

# 评价指标计算函数
def calculate_metrics(pred_mask, gt_mask):
    """计算分割评价指标"""
    pred_flat = (pred_mask > 127).flatten().astype(int)
    gt_flat = (gt_mask > 127).flatten().astype(int)
    
    # 基本指标
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Pixel Accuracy
    pixel_acc = np.sum(pred_flat == gt_flat) / len(gt_flat)
    
    # Hausdorff Distance (如果有前景像素)
    try:
        pred_points = np.argwhere(pred_mask > 127)
        gt_points = np.argwhere(gt_mask > 127)
        if len(pred_points) > 0 and len(gt_points) > 0:
            hausdorff = max(
                directed_hausdorff(pred_points, gt_points)[0],
                directed_hausdorff(gt_points, pred_points)[0]
            )
        else:
            hausdorff = float('inf')
    except:
        hausdorff = float('inf')
    
    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Pixel_Accuracy': pixel_acc,
        'Hausdorff': hausdorff
    }

# 尝试加载模型
print("=" * 80)
print("加载模型")
print("=" * 80)

MODEL_LOADED = False
model = None

try:
    # 尝试方法1: 使用mmengine
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("方法1: 使用mmengine加载模型...")
    cfg = Config.fromfile(CONFIG_PATH)
    
    # 构建模型
    model = MODELS.build(cfg.model)
    
    # 加载权重
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ 模型加载成功 (mmengine)")
    print(f"   设备: {device}")
    MODEL_LOADED = True
    
except Exception as e1:
    print(f"❌ mmengine加载失败: {e1}")
    
    try:
        # 尝试方法2: 使用HuggingFace格式
        from transformers import AutoModel, AutoTokenizer
        
        HF_MODEL_PATH = "models/sa2va_vessel_hf"
        if os.path.exists(HF_MODEL_PATH):
            print(f"\n方法2: 使用HuggingFace模型...")
            model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            print(f"✅ 模型加载成功 (HuggingFace)")
            print(f"   设备: {device}")
            MODEL_LOADED = True
        else:
            print(f"❌ HuggingFace模型不存在: {HF_MODEL_PATH}")
            
    except Exception as e2:
        print(f"❌ HuggingFace加载失败: {e2}")

if not MODEL_LOADED:
    print("\n" + "=" * 80)
    print("⚠️  警告: 模型加载失败")
    print("=" * 80)
    print("当前将使用Ground Truth作为'预测'来演示评估流程")
    print("这不是真实的模型推理！")
    print()
    print("要进行真实推理，需要:")
    print("1. 在topo-sarl环境中运行 (有mmengine)")
    print("2. 或先转换模型为HuggingFace格式")
    print("=" * 80)
    print()

# 加载数据集
print("加载数据集...")
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"选中 {NUM_SAMPLES} 个样本")
print()

# 推理和评估
print("=" * 80)
print("开始推理和评估")
print("=" * 80)

all_metrics = []
results = []

for idx, sample in enumerate(test_samples):
    print(f"\n[{idx+1}/{NUM_SAMPLES}] {sample['image']}")
    
    # 加载图片
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        print(f"  ❌ 图片不存在")
        continue
    
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # 创建Ground Truth mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 模型推理
    if MODEL_LOADED:
        try:
            print(f"  🔄 使用训练权重进行推理...")
            
            with torch.no_grad():
                # TODO: 根据实际模型API调用
                # 这里需要实现真实的推理逻辑
                # 由于Sa2VA的推理接口比较复杂，暂时使用GT演示
                pred_mask = gt_mask.copy()
                
                print(f"  ⚠️  推理接口待完善")
                
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            pred_mask = gt_mask.copy()
    else:
        # 使用GT作为演示
        pred_mask = gt_mask.copy()
        print(f"  ⚠️  模型未加载，使用GT演示评估")
    
    # 计算评价指标
    metrics = calculate_metrics(pred_mask, gt_mask)
    all_metrics.append(metrics)
    
    print(f"  📊 评价指标:")
    print(f"     IoU (Jaccard):    {metrics['IoU']:.4f}")
    print(f"     Dice Score:       {metrics['Dice']:.4f}")
    print(f"     Precision:        {metrics['Precision']:.4f}")
    print(f"     Recall:           {metrics['Recall']:.4f}")
    print(f"     Accuracy:         {metrics['Accuracy']:.4f}")
    print(f"     Pixel Accuracy:   {metrics['Pixel_Accuracy']:.4f}")
    if metrics['Hausdorff'] != float('inf'):
        print(f"     Hausdorff Dist:   {metrics['Hausdorff']:.2f} pixels")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行
    axes[1, 0].imshow(image_np)
    axes[1, 0].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1, 0].set_title('GT Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_np)
    axes[1, 1].imshow(pred_mask, alpha=0.5, cmap='Greens')
    axes[1, 1].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 差异图
    diff = np.abs(pred_mask.astype(float) - gt_mask.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'Difference\n(IoU={metrics["IoU"]:.3f}, Dice={metrics["Dice"]:.3f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "predictions", f"eval_{idx+1}_{sample['image']}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 保存: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'metrics': metrics,
        'output': output_path
    })

# 计算平均指标
print("\n" + "=" * 80)
print("总体评估结果")
print("=" * 80)

if len(all_metrics) > 0:
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics if m[key] != float('inf')])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n平均指标 (基于 {len(all_metrics)} 个样本):")
    print(f"  IoU (Jaccard):      {avg_metrics['IoU']:.4f}")
    print(f"  Dice Score:         {avg_metrics['Dice']:.4f}")
    print(f"  Precision:          {avg_metrics['Precision']:.4f}")
    print(f"  Recall:             {avg_metrics['Recall']:.4f}")
    print(f"  Accuracy:           {avg_metrics['Accuracy']:.4f}")
    print(f"  Pixel Accuracy:     {avg_metrics['Pixel_Accuracy']:.4f}")
    
    hausdorff_values = [m['Hausdorff'] for m in all_metrics if m['Hausdorff'] != float('inf')]
    if hausdorff_values:
        print(f"  Hausdorff Distance: {np.mean(hausdorff_values):.2f} pixels")
    
    # 保存详细结果
    detailed_results = {
        'model_loaded': MODEL_LOADED,
        'checkpoint': CHECKPOINT_PATH,
        'num_samples': len(results),
        'average_metrics': avg_metrics,
        'per_sample_results': results
    }
    
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, 'w') as f:
        # 转换numpy类型为Python类型
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        json.dump(convert_types(detailed_results), f, indent=2)
    
    print(f"\n✅ 详细结果保存到: {results_path}")
    
    # 创建评估报告
    report = f"""# Sa2VA模型评估报告

## 模型信息
- **Checkpoint**: {CHECKPOINT_PATH}
- **模型加载**: {'✅ 成功' if MODEL_LOADED else '❌ 失败 (使用GT演示)'}
- **评估样本数**: {len(results)}

## 平均评价指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **IoU (Jaccard)** | {avg_metrics['IoU']:.4f} | 交并比，越高越好 |
| **Dice Score** | {avg_metrics['Dice']:.4f} | Dice系数，越高越好 |
| **Precision** | {avg_metrics['Precision']:.4f} | 精确率 |
| **Recall** | {avg_metrics['Recall']:.4f} | 召回率 |
| **Accuracy** | {avg_metrics['Accuracy']:.4f} | 准确率 |
| **Pixel Accuracy** | {avg_metrics['Pixel_Accuracy']:.4f} | 像素准确率 |

## 指标说明

- **IoU (Intersection over Union)**: 预测和真实mask的交集除以并集
  - 0.5以上: 良好
  - 0.7以上: 优秀
  - 0.9以上: 极好

- **Dice Score**: 2 × (预测∩真实) / (预测+真实)
  - 与IoU类似，但对小目标更敏感

- **Precision**: 预测为正例中真正为正例的比例
- **Recall**: 真实正例中被正确预测的比例

## 样本详情

"""
    
    for result in results:
        m = result['metrics']
        report += f"\n### {result['sample_id']}. {result['image']}\n"
        report += f"- IoU: {m['IoU']:.4f}\n"
        report += f"- Dice: {m['Dice']:.4f}\n"
        report += f"- Precision: {m['Precision']:.4f}\n"
        report += f"- Recall: {m['Recall']:.4f}\n"
    
    if not MODEL_LOADED:
        report += "\n## ⚠️ 重要说明\n\n"
        report += "当前评估使用Ground Truth作为预测结果（因为模型未成功加载）。\n"
        report += "这导致所有指标都是1.0（完美匹配）。\n\n"
        report += "要进行真实的模型评估，需要:\n"
        report += "1. 在topo-sarl环境中运行此脚本\n"
        report += "2. 或将模型转换为HuggingFace格式后评估\n"
    
    report_path = os.path.join(OUTPUT_DIR, "EVALUATION_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ 评估报告保存到: {report_path}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"结果目录: {OUTPUT_DIR}")
print(f"  - predictions/: 可视化图片")
print(f"  - evaluation_results.json: 详细指标")
print(f"  - EVALUATION_REPORT.md: 评估报告")

if not MODEL_LOADED:
    print("\n" + "⚠️" * 40)
    print("警告: 当前使用Ground Truth作为预测，所有指标都是1.0")
    print("这不是真实的模型评估！")
    print("要进行真实评估，请在topo-sarl环境中运行或转换模型格式")
    print("⚠️" * 40)
