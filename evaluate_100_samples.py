"""
使用100张图片评估新旧模型，进行更充分的对比
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
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA')

print("=" * 80)
print("100张图片大规模对比评估：新模型 vs 旧模型")
print("=" * 80)

# 配置
OLD_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
NEW_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/comparison_100_samples"
NUM_SAMPLES = 100  # 增加到100张

print(f"旧模型: {OLD_MODEL_PATH}")
print(f"新模型: {NEW_MODEL_PATH}")
print(f"评估样本数: {NUM_SAMPLES}")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# 评价指标计算
def calculate_metrics(pred_mask, gt_mask):
    """计算分割评价指标"""
    pred_flat = (pred_mask > 127).flatten().astype(int)
    gt_flat = (gt_mask > 127).flatten().astype(int)
    
    if len(np.unique(gt_flat)) == 1 and len(np.unique(pred_flat)) == 1:
        if gt_flat[0] == pred_flat[0]:
            return {'IoU': 1.0, 'Dice': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'Accuracy': 1.0}
        else:
            return {'IoU': 0.0, 'Dice': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0}
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    return {
        'IoU': float(iou),
        'Dice': float(dice),
        'Precision': float(precision),
        'Recall': float(recall),
        'Accuracy': float(accuracy)
    }

def infer_with_model(model, tokenizer, image, text="<image>Please segment the blood vessel."):
    """使用模型推理"""
    try:
        result = model.predict_forward(
            image=image,
            text=text,
            tokenizer=tokenizer,
            processor=None,
        )
        
        prediction_text = result.get('prediction', '')
        
        if '[SEG]' in prediction_text and 'prediction_masks' in result:
            pred_masks = result['prediction_masks']
            if len(pred_masks) > 0:
                pred_mask = pred_masks[0][0]
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu().numpy()
                return pred_mask, True
        
        return None, False
    except Exception as e:
        print(f"推理错误: {e}")
        return None, False

# 加载数据集
print("=" * 80)
print("加载数据集")
print("=" * 80)

with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集总数: {len(dataset)}")

# 随机选择100张（使用固定种子保证可重复）
random.seed(42)
test_samples = random.sample(dataset, min(NUM_SAMPLES, len(dataset)))
print(f"已选择 {len(test_samples)} 张图片进行评估\n")

# ============================================================================
# 第1部分：评估旧模型
# ============================================================================
print("=" * 80)
print("第1部分：评估旧模型 (iter_12192)")
print("=" * 80)

print("\n加载旧模型...")
tokenizer_old = AutoTokenizer.from_pretrained(OLD_MODEL_PATH, trust_remote_code=True)
model_old = AutoModelForCausalLM.from_pretrained(
    OLD_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model_old.eval()
print("✅ 旧模型加载完成\n")

old_results = []
old_metrics = []

print("开始推理...")
for idx, sample in enumerate(tqdm(test_samples, desc="旧模型推理")):
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        continue
    
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # Ground Truth
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 推理
    pred_mask, success = infer_with_model(model_old, tokenizer_old, image)
    
    if success and pred_mask is not None:
        if pred_mask.shape != (h, w):
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if pred_mask.max() <= 1.0:
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        else:
            pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    else:
        pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    metrics = calculate_metrics(pred_mask, gt_mask)
    old_metrics.append(metrics)
    old_results.append({
        'image': sample['image'],
        'metrics': metrics,
        'success': success
    })

# 释放旧模型内存
del model_old
del tokenizer_old
torch.cuda.empty_cache()

print("\n✅ 旧模型评估完成")
old_avg = {k: np.mean([m[k] for m in old_metrics]) for k in old_metrics[0].keys()}
print(f"旧模型平均IoU: {old_avg['IoU']:.4f}, Dice: {old_avg['Dice']:.4f}")

# ============================================================================
# 第2部分：评估新模型
# ============================================================================
print("\n" + "=" * 80)
print("第2部分：评估新模型 (iter_3672)")
print("=" * 80)

print("\n加载新模型...")
tokenizer_new = AutoTokenizer.from_pretrained(NEW_MODEL_PATH, trust_remote_code=True)
model_new = AutoModelForCausalLM.from_pretrained(
    NEW_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model_new.eval()
print("✅ 新模型加载完成\n")

new_results = []
new_metrics = []

print("开始推理...")
for idx, sample in enumerate(tqdm(test_samples, desc="新模型推理")):
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        continue
    
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # Ground Truth
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 推理
    pred_mask, success = infer_with_model(model_new, tokenizer_new, image)
    
    if success and pred_mask is not None:
        if pred_mask.shape != (h, w):
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if pred_mask.max() <= 1.0:
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        else:
            pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    else:
        pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    metrics = calculate_metrics(pred_mask, gt_mask)
    new_metrics.append(metrics)
    new_results.append({
        'image': sample['image'],
        'metrics': metrics,
        'success': success
    })

del model_new
del tokenizer_new
torch.cuda.empty_cache()

print("\n✅ 新模型评估完成")
new_avg = {k: np.mean([m[k] for m in new_metrics]) for k in new_metrics[0].keys()}
print(f"新模型平均IoU: {new_avg['IoU']:.4f}, Dice: {new_avg['Dice']:.4f}")

# ============================================================================
# 第3部分：对比分析
# ============================================================================
print("\n" + "=" * 80)
print("第3部分：对比分析 (100张图片)")
print("=" * 80)

# 计算差异
differences = []
for i in range(len(old_results)):
    old_m = old_results[i]['metrics']
    new_m = new_results[i]['metrics']
    
    diff = {
        'image': old_results[i]['image'],
        'old_IoU': old_m['IoU'],
        'new_IoU': new_m['IoU'],
        'IoU_diff': new_m['IoU'] - old_m['IoU'],
        'old_Dice': old_m['Dice'],
        'new_Dice': new_m['Dice'],
        'Dice_diff': new_m['Dice'] - old_m['Dice']
    }
    differences.append(diff)

# 统计
iou_diffs = [d['IoU_diff'] for d in differences]
dice_diffs = [d['Dice_diff'] for d in differences]

print(f"\n平均指标对比:")
print(f"{'指标':<15} {'旧模型':<10} {'新模型':<10} {'差异':<10}")
print("-" * 50)
for key in ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']:
    old_val = old_avg[key]
    new_val = new_avg[key]
    diff = new_val - old_val
    print(f"{key:<15} {old_val:<10.4f} {new_val:<10.4f} {diff:+.4f}")

print(f"\nIoU差异统计:")
print(f"  平均差异: {np.mean(iou_diffs):+.4f}")
print(f"  标准差: {np.std(iou_diffs):.4f}")
print(f"  最小差异: {np.min(iou_diffs):+.4f}")
print(f"  最大差异: {np.max(iou_diffs):+.4f}")
print(f"  中位数差异: {np.median(iou_diffs):+.4f}")

print(f"\nDice差异统计:")
print(f"  平均差异: {np.mean(dice_diffs):+.4f}")
print(f"  标准差: {np.std(dice_diffs):.4f}")
print(f"  最小差异: {np.min(dice_diffs):+.4f}")
print(f"  最大差异: {np.max(dice_diffs):+.4f}")
print(f"  中位数差异: {np.median(dice_diffs):+.4f}")

# 显著性检验
from scipy import stats
t_stat, p_value = stats.ttest_rel([m['IoU'] for m in old_metrics], 
                                   [m['IoU'] for m in new_metrics])
print(f"\nIoU配对t检验:")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.4e}")
if p_value < 0.05:
    print(f"  ✅ 差异显著 (p < 0.05)")
else:
    print(f"  ❌ 差异不显著 (p >= 0.05)")

# 找出差异最大的样本
differences_sorted = sorted(differences, key=lambda x: abs(x['IoU_diff']), reverse=True)

print(f"\nIoU差异最大的10个样本:")
for i, d in enumerate(differences_sorted[:10], 1):
    print(f"  {i}. {d['image']}")
    print(f"     旧IoU: {d['old_IoU']:.4f}, 新IoU: {d['new_IoU']:.4f}, 差异: {d['IoU_diff']:+.4f}")

# 可视化差异最大的前5个样本
print(f"\n生成可视化...")
for i, d in enumerate(differences_sorted[:5], 1):
    sample = next(s for s in test_samples if s['image'] == d['image'])
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    
    if os.path.exists(img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.text(0.5, 0.5, f"{d['image']}\n旧IoU: {d['old_IoU']:.4f} → 新IoU: {d['new_IoU']:.4f}\n差异: {d['IoU_diff']:+.4f}",
                ha='center', va='center', fontsize=12, wrap=True)
        ax.axis('off')
        
        output_path = os.path.join(OUTPUT_DIR, "visualizations", f"diff_{i:02d}_{os.path.basename(sample['image'])}")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

# 保存完整结果
results = {
    'num_samples': len(test_samples),
    'old_model': {
        'path': OLD_MODEL_PATH,
        'avg_metrics': old_avg,
        'results': old_results
    },
    'new_model': {
        'path': NEW_MODEL_PATH,
        'avg_metrics': new_avg,
        'results': new_results
    },
    'comparison': {
        'metric_differences': {
            key: new_avg[key] - old_avg[key] 
            for key in old_avg.keys()
        },
        'iou_diff_stats': {
            'mean': float(np.mean(iou_diffs)),
            'std': float(np.std(iou_diffs)),
            'min': float(np.min(iou_diffs)),
            'max': float(np.max(iou_diffs)),
            'median': float(np.median(iou_diffs))
        },
        't_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        },
        'per_sample_differences': differences
    }
}

# 转换numpy类型为Python原生类型
def convert_to_json_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 递归转换
import json as json_module
results_json = json_module.loads(json_module.dumps(results, default=convert_to_json_serializable))

results_path = os.path.join(OUTPUT_DIR, "comparison_100_samples.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存到: {results_path}")

print("\n" + "=" * 80)
print("评估完成！")
print("=" * 80)
