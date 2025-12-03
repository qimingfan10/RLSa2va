"""
Sa2VA 推理 + TTA (Test Time Augmentation)
对同一张图进行多次变换，预测后取平均
"""
import os
import sys
import json
import random
import numpy as np
import torch
from PIL import Image
import cv2
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/Sa2VA')

print("=" * 80)
print("Sa2VA + TTA (Test Time Augmentation)")
print("=" * 80)

# 配置
HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/evaluation_tta_results"
NUM_SAMPLES = 10

print(f"HF模型路径: {HF_MODEL_PATH}")
print(f"数据路径: {DATA_ROOT}")
print(f"评估样本数: {NUM_SAMPLES}")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        'Accuracy': float(accuracy),
    }


def predict_single(model, tokenizer, image):
    """单次预测"""
    text = "<image>Please segment the blood vessel."
    
    with torch.no_grad():
        result = model.predict_forward(
            image=image,
            text=text,
            tokenizer=tokenizer,
            processor=None,
        )
    
    prediction_text = result.get('prediction', '')
    
    if '[SEG]' in prediction_text and 'probability_maps' in result:
        prob_maps = result['probability_maps']
        if len(prob_maps) > 0:
            prob_map = prob_maps[0][0]
            if isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.cpu().numpy()
            return prob_map
    
    return None


def predict_with_tta(model, tokenizer, image):
    """
    TTA预测：对图像进行多种变换，预测后取平均
    变换：原图、水平翻转、垂直翻转、水平+垂直翻转
    """
    h, w = image.size[1], image.size[0]  # PIL: (w, h)
    
    # 收集所有概率图
    prob_maps = []
    
    # 1. 原图
    prob = predict_single(model, tokenizer, image)
    if prob is not None:
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_maps.append(prob)
    
    # 2. 水平翻转
    image_hflip = image.transpose(Image.FLIP_LEFT_RIGHT)
    prob = predict_single(model, tokenizer, image_hflip)
    if prob is not None:
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        # 翻转回来
        prob = np.fliplr(prob)
        prob_maps.append(prob)
    
    # 3. 垂直翻转
    image_vflip = image.transpose(Image.FLIP_TOP_BOTTOM)
    prob = predict_single(model, tokenizer, image_vflip)
    if prob is not None:
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        # 翻转回来
        prob = np.flipud(prob)
        prob_maps.append(prob)
    
    # 4. 水平+垂直翻转 (180度旋转)
    image_hvflip = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    prob = predict_single(model, tokenizer, image_hvflip)
    if prob is not None:
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        # 翻转回来
        prob = np.flipud(np.fliplr(prob))
        prob_maps.append(prob)
    
    if len(prob_maps) == 0:
        return None
    
    # 平均所有概率图
    avg_prob = np.mean(prob_maps, axis=0)
    
    # 二值化
    pred_mask = (avg_prob > 0.5).astype(np.uint8) * 255
    
    return pred_mask, len(prob_maps)


# 加载模型
print("📥 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
model.eval()
print("✅ Model loaded")

# 加载数据
ann_file = os.path.join(DATA_ROOT, "annotations.json")
with open(ann_file, 'r') as f:
    annotations = json.load(f)

# 筛选有效样本
valid_samples = []
for item in annotations:
    if 'image' in item and 'mask' in item and len(item['mask']) > 0:
        img_path = os.path.join(DATA_ROOT, "images", item['image'])
        if os.path.exists(img_path):
            valid_samples.append(item)

print(f"有效样本: {len(valid_samples)}")

# 随机选择样本
random.seed(42)
if len(valid_samples) > NUM_SAMPLES:
    test_samples = random.sample(valid_samples, NUM_SAMPLES)
else:
    test_samples = valid_samples

print(f"测试样本: {len(test_samples)}")
print()

# 评估
all_metrics = []
all_metrics_no_tta = []

for idx, sample in enumerate(test_samples):
    print(f"\n[{idx+1}/{len(test_samples)}] {sample['image']}")
    
    # 加载图像
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    
    # 创建GT mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 无TTA预测
    prob_no_tta = predict_single(model, tokenizer, image)
    if prob_no_tta is not None:
        if prob_no_tta.shape != (h, w):
            prob_no_tta = cv2.resize(prob_no_tta, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask_no_tta = (prob_no_tta > 0.5).astype(np.uint8) * 255
        metrics_no_tta = calculate_metrics(pred_mask_no_tta, gt_mask)
    else:
        pred_mask_no_tta = np.zeros((h, w), dtype=np.uint8)
        metrics_no_tta = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'Accuracy': 0}
    
    all_metrics_no_tta.append(metrics_no_tta)
    
    # TTA预测
    result = predict_with_tta(model, tokenizer, image)
    if result is not None:
        pred_mask_tta, num_augs = result
        metrics_tta = calculate_metrics(pred_mask_tta, gt_mask)
    else:
        pred_mask_tta = np.zeros((h, w), dtype=np.uint8)
        num_augs = 0
        metrics_tta = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'Accuracy': 0}
    
    all_metrics.append(metrics_tta)
    
    print(f"  无TTA: Dice={metrics_no_tta['Dice']:.4f}")
    print(f"  有TTA ({num_augs}x): Dice={metrics_tta['Dice']:.4f} (Δ={metrics_tta['Dice']-metrics_no_tta['Dice']:+.4f})")

# 汇总
print("\n" + "=" * 80)
print("📊 结果汇总")
print("=" * 80)

mean_no_tta = {k: np.mean([m[k] for m in all_metrics_no_tta]) for k in all_metrics_no_tta[0].keys()}
mean_tta = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

print(f"\n无TTA (Baseline):")
print(f"  IoU:       {mean_no_tta['IoU']:.4f}")
print(f"  Dice:      {mean_no_tta['Dice']:.4f}")
print(f"  Precision: {mean_no_tta['Precision']:.4f}")
print(f"  Recall:    {mean_no_tta['Recall']:.4f}")

print(f"\n有TTA (4x Augmentation):")
print(f"  IoU:       {mean_tta['IoU']:.4f} (Δ={mean_tta['IoU']-mean_no_tta['IoU']:+.4f})")
print(f"  Dice:      {mean_tta['Dice']:.4f} (Δ={mean_tta['Dice']-mean_no_tta['Dice']:+.4f})")
print(f"  Precision: {mean_tta['Precision']:.4f} (Δ={mean_tta['Precision']-mean_no_tta['Precision']:+.4f})")
print(f"  Recall:    {mean_tta['Recall']:.4f} (Δ={mean_tta['Recall']-mean_no_tta['Recall']:+.4f})")

print("\n" + "=" * 80)
