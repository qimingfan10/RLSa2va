"""
正确的Sa2VA HuggingFace模型推理
按照官方demo.py的方式进行推理
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
print("正确的Sa2VA HuggingFace模型推理")
print("=" * 80)

# 配置
HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"  # 使用现有的HF模型
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/correct_hf_inference_results"
NUM_SAMPLES = 5

print(f"HF模型路径: {HF_MODEL_PATH}")
print(f"数据路径: {DATA_ROOT}")
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
            return {'IoU': 1.0, 'Dice': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'Accuracy': 1.0, 'Pixel_Accuracy': 1.0}
        else:
            return {'IoU': 0.0, 'Dice': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0, 'Pixel_Accuracy': 0.0}
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    pixel_acc = np.sum(pred_flat == gt_flat) / len(gt_flat)
    
    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Pixel_Accuracy': pixel_acc
    }

# 加载HuggingFace模型
print("=" * 80)
print("加载HuggingFace模型")
print("=" * 80)

if not os.path.exists(HF_MODEL_PATH):
    print(f"❌ HuggingFace模型不存在: {HF_MODEL_PATH}")
    print("请先运行转换脚本: bash convert_to_hf.sh")
    exit(1)

try:
    print("步骤1: 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载成功")
    
    print("\n步骤2: 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分配到多GPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    print(f"设备分配: {model.hf_device_map}")
    
    model.eval()
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    MODEL_LOADED = False

if not MODEL_LOADED:
    print("\n模型加载失败，退出程序")
    exit(1)

# 加载数据集
print("\n加载数据集...")
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"选中 {NUM_SAMPLES} 个样本进行推理")
print()

# 推理和评估
print("=" * 80)
print("开始正确的HuggingFace推理")
print("=" * 80)

all_metrics = []
results = []
successful_inferences = 0

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
            points = np.array(mask_coords).reshape(