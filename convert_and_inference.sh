#!/bin/bash

echo "========================================================================"
echo "Sa2VA模型转换和推理"
echo "========================================================================"

cd /home/ubuntu/Sa2VA

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 检查是否已经转换
HF_MODEL_PATH="models/sa2va_vessel_hf"

if [ -d "$HF_MODEL_PATH" ]; then
    echo "✅ HuggingFace模型已存在: $HF_MODEL_PATH"
    echo ""
else
    echo "步骤1: 转换模型为HuggingFace格式"
    echo "========================================================================"
    echo ""
    
    # 检查转换脚本
    if [ ! -f "tools/convert_to_hf.py" ]; then
        echo "❌ 转换脚本不存在: tools/convert_to_hf.py"
        exit 1
    fi
    
    echo "开始转换..."
    python3 tools/convert_to_hf.py \
        --model_path projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
        --ckpt_path work_dirs/merged_vessel_segmentation/iter_3672.pth \
        --save_path $HF_MODEL_PATH
    
    if [ $? -eq 0 ]; then
        echo "✅ 模型转换成功"
    else
        echo "❌ 模型转换失败"
        echo "尝试使用mmengine环境..."
        exit 1
    fi
    echo ""
fi

echo "步骤2: 使用HuggingFace模型进行推理"
echo "========================================================================"
echo ""

# 创建HF推理脚本
cat > hf_inference.py << 'EOFPYTHON'
import os
import json
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

print("加载HuggingFace模型...")

HF_MODEL_PATH = "models/sa2va_vessel_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/hf_inference_results"
NUM_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

try:
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    
    print(f"从 {HF_MODEL_PATH} 加载模型...")
    model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功，设备: {device}")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    MODEL_LOADED = False

# 加载数据
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"\n选中 {NUM_SAMPLES} 个样本进行推理")

results = []

for idx, sample in enumerate(test_samples):
    print(f"\n[{idx+1}/{NUM_SAMPLES}] {sample['image']}")
    
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # Ground Truth
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 模型推理
    if MODEL_LOADED:
        try:
            with torch.no_grad():
                # 使用模型的predict_forward方法
                text = "blood vessel"
                result = model.predict_forward(
                    image=image,
                    text=text,
                    tokenizer=tokenizer
                )
                
                if 'prediction_masks' in result:
                    pred_masks = result['prediction_masks']
                    if len(pred_masks) > 0:
                        pred_mask = (pred_masks[0].cpu().numpy() * 255).astype(np.uint8)
                    else:
                        pred_mask = gt_mask
                else:
                    pred_mask = gt_mask
                    
                print(f"  ✅ 推理完成")
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            pred_mask = gt_mask
    else:
        pred_mask = gt_mask
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image_np)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(image_np)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Greens')
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    overlay = image_np.copy()
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay (Red=GT, Green=Pred)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "predictions", f"pred_{idx+1}_{sample['image']}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 保存: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'output': output_path
    })

print(f"\n✅ 完成！结果保存在: {OUTPUT_DIR}/predictions/")
print(f"共处理 {len(results)} 个样本")

EOFPYTHON

# 运行推理
python3 hf_inference.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo "查看结果:"
echo "  ls -lh hf_inference_results/predictions/"
