#!/usr/bin/env python3
"""
Evaluate the LoRA fine-tuned Sa2VA model on vessel segmentation
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def dice_score(pred, target):
    """Calculate Dice score"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target):
    """Calculate IoU score"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)

def main():
    # Model path - 8B模型
    model_path = "/home/ubuntu/Sa2VA/work_dirs/sa2va_vessel_lora_finetune_8b_extreme/iter_15280_hf"
    
    # Dataset path
    data_dir = "/home/ubuntu/Sa2VA/Segment_DATA_Merged_512"
    annotations_file = os.path.join(data_dir, "annotations.json")
    
    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Evaluate on a subset
    num_samples = min(100, len(annotations))  # Test on 100 samples
    print(f"Evaluating on {num_samples} samples...")
    
    dice_scores = []
    iou_scores = []
    
    for i, ann in enumerate(tqdm(annotations[:num_samples], desc="Evaluating")):
        try:
            # Load image
            img_name = ann['image']
            img_path = os.path.join(data_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, "images", img_name)
            if not os.path.exists(img_path):
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            # Get mask path (derive from image name)
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(data_dir, "masks", f"{base_name}_mask.png")
            if not os.path.exists(mask_path):
                # Try alternative naming
                mask_path = os.path.join(data_dir, "masks", f"{base_name}.png")
            if not os.path.exists(mask_path):
                continue
            
            gt_mask = np.array(Image.open(mask_path).convert('L')) > 127
            
            # Run inference
            text = "<image>\nPlease segment the blood vessel."
            with torch.no_grad():
                result = model.predict_forward(
                    image=image,
                    text=text,
                    tokenizer=tokenizer,
                )
            
            # Get prediction mask (returns 'prediction_masks' list)
            pred_masks = result.get('prediction_masks', [])
            if not pred_masks or len(pred_masks) == 0:
                print(f"No mask predicted for sample {i}, result keys: {result.keys()}")
                continue
            
            pred_mask = pred_masks[0]  # First mask
            if pred_mask is None:
                continue
            
            # Handle multi-frame output
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]  # Take first frame
            
            # Convert to numpy if needed
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.cpu().numpy()
            
            # Resize if needed
            if pred_mask.shape != gt_mask.shape:
                from PIL import Image as PILImage
                pred_mask = np.array(PILImage.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST)) > 127
            
            pred_mask = pred_mask.astype(float)
            gt_mask = gt_mask.astype(float)
            
            # Calculate metrics
            dice = dice_score(pred_mask, gt_mask)
            iou = iou_score(pred_mask, gt_mask)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Samples evaluated: {len(dice_scores)}")
    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Min Dice: {np.min(dice_scores):.4f}, Max Dice: {np.max(dice_scores):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
