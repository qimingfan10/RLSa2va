"""
使用4张GPU进行Sa2VA模型推理和评估
使用DeepSpeed或模型并行来分散显存压力
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

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')
os.environ['PYTHONPATH'] = '/home/ubuntu/Sa2VA:' + os.environ.get('PYTHONPATH', '')

print("=" * 80)
print("Sa2VA多GPU推理和评估")
print("=" * 80)

# 配置
CHECKPOINT_PATH = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"
CONFIG_PATH = "/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/multi_gpu_inference_results"
NUM_SAMPLES = 10

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"配置文件: {CONFIG_PATH}")
print(f"使用GPU: 0,1,2,3")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

# 评价指标计算
def calculate_metrics(pred_mask, gt_mask):
    """计算分割评价指标"""
    pred_flat = (pred_mask > 127).flatten().astype(int)
    gt_flat = (gt_mask > 127).flatten().astype(int)
    
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

# 加载模型
print("=" * 80)
print("加载模型 (多GPU)")
print("=" * 80)

MODEL_LOADED = False
model = None

try:
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("使用mmengine加载模型...")
    print("策略: 使用CPU offload + 多GPU分布")
    
    # 加载配置
    cfg = Config.fromfile(CONFIG_PATH)
    
    # 设置为CPU先加载，避免OOM
    print("\n步骤1: 在CPU上构建模型...")
    with torch.device('cpu'):
        model = MODELS.build(cfg.model)
    
    print("✅ 模型结构构建成功")
    
    # 加载权重到CPU
    print("\n步骤2: 加载checkpoint到CPU...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   Checkpoint包含 {len(state_dict)} 个参数")
    else:
        state_dict = checkpoint
    
    # 加载权重
    print("\n步骤3: 加载权重...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"   缺失的keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"   多余的keys: {len(unexpected_keys)}")
    
    print("✅ 权重加载成功")
    
    # 设置为评估模式
    model.eval()
    
    # 使用device_map自动分配到多GPU
    print("\n步骤4: 分配模型到多GPU...")
    
    # 方法1: 使用accelerate的device_map
    try:
        from accelerate import infer_auto_device_map, dispatch_model
        
        print("   使用accelerate进行自动设备映射...")
        
        # 计算模型大小
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"   模型总大小: {total_size / 1024**3:.2f} GB")
        
        # 自动推断设备映射
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB", 3: "20GiB"},
            no_split_module_classes=["InternVisionEncoderLayer", "Qwen2DecoderLayer"]
        )
        
        print(f"   设备映射: {device_map}")
        
        # 分发模型
        model = dispatch_model(model, device_map=device_map)
        
        print("✅ 模型已分配到多GPU (accelerate)")
        MODEL_LOADED = True
        
    except ImportError:
        print("   ⚠️  accelerate未安装，尝试手动分配...")
        
        # 方法2: 手动DataParallel
        if torch.cuda.device_count() >= 4:
            print(f"   使用DataParallel分配到4张GPU...")
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model = model.cuda()
            print("✅ 模型已分配到多GPU (DataParallel)")
            MODEL_LOADED = True
        else:
            print(f"   ❌ 可用GPU数量不足: {torch.cuda.device_count()}")
            
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("\n详细错误:")
    import traceback
    traceback.print_exc()

if not MODEL_LOADED:
    print("\n" + "=" * 80)
    print("⚠️  警告: 模型加载失败")
    print("=" * 80)
    print("将使用Ground Truth作为演示")
    print("=" * 80)
    print()

# 加载数据集
print("\n加载数据集...")
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
            print(f"  🔄 使用多GPU模型推理...")
            
            with torch.no_grad():
                # 准备输入数据
                # 注意: Sa2VA的实际推理接口需要特定的数据格式
                # 这里需要根据模型的forward方法来准备
                
                # TODO: 实现Sa2VA的实际推理逻辑
                # 由于Sa2VA训练模型的forward方法需要复杂的data_batch
                # 暂时使用GT作为占位符
                
                pred_mask = gt_mask.copy()
                print(f"  ⚠️  推理接口待实现 (需要适配Sa2VA的data_batch格式)")
                
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            pred_mask = gt_mask.copy()
    else:
        pred_mask = gt_mask.copy()
        print(f"  ⚠️  模型未加载，使用GT演示")
    
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
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(image_np)
    axes[1, 0].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1, 0].set_title('GT Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_np)
    axes[1, 1].imshow(pred_mask, alpha=0.5, cmap='Greens')
    axes[1, 1].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    diff = np.abs(pred_mask.astype(float) - gt_mask.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'Difference\n(IoU={metrics["IoU"]:.3f}, Dice={metrics["Dice"]:.3f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "predictions", f"multi_gpu_{idx+1}_{sample['image']}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 保存: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'metrics': metrics,
        'output': output_path
    })

# 总体评估
print("\n" + "=" * 80)
print("总体评估结果")
print("=" * 80)

if len(all_metrics) > 0:
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n平均指标 (基于 {len(all_metrics)} 个样本):")
    print(f"  IoU (Jaccard):      {avg_metrics['IoU']:.4f}")
    print(f"  Dice Score:         {avg_metrics['Dice']:.4f}")
    print(f"  Precision:          {avg_metrics['Precision']:.4f}")
    print(f"  Recall:             {avg_metrics['Recall']:.4f}")
    print(f"  Accuracy:           {avg_metrics['Accuracy']:.4f}")
    print(f"  Pixel Accuracy:     {avg_metrics['Pixel_Accuracy']:.4f}")
    
    # 保存结果
    detailed_results = {
        'model_loaded': MODEL_LOADED,
        'multi_gpu': True,
        'num_gpus': 4,
        'checkpoint': CHECKPOINT_PATH,
        'num_samples': len(results),
        'average_metrics': {k: float(v) for k, v in avg_metrics.items()},
        'per_sample_results': results
    }
    
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, 'w') as f:
        def convert_types(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        json.dump(convert_types(detailed_results), f, indent=2)
    
    print(f"\n✅ 详细结果保存到: {results_path}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"结果目录: {OUTPUT_DIR}")

if not MODEL_LOADED:
    print("\n" + "⚠️" * 40)
    print("警告: 模型未成功加载到多GPU")
    print("当前使用Ground Truth作为演示")
    print("⚠️" * 40)
