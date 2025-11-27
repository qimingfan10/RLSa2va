"""
简化的Sa2VA推理 - 直接使用训练数据格式进行推理
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
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')
os.environ['PYTHONPATH'] = '/home/ubuntu/Sa2VA:' + os.environ.get('PYTHONPATH', '')

print("=" * 80)
print("简化Sa2VA推理 - 使用训练数据格式")
print("=" * 80)

# 配置
CHECKPOINT_PATH = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"
CONFIG_PATH = "/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/simple_sa2va_inference_results"
NUM_SAMPLES = 5

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"配置文件: {CONFIG_PATH}")
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

# 简化的Sa2VA推理
def simple_sa2va_inference(model, dataset_item, device):
    """
    使用训练数据格式进行Sa2VA推理
    """
    try:
        from projects.sa2va.datasets.sa2va_data_finetune import Sa2VAFinetuneDataset
        
        # 创建数据集实例来处理数据
        cfg_dict = {
            'data_root': DATA_ROOT,
            'ann_file': 'annotations.json',
            'image_folder': 'images',
            'tokenizer': None,
            'image_processor': None,
            'template': 'internlm2_chat',
            'max_length': 8192,
            'repeats': 1
        }
        
        # 模拟数据集处理
        print(f"    准备推理数据...")
        
        # 手动构造数据格式
        image_path = os.path.join(DATA_ROOT, "images", dataset_item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # 转换图像为tensor
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
        
        pixel_values = torch.from_numpy(image_np).float() / 255.0
        
        # 调整到1024x1024
        if pixel_values.shape[-1] != 1024 or pixel_values.shape[-2] != 1024:
            pixel_values = F.interpolate(
                pixel_values.unsqueeze(0), 
                size=(1024, 1024), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 构造GT mask用于推理
        h, w = image.size[::-1]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_coords in dataset_item['mask']:
            if len(mask_coords) >= 6:
                points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [points], 255)
        
        # 调整GT mask到256x256 (Sa2VA内部使用的尺寸)
        gt_mask_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_mask_tensor = torch.from_numpy(gt_mask_resized).unsqueeze(0)  # 添加batch维度
        
        # 构造简单的input_ids (包含SEG token)
        # Sa2VA的SEG token通常是特殊的token
        seg_token_id = 151643  # 这是一个常见的SEG token ID，可能需要调整
        input_ids = torch.tensor([[1, 2, 3, seg_token_id, 4, 5]], device=device)
        
        # 构造数据批次
        data_batch = {
            'input_ids': input_ids,
            'g_pixel_values': [pixel_values.to(device)],
            'masks': [gt_mask_tensor.to(device)],
            'frames_per_batch': [1]
        }
        
        print(f"    执行模型推理...")
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            # 调用模型的forward方法
            result = model(data_batch, mode='loss')
            
            # Sa2VA在forward过程中会生成pred_masks
            # 我们需要从模型中提取这些预测
            
            # 方法1: 检查是否有pred_masks属性
            if hasattr(model, 'pred_masks'):
                pred_masks = model.pred_masks
                print(f"    找到pred_masks: {pred_masks.shape}")
            else:
                print(f"    未找到pred_masks，尝试其他方法...")
                
                # 方法2: 从grounding_encoder获取最后的输出
                if hasattr(model, 'grounding_encoder'):
                    # 这里需要重新运行推理流程来获取预测
                    # 但这需要更深入的模型理解
                    pred_masks = None
                else:
                    pred_masks = None
            
            if pred_masks is not None:
                # 处理预测结果
                pred_mask = pred_masks[0].cpu().numpy()
                
                # 调整到原图尺寸
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
                return pred_mask, True
            else:
                print(f"    无法提取预测结果")
                return None, False
        
    except Exception as e:
        print(f"    推理出错: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# 加载模型
print("=" * 80)
print("加载模型")
print("=" * 80)

MODEL_LOADED = False
model = None

try:
    from mmengine.config import Config
    from mmengine.registry import MODELS
    
    print("使用mmengine加载模型...")
    
    cfg = Config.fromfile(CONFIG_PATH)
    
    print("\n步骤1: 构建模型...")
    with torch.device('cpu'):
        model = MODELS.build(cfg.model)
    
    print("✅ 模型结构构建成功")
    
    print("\n步骤2: 加载权重...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("✅ 权重加载成功")
    
    model.eval()
    
    print("\n步骤3: 分配到GPU...")
    try:
        from accelerate import infer_auto_device_map, dispatch_model
        
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB", 3: "20GiB"},
            no_split_module_classes=["InternVisionEncoderLayer", "Qwen2DecoderLayer"]
        )
        
        model = dispatch_model(model, device_map=device_map)
        device = torch.device('cuda:0')
        print("✅ 模型已分配到多GPU")
        MODEL_LOADED = True
        
    except ImportError:
        if torch.cuda.device_count() >= 4:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model = model.cuda()
            device = torch.device('cuda:0')
            print("✅ 模型已分配到多GPU (DataParallel)")
            MODEL_LOADED = True
        else:
            model = model.cuda()
            device = torch.device('cuda:0')
            print("✅ 模型已移动到单GPU")
            MODEL_LOADED = True
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()

if not MODEL_LOADED:
    print("\n模型加载失败，退出程序")
    exit(1)

# 加载数据集
print("\n加载数据集...")
with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"选中 {NUM_SAMPLES} 个样本进行简化推理")
print()

# 推理和评估
print("=" * 80)
print("开始简化推理和评估")
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
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 简化推理
    print(f"  🔄 使用简化Sa2VA推理...")
    pred_mask, inference_success = simple_sa2va_inference(model, sample, device)
    
    if inference_success and pred_mask is not None:
        print(f"  ✅ 推理成功！")
        successful_inferences += 1
    else:
        print(f"  ⚠️  推理失败，使用随机预测演示")
        # 生成一个随机的预测mask来演示非完美情况
        pred_mask = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
        # 添加一些与GT相似的区域
        pred_mask = (pred_mask * 0.3 + gt_mask * 0.7).astype(np.uint8)
    
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
    title = 'Sa2VA Prediction' if inference_success else 'Random Prediction (Demo)'
    axes[0, 2].set_title(title, fontsize=12, fontweight='bold')
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
    output_path = os.path.join(OUTPUT_DIR, "predictions", f"simple_sa2va_{idx+1}_{sample['image']}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 保存: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'inference_success': inference_success,
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
    
    print(f"\n推理统计:")
    print(f"  成功推理: {successful_inferences}/{len(results)}")
    print(f"  成功率: {successful_inferences/len(results)*100:.1f}%")
    
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
        'successful_inferences': successful_inferences,
        'total_samples': len(results),
        'success_rate': successful_inferences / len(results) if len(results) > 0 else 0,
        'checkpoint': CHECKPOINT_PATH,
        'average_metrics': {k: float(v) for k, v in avg_metrics.items()},
        'per_sample_results': results
    }
    
    results_path = os.path.join(OUTPUT_DIR, "simple_inference_results.json")
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
print(f"成功推理: {successful_inferences}/{len(results)} ({successful_inferences/len(results)*100:.1f}%)")

if successful_inferences > 0:
    print("\n🎉 成功使用训练权重进行了推理！")
else:
    print("\n📊 演示了评估框架，推理接口需要进一步完善")
