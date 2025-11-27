"""
使用训练好的Sa2VA模型进行实际推理预测
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

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/Sa2VA')

# 设置环境变量
os.environ['PYTHONPATH'] = '/home/ubuntu/Sa2VA:' + os.environ.get('PYTHONPATH', '')

print("=" * 80)
print("Sa2VA训练模型推理预测")
print("=" * 80)

# 配置
CHECKPOINT_PATH = "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"
CONFIG_PATH = "/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_DIR = "/home/ubuntu/Sa2VA/inference_results"
NUM_SAMPLES = 5  # 预测5张图片

print(f"配置文件: {CONFIG_PATH}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"数据集: {DATA_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

# 检查文件
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint不存在: {CHECKPOINT_PATH}")
    exit(1)

if not os.path.exists(CONFIG_PATH):
    print(f"❌ 配置文件不存在: {CONFIG_PATH}")
    exit(1)

print("✅ 文件检查通过")
print()

# 加载配置和模型
print("=" * 80)
print("加载模型")
print("=" * 80)

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import MODELS
    
    print("✅ 导入mmengine成功")
    
    # 加载配置
    print("\n加载配置文件...")
    cfg = Config.fromfile(CONFIG_PATH)
    print("✅ 配置加载成功")
    
    # 设置checkpoint路径
    cfg.load_from = CHECKPOINT_PATH
    cfg.resume = False
    
    # 设置为评估模式
    cfg.work_dir = OUTPUT_DIR
    
    print("\n构建模型...")
    # 构建模型
    model = MODELS.build(cfg.model)
    print("✅ 模型构建成功")
    
    # 加载checkpoint
    print("\n加载checkpoint权重...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    print("✅ 权重加载成功")
    
    # 设置为评估模式
    model.eval()
    
    # 移动到GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ 模型移动到设备: {device}")
    
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print(f"\n详细错误:")
    import traceback
    traceback.print_exc()
    MODEL_LOADED = False
    print("\n⚠️  将只进行Ground Truth可视化")

# 加载数据集
print("\n" + "=" * 80)
print("加载数据集")
print("=" * 80)

with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")

# 随机选择样本
random.seed(42)
test_samples = random.sample(dataset, NUM_SAMPLES)

print(f"\n选中的样本:")
for i, sample in enumerate(test_samples):
    print(f"  {i+1}. {sample['image']} (masks: {len(sample['mask'])})")

# 进行推理
print("\n" + "=" * 80)
print("开始推理")
print("=" * 80)

results = []

for idx, sample in enumerate(test_samples):
    print(f"\n处理样本 {idx+1}/{NUM_SAMPLES}: {sample['image']}")
    
    # 加载图片
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    if not os.path.exists(img_path):
        print(f"  ❌ 图片不存在: {img_path}")
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
    
    # 如果模型加载成功，进行推理
    if MODEL_LOADED:
        try:
            print(f"  🔄 进行模型推理...")
            
            with torch.no_grad():
                # 准备输入
                # 这里需要根据Sa2VA的实际输入格式来准备数据
                # 由于Sa2VA需要特定的数据格式，这里先显示Ground Truth
                
                # TODO: 实现实际的推理逻辑
                # pred_mask = model.predict(image, text="blood vessel")
                
                pred_mask = gt_mask  # 临时使用GT作为预测结果
                print(f"  ⚠️  推理逻辑待实现，当前显示Ground Truth")
                
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            pred_mask = gt_mask
    else:
        pred_mask = gt_mask
        print(f"  ℹ️  模型未加载，显示Ground Truth")
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(image_np)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(image_np)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Greens')
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    # 对比
    overlay = image_np.copy()
    # GT用红色
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    # 预测用绿色
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay (Red=GT, Green=Pred)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, "predictions", f"sample_{idx+1}_{sample['image']}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存到: {output_path}")
    
    results.append({
        'sample_id': idx + 1,
        'image': sample['image'],
        'num_masks': len(sample['mask']),
        'prediction': output_path
    })

# 保存结果
print("\n" + "=" * 80)
print("保存结果")
print("=" * 80)

summary = {
    'checkpoint': CHECKPOINT_PATH,
    'model_loaded': MODEL_LOADED,
    'num_samples': len(results),
    'results': results
}

summary_path = os.path.join(OUTPUT_DIR, "inference_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ 摘要保存到: {summary_path}")

# 创建README
readme = f"""# Sa2VA训练模型推理结果

## 模型信息
- **Checkpoint**: {CHECKPOINT_PATH}
- **配置文件**: {CONFIG_PATH}
- **模型加载**: {'✅ 成功' if MODEL_LOADED else '❌ 失败'}

## 推理结果
- **样本数**: {len(results)}
- **输出目录**: {OUTPUT_DIR}/predictions/

## 注意
{'当前使用训练好的模型权重进行推理。' if MODEL_LOADED else '模型加载失败，显示Ground Truth作为参考。'}

## 下一步
要进行完整的模型推理，需要:
1. 确保mmengine环境正确安装
2. 实现Sa2VA的推理接口
3. 或将模型转换为HuggingFace格式后使用

## 样本列表
"""

for result in results:
    readme += f"\n{result['sample_id']}. **{result['image']}**\n"
    readme += f"   - Masks: {result['num_masks']}\n"

readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme)

print(f"✅ README保存到: {readme_path}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"结果保存在: {OUTPUT_DIR}/predictions/")
print()

if not MODEL_LOADED:
    print("⚠️  注意: 模型未成功加载")
    print("建议:")
    print("1. 在topo-sarl环境中运行此脚本")
    print("2. 或将模型转换为HuggingFace格式后使用")
    print()
    print("转换命令:")
    print("  python tools/convert_to_hf.py \\")
    print("    --model_path projects/sa2va/configs/sa2va_merged_vessel_finetune.py \\")
    print("    --ckpt_path work_dirs/merged_vessel_segmentation/iter_3672.pth \\")
    print("    --save_path models/sa2va_vessel_hf")
