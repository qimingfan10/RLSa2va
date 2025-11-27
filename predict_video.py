"""
Sa2VA视频预测 - 使用官方predict_forward方法
对视频序列进行预测并生成对比MP4视频
"""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import cv2
from collections import defaultdict
from sklearn.metrics import jaccard_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/Sa2VA')

print("=" * 80)
print("Sa2VA视频预测 - 使用官方predict_forward方法")
print("=" * 80)

# 配置
HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_BASE_DIR = "/home/ubuntu/Sa2VA/video_prediction_results"
NUM_VIDEOS = 5  # 预测前5个视频序列
START_VIDEO_INDEX = 0  # 从第几个视频开始（0-based）

print(f"HF模型路径: {HF_MODEL_PATH}")
print(f"数据路径: {DATA_ROOT}")
print(f"预测视频数量: {NUM_VIDEOS}")
print(f"起始视频索引: {START_VIDEO_INDEX}")
print()

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 评价指标计算
def calculate_metrics(pred_mask, gt_mask):
    """计算分割评价指标"""
    pred_flat = (pred_mask > 127).flatten().astype(int)
    gt_flat = (gt_mask > 127).flatten().astype(int)
    
    if len(np.unique(gt_flat)) == 1 and len(np.unique(pred_flat)) == 1:
        if gt_flat[0] == pred_flat[0]:
            return {'IoU': 1.0, 'Dice': 1.0}
        else:
            return {'IoU': 0.0, 'Dice': 0.0}
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    
    return {'IoU': float(iou), 'Dice': float(dice)}

# 步骤1: 分析数据集，找出视频序列
print("=" * 80)
print("步骤1: 分析数据集")
print("=" * 80)

with open(os.path.join(DATA_ROOT, "annotations.json")) as f:
    dataset = json.load(f)

print(f"数据集总数: {len(dataset)}")

# 按前缀分组
video_groups = defaultdict(list)
for item in dataset:
    img_name = item['image']
    if '_frame_' in img_name:
        prefix = img_name.split('_frame_')[0]
        frame_num = img_name.split('_frame_')[1].split('.')[0]
        video_groups[prefix].append((frame_num, item))
    else:
        video_groups[img_name].append(('000000', item))

# 筛选视频序列（至少3帧）
video_sequences = []
for prefix, frames in sorted(video_groups.items()):
    if len(frames) >= 3:
        sorted_frames = sorted(frames, key=lambda x: x[0])
        video_sequences.append((prefix, sorted_frames))

print(f"\n找到 {len(video_sequences)} 个视频序列（>=3帧）")

# 显示所有视频
print("\n可用的视频序列:")
for i, (prefix, frames) in enumerate(video_sequences[:20]):
    print(f"  {i}. {prefix}: {len(frames)}帧 (帧{frames[0][0]}-{frames[-1][0]})")
if len(video_sequences) > 20:
    print(f"  ... 还有 {len(video_sequences) - 20} 个视频")

# 选择要预测的视频范围
end_index = min(START_VIDEO_INDEX + NUM_VIDEOS, len(video_sequences))
selected_videos = video_sequences[START_VIDEO_INDEX:end_index]

print(f"\n✅ 将预测 {len(selected_videos)} 个视频 (索引 {START_VIDEO_INDEX} 到 {end_index-1})")
for i, (prefix, frames) in enumerate(selected_videos):
    print(f"  #{START_VIDEO_INDEX + i}. {prefix}: {len(frames)}帧")
print()

# 步骤2: 加载HuggingFace模型
print("=" * 80)
print("步骤2: 加载HuggingFace模型")
print("=" * 80)

try:
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载成功")
    
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    
    model.eval()
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 步骤3: 对视频序列进行推理
print("\n" + "=" * 80)
print("步骤3: 视频序列推理")
print("=" * 80)

all_metrics = []
frame_results = []
successful_frames = 0

for idx, (frame_num, sample) in enumerate(tqdm(selected_frames, desc="推理进度")):
    img_path = os.path.join(DATA_ROOT, "images", sample['image'])
    
    if not os.path.exists(img_path):
        print(f"⚠️  跳过不存在的图片: {sample['image']}")
        continue
    
    # 加载图片
    image = Image.open(img_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # 创建Ground Truth mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_coords in sample['mask']:
        if len(mask_coords) >= 6:
            points = np.array(mask_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [points], 255)
    
    # 使用predict_forward进行推理
    try:
        text = "<image>Please segment the blood vessel."
        
        result = model.predict_forward(
            image=image,
            text=text,
            tokenizer=tokenizer,
            processor=None,
        )
        
        prediction_text = result.get('prediction', '')
        
        # 提取预测mask
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
                
                successful_frames += 1
            else:
                pred_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    except Exception as e:
        print(f"\n⚠️  帧 {frame_num} 推理失败: {e}")
        pred_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 计算指标
    metrics = calculate_metrics(pred_mask, gt_mask)
    all_metrics.append(metrics)
    
    # 保存结果
    frame_results.append({
        'frame_num': frame_num,
        'image': sample['image'],
        'image_np': image_np,
        'gt_mask': gt_mask,
        'pred_mask': pred_mask,
        'metrics': metrics
    })

print(f"\n✅ 推理完成!")
print(f"   成功: {successful_frames}/{len(selected_frames)}")
print(f"   成功率: {successful_frames/len(selected_frames)*100:.1f}%")

# 计算平均指标
avg_metrics = {
    key: np.mean([m[key] for m in all_metrics])
    for key in all_metrics[0].keys()
}

print(f"\n视频平均指标:")
print(f"   IoU (Jaccard): {avg_metrics['IoU']:.4f}")
print(f"   Dice Score:    {avg_metrics['Dice']:.4f}")

# 步骤4: 生成对比视频
print("\n" + "=" * 80)
print("步骤4: 生成对比MP4视频")
print("=" * 80)

# 确定视频参数
h, w = frame_results[0]['image_np'].shape[:2]
fps = 10  # 10 FPS

# 创建三种视频：原图+GT、原图+预测、GT vs 预测对比
video_configs = [
    {
        'name': 'original_with_gt.mp4',
        'description': '原图 + Ground Truth叠加',
        'function': lambda frame: create_overlay(frame['image_np'], frame['gt_mask'], (0, 0, 255))
    },
    {
        'name': 'original_with_pred.mp4',
        'description': '原图 + Sa2VA预测叠加',
        'function': lambda frame: create_overlay(frame['image_np'], frame['pred_mask'], (0, 255, 0))
    },
    {
        'name': 'comparison.mp4',
        'description': 'GT vs 预测对比',
        'function': lambda frame: create_comparison(frame)
    }
]

def create_overlay(image, mask, color):
    """创建mask叠加图"""
    result = image.copy()
    mask_colored = np.zeros_like(result)
    mask_colored[mask > 127] = color
    result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
    return result

def create_comparison(frame):
    """创建GT vs 预测对比图"""
    image = frame['image_np']
    gt_mask = frame['gt_mask']
    pred_mask = frame['pred_mask']
    metrics = frame['metrics']
    
    # 创建2x2网格
    # 左上: 原图, 右上: GT
    # 左下: 预测, 右下: GT vs 预测叠加
    
    h, w = image.shape[:2]
    
    # 原图
    img1 = image.copy()
    cv2.putText(img1, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # GT mask (彩色)
    img2 = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(img2, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 预测mask (彩色)
    img3 = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(img3, 'Sa2VA Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # GT vs 预测叠加
    img4 = image.copy()
    # GT用红色
    gt_overlay = np.zeros_like(img4)
    gt_overlay[gt_mask > 127] = [0, 0, 255]
    # 预测用绿色
    pred_overlay = np.zeros_like(img4)
    pred_overlay[pred_mask > 127] = [0, 255, 0]
    
    img4 = cv2.addWeighted(img4, 0.5, gt_overlay, 0.3, 0)
    img4 = cv2.addWeighted(img4, 1.0, pred_overlay, 0.3, 0)
    
    # 添加指标文本
    metric_text = f"IoU:{metrics['IoU']:.3f} Dice:{metrics['Dice']:.3f}"
    cv2.putText(img4, 'GT(Red) vs Pred(Green)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img4, metric_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 拼接成2x2网格
    top_row = np.hstack([img1, img2])
    bottom_row = np.hstack([img3, img4])
    result = np.vstack([top_row, bottom_row])
    
    return result

# 生成视频
for video_config in video_configs:
    video_name = video_config['name']
    video_path = os.path.join(OUTPUT_DIR, video_name)
    description = video_config['description']
    func = video_config['function']
    
    print(f"\n生成视频: {description}")
    print(f"   文件: {video_name}")
    
    # 获取第一帧的尺寸
    first_frame = func(frame_results[0])
    video_h, video_w = first_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (video_w, video_h))
    
    # 写入所有帧
    for frame in tqdm(frame_results, desc=f"  生成{video_name}"):
        frame_img = func(frame)
        video_writer.write(frame_img)
    
    video_writer.release()
    
    # 检查文件大小
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"   ✅ 完成! 大小: {file_size:.2f}MB")

# 步骤5: 生成报告
print("\n" + "=" * 80)
print("步骤5: 生成评估报告")
print("=" * 80)

report_path = os.path.join(OUTPUT_DIR, "video_evaluation_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# Sa2VA视频预测评估报告\n\n")
    f.write(f"## 视频信息\n\n")
    f.write(f"- **视频ID**: {selected_prefix}\n")
    f.write(f"- **总帧数**: {len(selected_frames)}\n")
    f.write(f"- **帧范围**: {selected_frames[0][0]} - {selected_frames[-1][0]}\n")
    f.write(f"- **分辨率**: {w}x{h}\n")
    f.write(f"- **帧率**: {fps} FPS\n\n")
    
    f.write(f"## 推理统计\n\n")
    f.write(f"- **成功帧数**: {successful_frames}/{len(selected_frames)}\n")
    f.write(f"- **成功率**: {successful_frames/len(selected_frames)*100:.1f}%\n\n")
    
    f.write(f"## 平均性能指标\n\n")
    f.write(f"| 指标 | 数值 |\n")
    f.write(f"|------|------|\n")
    f.write(f"| IoU (Jaccard) | {avg_metrics['IoU']:.4f} |\n")
    f.write(f"| Dice Score | {avg_metrics['Dice']:.4f} |\n\n")
    
    f.write(f"## 输出视频\n\n")
    f.write(f"1. **original_with_gt.mp4** - 原图 + Ground Truth叠加（红色）\n")
    f.write(f"2. **original_with_pred.mp4** - 原图 + Sa2VA预测叠加（绿色）\n")
    f.write(f"3. **comparison.mp4** - 四宫格对比视频\n")
    f.write(f"   - 左上: 原图\n")
    f.write(f"   - 右上: Ground Truth\n")
    f.write(f"   - 左下: Sa2VA预测\n")
    f.write(f"   - 右下: GT(红) vs 预测(绿) 叠加\n\n")
    
    f.write(f"## 逐帧指标\n\n")
    f.write(f"| 帧号 | 文件名 | IoU | Dice |\n")
    f.write(f"|------|--------|-----|------|\n")
    for frame in frame_results:
        f.write(f"| {frame['frame_num']} | {frame['image']} | {frame['metrics']['IoU']:.4f} | {frame['metrics']['Dice']:.4f} |\n")

print(f"✅ 报告已保存: {report_path}")

# 保存JSON结果
json_path = os.path.join(OUTPUT_DIR, "video_evaluation_results.json")
results_data = {
    'video_id': selected_prefix,
    'total_frames': len(selected_frames),
    'successful_frames': successful_frames,
    'success_rate': successful_frames / len(selected_frames),
    'average_metrics': avg_metrics,
    'frame_metrics': [
        {
            'frame_num': f['frame_num'],
            'image': f['image'],
            'metrics': f['metrics']
        }
        for f in frame_results
    ],
    'output_videos': [
        'original_with_gt.mp4',
        'original_with_pred.mp4',
        'comparison.mp4'
    ]
}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON结果已保存: {json_path}")

# 总结
print("\n" + "=" * 80)
print("🎉 视频预测完成！")
print("=" * 80)
print(f"\n视频信息:")
print(f"  视频ID: {selected_prefix}")
print(f"  总帧数: {len(selected_frames)}")
print(f"  成功率: {successful_frames/len(selected_frames)*100:.1f}%")
print(f"\n平均性能:")
print(f"  IoU:  {avg_metrics['IoU']:.4f}")
print(f"  Dice: {avg_metrics['Dice']:.4f}")
print(f"\n输出文件:")
print(f"  📁 {OUTPUT_DIR}/")
print(f"     🎬 original_with_gt.mp4 - 原图+GT")
print(f"     🎬 original_with_pred.mp4 - 原图+预测")
print(f"     🎬 comparison.mp4 - 四宫格对比")
print(f"     📄 video_evaluation_report.md")
print(f"     📄 video_evaluation_results.json")
print()
print("=" * 80)
