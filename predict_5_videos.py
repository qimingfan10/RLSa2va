"""
Sa2VA视频预测 - 预测5个视频序列
使用官方predict_forward方法，对多个视频序列进行预测并生成对比MP4视频
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
print("Sa2VA视频预测 - 预测5个视频序列")
print("=" * 80)

# 配置
HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT = "/home/ubuntu/Sa2VA/data/merged_vessel_data/"
OUTPUT_BASE_DIR = "/home/ubuntu/Sa2VA/video_prediction_5_videos"
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

def create_comparison_video(frame_results, output_path, fps=5, video_type='4panel'):
    """创建对比视频"""
    if not frame_results:
        print(f"⚠️  没有帧数据，跳过视频生成")
        return False
    
    first_frame = frame_results[0]
    h, w = first_frame['image_np'].shape[:2]
    
    if video_type == '4panel':
        out_h, out_w = h * 2, w * 2
    elif video_type == 'gt_only':
        out_h, out_w = h, w
    else:  # pred_only
        out_h, out_w = h, w
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    for frame_data in frame_results:
        image_np = frame_data['image_np']
        gt_mask = frame_data['gt_mask']
        pred_mask = frame_data['pred_mask']
        
        # 转换为BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 创建彩色overlay
        gt_overlay = image_bgr.copy()
        pred_overlay = image_bgr.copy()
        
        gt_mask_colored = np.zeros_like(image_bgr)
        gt_mask_colored[:, :, 1] = gt_mask  # 绿色
        pred_mask_colored = np.zeros_like(image_bgr)
        pred_mask_colored[:, :, 2] = pred_mask  # 红色
        
        gt_overlay = cv2.addWeighted(gt_overlay, 0.7, gt_mask_colored, 0.3, 0)
        pred_overlay = cv2.addWeighted(pred_overlay, 0.7, pred_mask_colored, 0.3, 0)
        
        if video_type == '4panel':
            # 添加标签
            cv2.putText(image_bgr, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(gt_overlay, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(pred_overlay, 'Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # IoU文本
            iou_text = f"IoU: {frame_data['metrics']['IoU']:.3f}"
            both_overlay = image_bgr.copy()
            cv2.putText(both_overlay, f'IoU: {frame_data["metrics"]["IoU"]:.3f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # 组合4个panel
            top = np.hstack([image_bgr, gt_overlay])
            bottom = np.hstack([pred_overlay, both_overlay])
            combined = np.vstack([top, bottom])
        elif video_type == 'gt_only':
            combined = gt_overlay
        else:  # pred_only
            combined = pred_overlay
        
        out.write(combined)
    
    out.release()
    return True

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

# 步骤3: 对多个视频序列进行推理
print("\n" + "=" * 80)
print("步骤3: 对5个视频序列进行推理")
print("=" * 80)

all_videos_results = []

for video_idx, (selected_prefix, selected_frames) in enumerate(selected_videos):
    video_num = START_VIDEO_INDEX + video_idx
    print(f"\n{'=' * 80}")
    print(f"正在处理视频 #{video_num}: {selected_prefix}")
    print(f"  总帧数: {len(selected_frames)}")
    print(f"{'=' * 80}")
    
    # 为每个视频创建输出目录
    video_output_dir = os.path.join(OUTPUT_BASE_DIR, f"video_{video_num:02d}_{selected_prefix}")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(os.path.join(video_output_dir, "frames"), exist_ok=True)
    
    all_metrics = []
    frame_results = []
    successful_frames = 0
    
    # 推理每一帧
    for idx, (frame_num, sample) in enumerate(tqdm(selected_frames, desc=f"视频{video_num}推理")):
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
    
    print(f"\n✅ 视频 #{video_num} 推理完成!")
    print(f"   成功: {successful_frames}/{len(selected_frames)}")
    print(f"   成功率: {successful_frames/len(selected_frames)*100:.1f}%")
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        print(f"\n视频 #{video_num} 平均指标:")
        print(f"   IoU (Jaccard): {avg_metrics['IoU']:.4f}")
        print(f"   Dice: {avg_metrics['Dice']:.4f}")
    else:
        avg_metrics = {'IoU': 0.0, 'Dice': 0.0}
    
    # 生成MP4视频
    print(f"\n生成MP4视频...")
    
    # 4-panel对比视频
    video_4panel_path = os.path.join(video_output_dir, f"video_{video_num:02d}_4panel_comparison.mp4")
    if create_comparison_video(frame_results, video_4panel_path, fps=5, video_type='4panel'):
        print(f"  ✅ 4-panel视频: {video_4panel_path}")
    
    # 保存JSON结果
    video_results = {
        'video_index': video_num,
        'video_name': selected_prefix,
        'num_frames': len(selected_frames),
        'successful_frames': successful_frames,
        'success_rate': successful_frames / len(selected_frames) if selected_frames else 0,
        'avg_metrics': avg_metrics,
        'per_frame_metrics': [
            {
                'frame_num': fr['frame_num'],
                'image': fr['image'],
                'metrics': fr['metrics']
            }
            for fr in frame_results
        ]
    }
    
    json_path = os.path.join(video_output_dir, f"video_{video_num:02d}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(video_results, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON结果: {json_path}")
    
    # 保存到总结果
    all_videos_results.append(video_results)

# 步骤4: 生成综合报告
print("\n" + "=" * 80)
print("步骤4: 生成综合报告")
print("=" * 80)

# 计算总体统计
total_frames = sum(vr['num_frames'] for vr in all_videos_results)
total_successful = sum(vr['successful_frames'] for vr in all_videos_results)
overall_avg_iou = np.mean([vr['avg_metrics']['IoU'] for vr in all_videos_results])
overall_avg_dice = np.mean([vr['avg_metrics']['Dice'] for vr in all_videos_results])

print(f"\n总体统计:")
print(f"  预测视频数: {len(all_videos_results)}")
print(f"  总帧数: {total_frames}")
print(f"  成功帧数: {total_successful}")
print(f"  总成功率: {total_successful/total_frames*100:.1f}%")
print(f"  平均IoU: {overall_avg_iou:.4f}")
print(f"  平均Dice: {overall_avg_dice:.4f}")

print(f"\n各视频详细结果:")
for vr in all_videos_results:
    print(f"\n  视频 #{vr['video_index']}: {vr['video_name']}")
    print(f"    帧数: {vr['num_frames']}")
    print(f"    成功率: {vr['success_rate']*100:.1f}%")
    print(f"    IoU: {vr['avg_metrics']['IoU']:.4f}")
    print(f"    Dice: {vr['avg_metrics']['Dice']:.4f}")

# 保存综合结果
summary_results = {
    'model_path': HF_MODEL_PATH,
    'num_videos': len(all_videos_results),
    'total_frames': total_frames,
    'total_successful_frames': total_successful,
    'overall_success_rate': total_successful / total_frames if total_frames > 0 else 0,
    'overall_avg_iou': overall_avg_iou,
    'overall_avg_dice': overall_avg_dice,
    'videos': all_videos_results
}

summary_path = os.path.join(OUTPUT_BASE_DIR, "summary_5_videos.json")
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 综合报告已保存: {summary_path}")

# 生成Markdown报告
md_path = os.path.join(OUTPUT_BASE_DIR, "SUMMARY_5_VIDEOS.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Sa2VA 5个视频预测结果总结\n\n")
    f.write(f"**模型**: `{HF_MODEL_PATH}`\n\n")
    f.write(f"**预测时间**: {total_frames} 帧来自 {len(all_videos_results)} 个视频\n\n")
    
    f.write("## 总体性能\n\n")
    f.write(f"- **总成功率**: {total_successful/total_frames*100:.2f}% ({total_successful}/{total_frames})\n")
    f.write(f"- **平均IoU**: {overall_avg_iou:.4f}\n")
    f.write(f"- **平均Dice**: {overall_avg_dice:.4f}\n\n")
    
    f.write("## 各视频详细结果\n\n")
    f.write("| 视频# | 视频名称 | 帧数 | 成功率 | IoU | Dice |\n")
    f.write("|------|---------|------|--------|-----|------|\n")
    for vr in all_videos_results:
        f.write(f"| {vr['video_index']} | {vr['video_name']} | {vr['num_frames']} | ")
        f.write(f"{vr['success_rate']*100:.1f}% | {vr['avg_metrics']['IoU']:.4f} | ")
        f.write(f"{vr['avg_metrics']['Dice']:.4f} |\n")
    
    f.write("\n## 输出文件\n\n")
    for vr in all_videos_results:
        video_dir = f"video_{vr['video_index']:02d}_{vr['video_name']}"
        f.write(f"### 视频 #{vr['video_index']}: {vr['video_name']}\n\n")
        f.write(f"- 4-panel视频: `{video_dir}/video_{vr['video_index']:02d}_4panel_comparison.mp4`\n")
        f.write(f"- JSON结果: `{video_dir}/video_{vr['video_index']:02d}_results.json`\n\n")

print(f"✅ Markdown报告已保存: {md_path}")

print("\n" + "=" * 80)
print("✅ 5个视频预测完成！")
print("=" * 80)
print(f"\n输出目录: {OUTPUT_BASE_DIR}")
print(f"\n查看结果:")
print(f"  - 综合报告: cat {md_path}")
print(f"  - JSON数据: cat {summary_path}")
print(f"  - 各视频MP4: ls {OUTPUT_BASE_DIR}/video_*/")
print()
