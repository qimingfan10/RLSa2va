"""
简化版：正确的阈值扫描实验
直接修改模型推理逻辑，获取概率图
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf')

from transformers import AutoModelForCausalLM, AutoTokenizer


def monkey_patch_predict_forward(model):
    """
    Monkey patch模型的predict_forward方法
    返回概率图而非二值化mask
    """
    original_predict_forward = model.predict_forward
    
    def predict_forward_with_probs(
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
            return_probs=False,  # 新参数
    ):
        # 调用原始方法，但我们需要拦截mask生成部分
        # 这里我们复制原始逻辑并修改关键部分
        
        if not model.init_prediction_config:
            assert tokenizer
            model.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            return original_predict_forward(
                image=image, video=video, text=text, past_text=past_text,
                mask_prompts=mask_prompts, tokenizer=tokenizer, processor=processor
            )
        
        # === 图像处理部分（与原始相同）===
        input_dict = {}
        ori_image_size = image.size

        # prepare grounding images
        g_image = np.array(image)
        g_image = model.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(model.torch_dtype)
        extra_pixel_values = [g_pixel_values]
        g_pixel_values = torch.stack([
            model.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
        ]).to(model.torch_dtype)

        # Dynamic preprocess
        from modeling_sa2va_chat import dynamic_preprocess
        images = dynamic_preprocess(image, model.min_dynamic_patch,
                                    model.max_dynamic_patch,
                                    model.image_size, model.use_thumbnail)

        if mask_prompts is not None:
            vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
            input_dict['vp_overall_mask'] = vp_overall_mask
        else:
            input_dict['vp_overall_mask'] = None

        pixel_values = [model.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(model.torch_dtype)
        num_image_tokens = pixel_values.shape[0] * model.patch_token
        num_frames = 1
        
        input_dict['g_pixel_values'] = g_pixel_values
        input_dict['pixel_values'] = pixel_values
        vp_token_str = ''

        image_token_str = f'{model.IMG_START_TOKEN}' \
                          f'{model.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{model.IMG_END_TOKEN}'
        image_token_str = image_token_str + '\n'
        image_token_str = image_token_str * num_frames
        image_token_str = image_token_str.strip()

        text = text.replace('<image>', image_token_str + vp_token_str)
        input_text = ''
        input_text += model.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=model.bot_name)
        input_text = past_text + input_text
        ids = model.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        mm_inputs = {
            'pixel_values': input_dict['pixel_values'],
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'prompt_masks': mask_prompts,
            'vp_overall_mask': input_dict['vp_overall_mask'],
        }

        # === 生成部分 ===
        generate_output = model.generate(
            **mm_inputs,
            generation_config=model.gen_config,
            streamer=None,
            bos_token_id=model.tokenizer.bos_token_id,
            stopping_criteria=model.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = model.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        # === Mask生成部分（关键修改）===
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        
        from modeling_sa2va_chat import get_seg_hidden_states
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=model.seg_token_idx
        )
        all_seg_hidden_states = model.text_hidden_fcs(seg_hidden_states)

        ret_masks = []
        ret_probs = []
        
        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = model.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            
            # ===== 关键修改：保留概率图 =====
            prob_maps = masks.sigmoid()  # 概率图 [0, 1]
            
            if return_probs:
                # 返回概率图
                ret_probs.append(prob_maps.cpu().numpy())
            else:
                # 返回二值化mask（兼容原有接口）
                binary_masks = prob_maps > 0.5
                ret_masks.append(binary_masks.cpu().numpy())

        if return_probs:
            return {
                'prediction': predict,
                'probability_maps': ret_probs,
            }
        else:
            return {
                'prediction': predict,
                'prediction_masks': ret_masks,
            }
    
    return predict_forward_with_probs


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    
    return {'dice': dice, 'iou': iou, 'recall': recall, 'precision': precision}


def main():
    print("="*80)
    print("正确的阈值扫描实验（简化版）")
    print("="*80)
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # 加载模型
    print("\n加载Sa2VA模型...")
    model_path = '/home/ubuntu/Sa2VA/models/sa2va_vessel_hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model.eval()
    print("✅ 模型加载成功")
    
    # Monkey patch推理方法
    print("修改推理方法以获取概率图...")
    predict_fn = monkey_patch_predict_forward(model)
    
    # 加载数据
    print("\n加载数据集...")
    data_root = '/home/ubuntu/Sa2VA/data/merged_vessel_data'
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        all_data = json.load(f)
    
    # 使用验证集前50张
    n_train = int(len(all_data) * 0.8)
    val_data = all_data[n_train:n_train + 50]
    print(f"使用{len(val_data)}张图像")
    
    # 步骤1: 推理获取概率图
    print("\n步骤1: 推理获取概率图...")
    prob_data = []
    
    for idx, item in enumerate(tqdm(val_data, desc="推理")):
        image_path = os.path.join(data_root, 'images', item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # 处理mask路径 - 跳过有问题的数据
        if 'mask' not in item:
            continue
        mask_item = item['mask']
        if isinstance(mask_item, list):
            if len(mask_item) == 0 or not isinstance(mask_item[0], str):
                continue
            mask_file = mask_item[0]
        elif isinstance(mask_item, str):
            mask_file = mask_item
        else:
            continue
            
        mask_path = os.path.join(data_root, 'masks', mask_file)
        if not os.path.exists(mask_path):
            continue
            
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        try:
            result = predict_fn(
                image=image,
                text='<image>\nPlease segment the blood vessel.',
                tokenizer=tokenizer,
                return_probs=True
            )
            
            if len(result['probability_maps']) > 0:
                prob_map = result['probability_maps'][0][0]
                prob_data.append({
                    'image': item['image'],
                    'prob_map': prob_map,
                    'gt_mask': gt_mask
                })
        except Exception as e:
            print(f"\n⚠️ 样本{idx}推理失败: {e}")
            continue
    
    print(f"\n✅ 成功推理{len(prob_data)}个样本")
    
    # 步骤2: 测试不同阈值
    print("\n步骤2: 测试不同阈值...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    all_results = {}
    
    print(f"\n{'阈值':<8} {'Dice':<8} {'Recall':<8} {'Precision':<10}")
    print("-" * 40)
    
    for threshold in thresholds:
        metrics_list = []
        
        for data in prob_data:
            pred_mask = (data['prob_map'] > threshold).astype(np.uint8)
            metrics = calculate_metrics(pred_mask, data['gt_mask'])
            metrics_list.append(metrics)
        
        avg_metrics = {
            'threshold': float(threshold),
            'dice': np.mean([m['dice'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'iou': np.mean([m['iou'] for m in metrics_list]),
        }
        
        all_results[threshold] = avg_metrics
        
        print(f"{threshold:<8.2f} {avg_metrics['dice']:<8.4f} {avg_metrics['recall']:<8.4f} {avg_metrics['precision']:<10.4f}")
    
    # 找最优
    print("\n" + "="*80)
    print("最优阈值")
    print("="*80)
    
    best_dice_thresh = max(all_results.items(), key=lambda x: x[1]['dice'])
    best_recall_thresh = max(all_results.items(), key=lambda x: x[1]['recall'])
    
    print(f"\n最高Dice: 阈值={best_dice_thresh[0]:.2f}")
    print(f"  Dice={best_dice_thresh[1]['dice']:.4f}")
    print(f"  Recall={best_dice_thresh[1]['recall']:.4f}")
    print(f"  Precision={best_dice_thresh[1]['precision']:.4f}")
    
    print(f"\n最高Recall: 阈值={best_recall_thresh[0]:.2f}")
    print(f"  Dice={best_recall_thresh[1]['dice']:.4f}")
    print(f"  Recall={best_recall_thresh[1]['recall']:.4f}")
    print(f"  Precision={best_recall_thresh[1]['precision']:.4f}")
    
    # 保存结果
    output_dir = './threshold_scan_correct'
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'num_samples': len(prob_data),
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n✅ 结果保存至: {results_path}")
    
    # 绘图
    import matplotlib.pyplot as plt
    
    thresholds = sorted(all_results.keys())
    dices = [all_results[t]['dice'] for t in thresholds]
    recalls = [all_results[t]['recall'] for t in thresholds]
    precisions = [all_results[t]['precision'] for t in thresholds]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, dices, 'b-o', label='Dice', linewidth=2, markersize=8)
    plt.plot(thresholds, recalls, 'g-s', label='Recall', linewidth=2, markersize=8)
    plt.plot(thresholds, precisions, 'm-^', label='Precision', linewidth=2, markersize=8)
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target 0.85')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Metrics vs Threshold (Correct Version)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'threshold_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ 曲线图保存至: {plot_path}")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()
