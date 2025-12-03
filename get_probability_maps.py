"""
修改Sa2VA推理代码，获取原始概率图而非二值化mask
关键修改：移除 > 0.5 的二值化步骤
"""
import torch
import torch.nn.functional as F
from typing import List, Dict


def predict_forward_with_probs(
        model,
        image=None,
        video=None,
        text=None,
        past_text='',
        mask_prompts=None,
        tokenizer=None,
        processor=None,
):
    """
    修改版的predict_forward，返回概率图而非二值化mask
    
    关键修改：
    原始代码: masks = masks.sigmoid() > 0.5  (二值化)
    修改后:   masks = masks.sigmoid()         (保留概率)
    """
    if not model.init_prediction_config:
        assert tokenizer
        model.preparing_for_generation(tokenizer=tokenizer)

    if image is None and video is None and '<image>' not in past_text:
        text = text.replace('<image>', "")
        input_text = ''
        input_text += model.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=model.bot_name)
        input_text = past_text + input_text
        ids = model.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        mm_inputs = {
            'pixel_values': None,
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'prompt_masks': None,
            'vp_overall_mask': None,
        }
        ret_masks = []
        ret_probs = []
    else:
        input_dict = {}
        if video is not None:
            raise NotImplementedError("Video not supported in this version")
        else:
            ori_image_size = image.size

            # prepare grounding images
            g_image = torch.tensor(image)  # for grounding
            g_image = model.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(model.torch_dtype)
            extra_pixel_values = [g_pixel_values]
            g_pixel_values = torch.stack([
                model.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(model.torch_dtype)

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

        if mask_prompts is not None:
            raise NotImplementedError("mask_prompts not supported in this version")
        else:
            vp_token_str = ''

        image_token_str = f'{model.IMG_START_TOKEN}' \
                          f'{model.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{model.IMG_END_TOKEN}'
        image_token_str = image_token_str + '\n'
        image_token_str = image_token_str * num_frames
        image_token_str = image_token_str.strip()

        ret_masks = []
        ret_probs = []

        if '<image>' in text or mask_prompts is not None:
            assert past_text is None or len(past_text) == 0
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

    if image is None and video is None and '<image>' not in past_text:
        return {
            'prediction': predict,
            'prediction_masks': ret_masks,
            'probability_maps': ret_probs,
        }

    # if have seg result, find the seg hidden states
    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    
    from modeling_sa2va_chat import get_seg_hidden_states
    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=model.seg_token_idx
    )
    all_seg_hidden_states = model.text_hidden_fcs(seg_hidden_states)

    for seg_hidden_states in all_seg_hidden_states:
        seg_hidden_states = seg_hidden_states.unsqueeze(0)
        g_pixel_values = input_dict['g_pixel_values']
        sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        pred_masks = model.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
        w, h = ori_image_size
        masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
        masks = masks[:, 0]
        
        # ===== 关键修改 =====
        # 原始: masks = masks.sigmoid() > 0.5  (二值化)
        # 修改: 分别保存概率图和二值化mask
        prob_maps = masks.sigmoid()  # 保留概率图 (0.0 - 1.0)
        binary_masks = prob_maps > 0.5  # 默认0.5阈值的二值化
        
        # 转为numpy
        prob_maps_np = prob_maps.cpu().numpy()
        binary_masks_np = binary_masks.cpu().numpy()
        
        ret_probs.append(prob_maps_np)
        ret_masks.append(binary_masks_np)

    return {
        'prediction': predict,
        'prediction_masks': ret_masks,  # 二值化mask (兼容性)
        'probability_maps': ret_probs,  # 概率图 (新增)
    }


def apply_threshold(probability_map, threshold):
    """
    对概率图应用阈值
    
    Args:
        probability_map: numpy array, shape (H, W), 值域 [0, 1]
        threshold: float, 阈值 (0, 1)
        
    Returns:
        binary_mask: numpy array, shape (H, W), 值域 {0, 1}
    """
    return (probability_map > threshold).astype(int)
