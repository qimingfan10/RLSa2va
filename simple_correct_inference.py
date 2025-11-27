import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

import torch

import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def split_model(model_path):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    print(f"Model {model_path} has {num_layers} layers.")

    # Since the first GPU will be used for ViT, treat it as 0.5 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    print(f"num_layers_per_gpu: {num_layers_per_gpu}")
    
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    device_map['grounding_encoder'] = rank
    device_map['text_hidden_fcs'] = rank

    return device_map

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('image_folder', help='Path to image file')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-8B")
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

if __name__ == "__main__":
    # 使用固定的模型路径
    model_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",  # 启用自动多GPU分配
        trust_remote_code=True
    )
    """
    # For distributed inference, uncomment the following lines to get device_map
    device_map=split_model(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    """

    if 'qwen' in model_path.lower():
        print("Using AutoProcessor for Qwen-VL model.")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )


    # 使用固定的测试图片
    test_image_path = "/home/ubuntu/Sa2VA/data/merged_vessel_data/images/Chen_Fang_0000103366__1-4_1_04B2D3CF_frame_000034.jpg"
    img_frame = Image.open(test_image_path).convert('RGB')
    
    text = "<image>Please segment the blood vessel."
    
    print(f"测试图片: {test_image_path}")
    print(f"推理文本: {text}")
    print()
    
    try:
        result = model.predict_forward(
            image=img_frame,
            text=text,
            tokenizer=tokenizer,
            processor=processor,
        )
        
        prediction = result['prediction']
        print(f"✅ 推理成功！")
        print(f"模型输出: {prediction}")
        print()
        
        if '[SEG]' in prediction:
            print("✅ 输出包含 [SEG] 标记")
            if 'prediction_masks' in result:
                pred_masks = result['prediction_masks']
                print(f"✅ 获得预测mask: {len(pred_masks)} 个")
                
                if Visualizer is not None:
                    work_dir = "/home/ubuntu/Sa2VA/simple_correct_inference_results"
                    os.makedirs(work_dir, exist_ok=True)
                    
                    _seg_idx = 0
                    pred_mask = pred_masks[_seg_idx][0]  # [seg_idx][frame_idx]
                    visualize(pred_mask, test_image_path, work_dir)
                    print(f"✅ 结果保存到: {work_dir}")
            else:
                print("⚠️  没有prediction_masks")
        else:
            print("⚠️  输出不包含 [SEG] 标记")
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
