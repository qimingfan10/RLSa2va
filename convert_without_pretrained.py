"""
修复版的HF转换脚本：不加载预训练权重，只使用训练checkpoint
"""
import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
import re

print("=" * 80)
print("修复版HF转换脚本：不加载预训练权重")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Fixed HF conversion script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument('--save-path', type=str, default=None, help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # 加载配置
    print("\n步骤1: 加载配置文件")
    print(f"  配置: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # ⚠️ 关键修改：禁用预训练权重加载
    print("\n步骤2: 禁用预训练权重加载")
    if hasattr(cfg.model, 'pretrained_pth') and cfg.model.pretrained_pth is not None:
        print(f"  原配置中的pretrained_pth: {cfg.model.pretrained_pth}")
        print(f"  ✅ 临时设置为None，避免加载预训练权重")
        original_pretrained_pth = cfg.model.pretrained_pth
        cfg.model.pretrained_pth = None
    else:
        original_pretrained_pth = None
        print(f"  配置中没有pretrained_pth")
    
    # 构建模型（不会加载预训练权重）
    print("\n步骤3: 构建模型（不加载预训练权重）")
    model = BUILDER.build(cfg.model)
    print("  ✅ 模型构建完成")
    
    backend = get_file_backend(args.pth_model)

    # 加载训练checkpoint
    print(f"\n步骤4: 加载训练checkpoint")
    print(f"  Checkpoint: {args.pth_model}")
    
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)
    else:
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)

    state_dict = state_dict['state_dict']
    print(f"  Checkpoint中的参数数: {len(state_dict)}")
    
    # 加载训练权重
    print(f"\n步骤5: 加载训练权重到模型")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f'  ✅ 训练权重加载完成')
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
        for key in missing_keys[:5]:
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")

    iter_str = os.path.basename(args.pth_model).split('.')[0]

    print(f"\n步骤6: 合并LoRA权重（如果有）")
    model._merge_lora()
    print("  ✅ LoRA合并完成")

    model.mllm.model.modules_to_save = None
    model.mllm.model.transfer_to_hf = True

    all_state_dict = model.all_state_dict()
    print(f"  模型总参数数: {len(all_state_dict)}")

    all_state_dict_new = {}

    # 构建HF格式模型
    print(f"\n步骤7: 构建HuggingFace格式模型")
    arch_type = cfg.model.get('arch_type', 'internvl')
    print(f"  架构类型: {arch_type}")
    
    from projects.sa2va.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.sa2va.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    if 'qwen' in arch_type:
        if 'qwen3' in cfg.path.lower():
            from projects.sa2va.hf.models_qwen3vl.configuration_sa2va_chat import Sa2VAChatConfigQwen
            from projects.sa2va.hf.models_qwen3vl.modeling_sa2va_qwen import Sa2VAChatModelQwen
        else:
            from projects.sa2va.hf.models_qwen2_5_vl.configuration_sa2va_chat import Sa2VAChatConfigQwen
            from projects.sa2va.hf.models_qwen2_5_vl.modeling_sa2va_qwen import Sa2VAChatModelQwen

    if 'qwen' not in arch_type:
        config = Sa2VAChatConfig.from_pretrained(cfg.path)
    else:
        config = Sa2VAChatConfigQwen.from_pretrained(cfg.path)
    
    config_dict = config.to_dict()
    
    if 'qwen' in arch_type:
        config_dict["text_config"]["vocab_size"] = len(model.mllm.tokenizer)
        config_dict["tie_word_embeddings"] = False
    else:
        config_dict["llm_config"]["vocab_size"] = len(model.mllm.tokenizer)

    template_str = cfg.template
    if 'qwen' in arch_type:
        print("  Qwen模型检测，移除system prompt")
        system_prompt_pattern = re.compile(
            r"{% if loop\.first and message\['role'] != 'system' %}.*?<\|im_end\|>\s*{% endif %}",
            re.DOTALL
        )
        template_str = system_prompt_pattern.sub('', template_str)

    config_dict["template"] = template_str

    if 'qwen' in arch_type:
        name_map = {'mllm.': '', '.gamma': '.g_weight'}
        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]

        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfigQwen',
         'AutoModel': 'modeling_sa2va_qwen.Sa2VAChatModelQwen',
         'AutoModelForCausalLM': 'modeling_sa2va_qwen.Sa2VAChatModelQwen'}

        sa2va_hf_config = Sa2VAChatConfigQwen(**config_dict)
        sa2va_hf_config.text_config.tie_word_embeddings = False

        sa2va_hf_config.save_pretrained("./tmp/sa2va_config_test_qwen")

    else:
        name_map = {'mllm.model.': '', '.gamma': '.g_weight'}

        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]
        
        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfig',
         'AutoModel': 'modeling_sa2va_chat.Sa2VAChatModel',
         'AutoModelForCausalLM': 'modeling_sa2va_chat.Sa2VAChatModel'}
        
        sa2va_hf_config = Sa2VAChatConfig(**config_dict)
        sa2va_hf_config.save_pretrained("./tmp/sa2va_config_test")

    # 创建HF模型
    print(f"\n步骤8: 创建HuggingFace模型实例")
    if 'qwen' not in arch_type:
        hf_sa2va_model = Sa2VAChatModel(sa2va_hf_config)
    else:
        hf_sa2va_model = Sa2VAChatModelQwen(sa2va_hf_config)

    print("\n--- 权重加载报告 ---")
    missing_keys_report, unexpected_keys_report = hf_sa2va_model.load_state_dict(
        all_state_dict_new, strict=False
    )

    if not missing_keys_report and not unexpected_keys_report:
        print("所有权重匹配成功！")
    else:
        if missing_keys_report:
            print(f"Missing keys: {len(missing_keys_report)}")
            for key in missing_keys_report[:10]:
                print(f"  - {key}")
        if unexpected_keys_report:
            print(f"Unexpected keys: {len(unexpected_keys_report)}")
            for key in unexpected_keys_report[:10]:
                print(f"  - {key}")

    # 保存模型
    print(f"\n步骤9: 保存HuggingFace模型")
    save_path = args.save_path if args.save_path else f"./{iter_str}_hf"
    print(f"  保存路径: {save_path}")
    
    hf_sa2va_model.save_pretrained(save_path)
    model.mllm.tokenizer.save_pretrained(save_path)
    
    # 复制必要的文件
    import shutil
    source_dir = "projects/sa2va/hf/models" if 'qwen' not in arch_type else "projects/sa2va/hf/models_qwen2_5_vl"
    files_to_copy = [
        'configuration_sa2va_chat.py',
        'modeling_sa2va_chat.py',
        'configuration_intern_vit.py',
        'modeling_intern_vit.py',
        'flash_attention.py',
        'sam2.py'
    ]
    
    for file in files_to_copy:
        src = os.path.join(source_dir, file)
        if os.path.exists(src):
            shutil.copy(src, save_path)
            print(f"  ✅ 复制 {file}")

    print(f"\n" + "=" * 80)
    print(f"✅ HuggingFace模型转换完成！")
    print(f"=" * 80)
    print(f"\n保存路径: {save_path}")
    print(f"\n验证要点:")
    print(f"  1. 此转换未加载预训练权重{original_pretrained_pth if original_pretrained_pth else 'N/A'}")
    print(f"  2. 所有权重来自训练checkpoint: {args.pth_model}")
    print(f"  3. 此HF模型应该保留训练checkpoint的所有差异")
    print()

if __name__ == '__main__':
    main()
