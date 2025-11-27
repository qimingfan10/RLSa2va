"""
检查HuggingFace转换过程是否正确保留了checkpoint的差异
"""
import torch
import os
import numpy as np
from safetensors import safe_open
import json

print("=" * 80)
print("检查HuggingFace转换过程")
print("=" * 80)

# ============================================================================
# 第1部分：检查原始checkpoint
# ============================================================================
print("\n第1部分：原始checkpoint对比")
print("=" * 80)

print("\n加载原始checkpoint...")
old_ckpt = torch.load(
    "/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192.pth",
    map_location='cpu',
    weights_only=False
)

new_ckpt = torch.load(
    "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth", 
    map_location='cpu',
    weights_only=False
)

old_state = old_ckpt['state_dict']
new_state = new_ckpt['state_dict']

print(f"旧checkpoint参数数: {len(old_state)}")
print(f"新checkpoint参数数: {len(new_state)}")

# 找几个关键参数对比
key_params = [
    'mllm.model.language_model.base_model.model.lm_head.modules_to_save.default.weight',
    'grounding_encoder.sam2_model.sam_mask_decoder.output_upscaling.0.weight'
]

print(f"\n原始checkpoint关键参数对比:")
for key in key_params:
    if key in old_state and key in new_state:
        old_param = old_state[key]
        new_param = new_state[key]
        
        diff = torch.abs(old_param - new_param).mean().item()
        max_diff = torch.abs(old_param - new_param).max().item()
        
        print(f"\n  {key}")
        print(f"    形状: {old_param.shape}")
        print(f"    平均差异: {diff:.6e}")
        print(f"    最大差异: {max_diff:.6e}")
        print(f"    旧模型统计: mean={old_param.mean():.4f}, std={old_param.std():.4f}")
        print(f"    新模型统计: mean={new_param.mean():.4f}, std={new_param.std():.4f}")

# ============================================================================
# 第2部分：检查HF转换后的模型
# ============================================================================
print("\n" + "=" * 80)
print("第2部分：HuggingFace模型对比")
print("=" * 80)

old_hf_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
new_hf_path = "/home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"

# 检查config是否相同
print("\n检查config文件...")
with open(os.path.join(old_hf_path, "config.json")) as f:
    old_config = json.load(f)
with open(os.path.join(new_hf_path, "config.json")) as f:
    new_config = json.load(f)

# 关键配置对比
key_configs = ['llm_config', 'vision_config', 'template', 'architectures']
print("\n配置对比:")
for key in key_configs:
    if key in old_config and key in new_config:
        if old_config[key] == new_config[key]:
            print(f"  ✅ {key}: 相同")
        else:
            print(f"  ⚠️  {key}: 不同")

# 检查safetensors权重文件
print("\n" + "=" * 80)
print("检查HF模型权重文件")
print("=" * 80)

old_hf_files = sorted([f for f in os.listdir(old_hf_path) if f.endswith('.safetensors')])
new_hf_files = sorted([f for f in os.listdir(new_hf_path) if f.endswith('.safetensors')])

print(f"\n旧HF模型文件: {len(old_hf_files)}个")
for f in old_hf_files:
    size = os.path.getsize(os.path.join(old_hf_path, f)) / 1024**3
    print(f"  - {f}: {size:.2f}GB")

print(f"\n新HF模型文件: {len(new_hf_files)}个")
for f in new_hf_files:
    size = os.path.getsize(os.path.join(new_hf_path, f)) / 1024**3
    print(f"  - {f}: {size:.2f}GB")

# 对比第一个safetensors文件中的权重
if old_hf_files and new_hf_files:
    print(f"\n详细对比第一个safetensors文件:")
    print(f"  旧: {old_hf_files[0]}")
    print(f"  新: {new_hf_files[0]}")
    
    with safe_open(os.path.join(old_hf_path, old_hf_files[0]), framework="pt", device="cpu") as f:
        old_hf_keys = f.keys()
        old_hf_tensors = {k: f.get_tensor(k) for k in list(old_hf_keys)[:5]}  # 只取前5个
    
    with safe_open(os.path.join(new_hf_path, new_hf_files[0]), framework="pt", device="cpu") as f:
        new_hf_keys = f.keys()
        new_hf_tensors = {k: f.get_tensor(k) for k in list(new_hf_keys)[:5]}  # 只取前5个
    
    print(f"\n  旧HF模型参数数: {len(old_hf_keys)}")
    print(f"  新HF模型参数数: {len(new_hf_keys)}")
    
    # 对比前5个参数
    common_keys = set(old_hf_tensors.keys()) & set(new_hf_tensors.keys())
    print(f"\n  前5个参数对比:")
    for key in sorted(common_keys):
        old_tensor = old_hf_tensors[key]
        new_tensor = new_hf_tensors[key]
        
        diff = torch.abs(old_tensor - new_tensor).mean().item()
        max_diff = torch.abs(old_tensor - new_tensor).max().item()
        
        print(f"\n    {key}")
        print(f"      形状: {old_tensor.shape}")
        print(f"      平均差异: {diff:.6e}")
        print(f"      最大差异: {max_diff:.6e}")

# ============================================================================
# 第3部分：检查转换日志
# ============================================================================
print("\n" + "=" * 80)
print("第3部分：检查转换日志")
print("=" * 80)

# 检查旧模型转换日志
print("\n旧模型转换日志 (convert_to_hf.log):")
if os.path.exists("/home/ubuntu/Sa2VA/convert_to_hf.log"):
    with open("/home/ubuntu/Sa2VA/convert_to_hf.log") as f:
        lines = f.readlines()
    
    # 查找关键信息
    for line in lines:
        if 'Load' in line or 'pretrained' in line or 'checkpoint' in line:
            print(f"  {line.strip()}")
else:
    print("  ⚠️  日志文件不存在")

# 检查新模型转换的最后输出
print("\n新模型转换最后输出:")
print("  检查shell输出...")

# ============================================================================
# 第4部分：关键发现总结
# ============================================================================
print("\n" + "=" * 80)
print("关键发现总结")
print("=" * 80)

print("""
检查要点:
1. ✅ 原始checkpoint权重确实不同（96%参数有差异）
2. ❓ 需要确认HF转换后权重是否保留了这些差异
3. ❓ 检查转换过程是否加载了额外的预训练权重覆盖了训练结果
4. ❓ 确认config中的pretrained_pth路径是否影响了转换

可能的问题:
- 转换脚本中可能加载了Sa2VA-26B.pth作为基础权重
- 这可能覆盖了训练checkpoint中的部分参数
- 导致两个HF模型实际上很相似

建议:
1. 查看convert_to_hf.py的第60-61行，看是否strict=False允许覆盖
2. 检查模型构建时是否先加载了预训练权重
3. 对比HF模型的完整权重文件确认差异
""")

print("\n" + "=" * 80)
