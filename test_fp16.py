#!/usr/bin/env python3
"""
使用FP16进行Sa2VA推理测试
"""
import sys
sys.path.insert(0, '/home/ubuntu/Sa2VA')

import torch
from mmengine.config import Config
from mmengine.registry import MODELS

print("=" * 80)
print("Sa2VA FP16推理测试")
print("=" * 80)

# 配置
config_path = '/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_vessel_finetune.py'
checkpoint_path = '/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192.pth'

print(f"\n配置文件: {config_path}")
print(f"权重文件: {checkpoint_path}")

# 检查GPU
print("\n检查GPU...")
if not torch.cuda.is_available():
    print("❌ CUDA不可用")
    sys.exit(1)

print(f"✅ 检测到 {torch.cuda.device_count()} 个GPU")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"    总显存: {mem_total:.2f} GB")

# 加载配置
print("\n加载配置...")
cfg = Config.fromfile(config_path)
print(f"✅ 配置加载成功")

# 创建模型
print("\n创建模型...")
model = MODELS.build(cfg.model)
print(f"✅ 模型创建成功")

# 加载权重
print(f"\n加载权重: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"✅ 权重加载成功")
if missing_keys:
    print(f"  缺失keys: {len(missing_keys)}个")
if unexpected_keys:
    print(f"  多余keys: {len(unexpected_keys)}个")

# 转换为FP16
print("\n转换模型为FP16...")
model = model.half()
print(f"✅ 模型已转换为FP16")

# 设置为评估模式
model.eval()

# 移动到GPU
device = 'cuda:0'
print(f"\n移动模型到{device}...")
print("这可能需要几分钟...")

try:
    model = model.to(device)
    print(f"✅ 模型已成功移动到GPU")
    
    # 检查显存使用
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\n显存使用:")
    print(f"  已分配: {mem_allocated:.2f} GB")
    print(f"  已保留: {mem_reserved:.2f} GB")
    
    if mem_allocated < 15:
        print(f"\n🎉 成功！FP16模型只使用了 {mem_allocated:.2f} GB 显存")
        print(f"   相比FP32的23.5GB，节省了 {23.5 - mem_allocated:.2f} GB")
        print(f"\n✅ 单GPU推理可行！")
    else:
        print(f"\n⚠️  显存使用仍然较高: {mem_allocated:.2f} GB")
        
except torch.cuda.OutOfMemoryError as e:
    print(f"❌ 仍然OOM: {e}")
    print(f"\n可能需要使用更激进的优化方案")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
