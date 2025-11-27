"""检查两个checkpoint的差异"""
import torch
import numpy as np

print("=" * 80)
print("检查两个checkpoint的权重差异")
print("=" * 80)

# 加载两个checkpoint
print("\n加载checkpoint...")
print("旧模型: work_dirs/vessel_segmentation/iter_12192.pth")
old_ckpt = torch.load(
    "/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation/iter_12192.pth",
    map_location='cpu',
    weights_only=False
)

print("新模型: work_dirs/merged_vessel_segmentation/iter_3672.pth")
new_ckpt = torch.load(
    "/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth", 
    map_location='cpu',
    weights_only=False
)

print("\n" + "=" * 80)
print("基本信息对比")
print("=" * 80)

print(f"\n旧模型键值:")
print(f"  总键数: {len(old_ckpt.keys())}")
for key in sorted(old_ckpt.keys()):
    if key != 'state_dict':
        print(f"  - {key}: {old_ckpt[key]}")

print(f"\n新模型键值:")
print(f"  总键数: {len(new_ckpt.keys())}")
for key in sorted(new_ckpt.keys()):
    if key != 'state_dict':
        print(f"  - {key}: {new_ckpt[key]}")

# 检查state_dict
old_state = old_ckpt['state_dict']
new_state = new_ckpt['state_dict']

print(f"\n" + "=" * 80)
print("state_dict对比")
print("=" * 80)

print(f"\n旧模型参数数量: {len(old_state)}")
print(f"新模型参数数量: {len(new_state)}")

# 检查键是否相同
old_keys = set(old_state.keys())
new_keys = set(new_state.keys())

only_old = old_keys - new_keys
only_new = new_keys - old_keys
common = old_keys & new_keys

print(f"\n共同参数: {len(common)}")
print(f"仅旧模型有: {len(only_old)}")
print(f"仅新模型有: {len(only_new)}")

if only_old:
    print("\n仅旧模型有的参数:")
    for key in sorted(only_old)[:10]:
        print(f"  - {key}")

if only_new:
    print("\n仅新模型有的参数:")
    for key in sorted(only_new)[:10]:
        print(f"  - {key}")

# 对比共同参数的权重差异
print(f"\n" + "=" * 80)
print("权重差异分析（共同参数）")
print("=" * 80)

identical_count = 0
different_count = 0
max_diff = 0
max_diff_key = None
diff_summary = []

for key in sorted(common):
    old_param = old_state[key]
    new_param = new_state[key]
    
    if old_param.shape != new_param.shape:
        print(f"\n⚠️  形状不匹配: {key}")
        print(f"   旧: {old_param.shape}, 新: {new_param.shape}")
        continue
    
    # 计算差异
    diff = torch.abs(old_param - new_param)
    mean_diff = diff.mean().item()
    max_diff_value = diff.max().item()
    
    if max_diff_value > max_diff:
        max_diff = max_diff_value
        max_diff_key = key
    
    if mean_diff < 1e-8:  # 几乎相同
        identical_count += 1
    else:
        different_count += 1
        diff_summary.append({
            'key': key,
            'mean_diff': mean_diff,
            'max_diff': max_diff_value,
            'shape': old_param.shape
        })

print(f"\n相同参数数量: {identical_count}/{len(common)} ({identical_count/len(common)*100:.2f}%)")
print(f"不同参数数量: {different_count}/{len(common)} ({different_count/len(common)*100:.2f}%)")

if max_diff_key:
    print(f"\n最大差异:")
    print(f"  参数名: {max_diff_key}")
    print(f"  最大差异值: {max_diff:.6e}")

if diff_summary:
    print(f"\n差异最大的前10个参数:")
    diff_summary.sort(key=lambda x: x['max_diff'], reverse=True)
    for i, item in enumerate(diff_summary[:10], 1):
        print(f"  {i}. {item['key']}")
        print(f"     平均差异: {item['mean_diff']:.6e}, 最大差异: {item['max_diff']:.6e}")
        print(f"     形状: {item['shape']}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if identical_count == len(common):
    print("\n✅ 两个checkpoint的权重完全相同！")
    print("   这解释了为什么推理性能完全一致。")
    print("\n可能原因:")
    print("  1. 两个checkpoint实际上是同一个文件的复制")
    print("  2. 训练没有实际更新权重")
    print("  3. 训练从相同的初始点开始且收敛到完全相同的解")
elif different_count < len(common) * 0.01:  # 少于1%不同
    print(f"\n⚠️  两个checkpoint几乎相同（{different_count/len(common)*100:.2f}%参数不同）")
    print("   这可能解释了为什么推理性能完全一致。")
else:
    print(f"\n❓ 两个checkpoint有明显差异（{different_count/len(common)*100:.2f}%参数不同）")
    print("   但推理性能相同，需要进一步分析原因。")
    print("\n可能原因:")
    print("  1. 差异的参数不影响推理结果")
    print("  2. 评估样本数太少（10张）")
    print("  3. 两个模型都收敛到相同的局部最优")

print("\n" + "=" * 80)
