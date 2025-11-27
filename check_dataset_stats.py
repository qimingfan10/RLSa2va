import json
from collections import Counter

# 加载annotations.json
with open('/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json') as f:
    data = json.load(f)

print("数据集统计")
print("=" * 50)
print(f"总样本数: {len(data)}")
print()

# 统计mask数量
total_masks = sum(len(item['mask']) for item in data)
print(f"总mask数: {total_masks}")
print(f"平均每张图片的mask数: {total_masks/len(data):.2f}")
print()

# 检查是否有重复
images = [item['image'] for item in data]
unique_images = set(images)
print(f"唯一图片数: {len(unique_images)}")
if len(images) != len(unique_images):
    print("是否有重复: 是")
else:
    print("是否有重复: 否")
print()

# 统计mask数量分布
mask_counts = [len(item['mask']) for item in data]
mask_distribution = Counter(mask_counts)
print("Mask数量分布:")
for count in sorted(mask_distribution.keys()):
    print(f"  {count}个mask: {mask_distribution[count]}张图片")
