#!/usr/bin/env python3
"""调试mask格式问题"""

import json
import numpy as np
from pycocotools import mask as mask_utils

# 加载数据
with open('/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json') as f:
    data = json.load(f)

print("=" * 60)
print("测试前10个样本的mask格式")
print("=" * 60)

for i in range(min(10, len(data))):
    sample = data[i]
    print(f"\n样本 {i}: {sample['image']}")
    print(f"  mask数量: {len(sample['mask'])}")
    print(f"  text数量: {len(sample['text'])}")
    
    for j, seg in enumerate(sample['mask']):
        print(f"\n  Mask {j}:")
        print(f"    类型: {type(seg)}")
        print(f"    长度: {len(seg)}")
        print(f"    前4个元素: {seg[:4]}")
        print(f"    前4个元素类型: {[type(x) for x in seg[:4]]}")
        
        # 尝试frPyObjects
        try:
            # 确保是list
            if not isinstance(seg, list):
                print(f"    ❌ seg不是list，是{type(seg)}")
                continue
            
            # 检查元素类型
            if len(seg) > 0 and not isinstance(seg[0], (int, float)):
                print(f"    ❌ seg[0]不是数字，是{type(seg[0])}")
                print(f"    seg内容: {seg}")
                continue
            
            # 尝试调用
            rles = mask_utils.frPyObjects([seg], 512, 512)
            print(f"    ✅ frPyObjects成功!")
            
        except Exception as e:
            print(f"    ❌ frPyObjects失败: {e}")
            print(f"    错误类型: {type(e)}")
            
            # 打印详细信息
            print(f"    seg详细信息:")
            print(f"      isinstance(seg, list): {isinstance(seg, list)}")
            print(f"      len(seg): {len(seg)}")
            if len(seg) > 0:
                print(f"      type(seg[0]): {type(seg[0])}")
                print(f"      seg[0]: {seg[0]}")
            
            # 尝试转换
            try:
                seg_converted = [float(x) for x in seg]
                rles = mask_utils.frPyObjects([seg_converted], 512, 512)
                print(f"    ✅ 转换为float后成功!")
            except Exception as e2:
                print(f"    ❌ 转换后仍失败: {e2}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
