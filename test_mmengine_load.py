#!/usr/bin/env python3
"""测试mmengine.load是否改变了数据类型"""

import json
import mmengine
from pycocotools import mask as mask_utils

# 方法1: 直接用json.load
print("=" * 60)
print("方法1: json.load")
print("=" * 60)
with open('/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json') as f:
    data1 = json.load(f)

sample1 = data1[0]
seg1 = sample1['mask'][0]
print(f"seg类型: {type(seg1)}")
print(f"seg[0]类型: {type(seg1[0])}")
print(f"前4个元素: {seg1[:4]}")

try:
    rles = mask_utils.frPyObjects([seg1], 512, 512)
    print("✅ frPyObjects成功!")
except Exception as e:
    print(f"❌ frPyObjects失败: {e}")

# 方法2: 用mmengine.load
print("\n" + "=" * 60)
print("方法2: mmengine.load")
print("=" * 60)
data2 = mmengine.load('/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json', file_format='json')

sample2 = data2[0]
seg2 = sample2['mask'][0]
print(f"seg类型: {type(seg2)}")
print(f"seg[0]类型: {type(seg2[0])}")
print(f"前4个元素: {seg2[:4]}")

try:
    rles = mask_utils.frPyObjects([seg2], 512, 512)
    print("✅ frPyObjects成功!")
except Exception as e:
    print(f"❌ frPyObjects失败: {e}")
    print(f"错误类型: {type(e)}")
    
    # 详细调试
    print(f"\nseg2详细信息:")
    print(f"  type(seg2): {type(seg2)}")
    print(f"  isinstance(seg2, list): {isinstance(seg2, list)}")
    print(f"  len(seg2): {len(seg2)}")
    if len(seg2) > 0:
        print(f"  type(seg2[0]): {type(seg2[0])}")
        print(f"  isinstance(seg2[0], (int, float)): {isinstance(seg2[0], (int, float))}")
        print(f"  isinstance(seg2[0], (int, float, np.number)): {isinstance(seg2[0], (int, float))}")
    
    # 尝试转换
    print("\n尝试转换:")
    seg2_list = list(seg2)
    print(f"  list(seg2)类型: {type(seg2_list)}")
    try:
        rles = mask_utils.frPyObjects([seg2_list], 512, 512)
        print("  ✅ list()转换后成功!")
    except Exception as e2:
        print(f"  ❌ list()转换后仍失败: {e2}")
        
        # 尝试深度转换
        seg2_float = [float(x) for x in seg2]
        print(f"  [float(x) for x in seg2]类型: {type(seg2_float)}")
        try:
            rles = mask_utils.frPyObjects([seg2_float], 512, 512)
            print("  ✅ float转换后成功!")
        except Exception as e3:
            print(f"  ❌ float转换后仍失败: {e3}")

print("\n" + "=" * 60)
