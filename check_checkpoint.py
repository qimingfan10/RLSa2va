#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查训练checkpoint的信息
"""

import torch
import sys

def check_checkpoint(checkpoint_path):
    """检查checkpoint内容"""
    print(f"正在加载checkpoint: {checkpoint_path}\n")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("=" * 80)
    print("Checkpoint 信息:")
    print("=" * 80)
    
    # 打印顶层键
    print(f"\n顶层键: {list(checkpoint.keys())}\n")
    
    # 检查state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"State Dict 包含 {len(state_dict)} 个参数")
        print(f"\n前10个参数键:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {i+1}. {key}: {shape}")
        print(f"  ... (共 {len(state_dict)} 个参数)")
    
    # 检查meta信息
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        print(f"\nMeta 信息:")
        for key, value in meta.items():
            if key not in ['state_dict']:
                print(f"  {key}: {value}")
    
    # 检查message_hub
    if 'message_hub' in checkpoint:
        message_hub = checkpoint['message_hub']
        print(f"\nMessage Hub:")
        for key, value in message_hub.items():
            print(f"  {key}: {value}")
    
    # 检查训练迭代信息
    for key in ['iter', 'epoch', 'step']:
        if key in checkpoint:
            print(f"\n{key.capitalize()}: {checkpoint[key]}")
    
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_checkpoint(checkpoint_path)
