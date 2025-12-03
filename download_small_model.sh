#!/bin/bash

# 下载InternVL3-2B模型（推荐用于24GB显存）

echo "开始下载InternVL3-2B模型..."

cd /home/ubuntu/Sa2VA/models

# 使用huggingface-cli下载
huggingface-cli download OpenGVLab/InternVL3-2B \
    --local-dir InternVL3-2B \
    --local-dir-use-symlinks False

echo "✅ 模型下载完成: /home/ubuntu/Sa2VA/models/InternVL3-2B"
echo ""
echo "修改配置文件中的path为："
echo "path = '/home/ubuntu/Sa2VA/models/InternVL3-2B'"
