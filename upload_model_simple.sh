#!/bin/bash

# 简化版HuggingFace模型上传脚本
# 直接上传两个模型

set -e

echo "========================================================================"
echo "Sa2VA模型上传到HuggingFace (简化版)"
echo "========================================================================"
echo ""

cd /home/ubuntu/Sa2VA

# 检查是否登录
echo "检查登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ 未登录HuggingFace"
    echo "请先运行: huggingface-cli login"
    exit 1
fi

echo "✅ 已登录HuggingFace"
echo ""

# 上传模型1
echo "========================================================================"
echo "上传模型1: sa2va-vessel-hf (iter_12192)"
echo "========================================================================"
echo ""

if [ -d "models/sa2va_vessel_hf" ]; then
    echo "源目录: models/sa2va_vessel_hf"
    echo "大小: $(du -sh models/sa2va_vessel_hf | cut -f1)"
    echo "目标: qimingfan10/sa2va-vessel-hf"
    echo ""
    echo "开始上传... (预计1-2小时)"
    echo ""
    
    huggingface-cli upload \
        qimingfan10/sa2va-vessel-hf \
        models/sa2va_vessel_hf \
        --repo-type model \
        --commit-message "Upload Sa2VA vessel segmentation model (iter_12192)"
    
    echo ""
    echo "✅ 模型1上传完成!"
    echo "地址: https://huggingface.co/qimingfan10/sa2va-vessel-hf"
    echo ""
else
    echo "⚠️  跳过：目录不存在 models/sa2va_vessel_hf"
    echo ""
fi

# 上传模型2
echo "========================================================================"
echo "上传模型2: sa2va-vessel-iter3672-hf (iter_3672)"
echo "========================================================================"
echo ""

if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
    echo "源目录: models/sa2va_vessel_iter3672_hf"
    echo "大小: $(du -sh models/sa2va_vessel_iter3672_hf | cut -f1)"
    echo "目标: qimingfan10/sa2va-vessel-iter3672-hf"
    echo ""
    echo "开始上传... (预计1-2小时)"
    echo ""
    
    huggingface-cli upload \
        qimingfan10/sa2va-vessel-iter3672-hf \
        models/sa2va_vessel_iter3672_hf \
        --repo-type model \
        --commit-message "Upload Sa2VA vessel segmentation model (iter_3672)"
    
    echo ""
    echo "✅ 模型2上传完成!"
    echo "地址: https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf"
    echo ""
else
    echo "⚠️  跳过：目录不存在 models/sa2va_vessel_iter3672_hf"
    echo ""
fi

echo "========================================================================"
echo "✅ 所有模型上传完成！"
echo "========================================================================"
echo ""
echo "模型地址:"
echo "  - https://huggingface.co/qimingfan10/sa2va-vessel-hf"
echo "  - https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf"
echo ""
echo "下一步："
echo "  1. 访问模型页面编辑README"
echo "  2. 添加标签和描述"
echo "  3. 验证模型可以下载"
