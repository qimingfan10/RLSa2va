#!/bin/bash

# Sa2VA模型后台上传脚本
# 账户: ly17
# 上传两个模型到HuggingFace

set -e

LOG_FILE="/home/ubuntu/Sa2VA/upload_models.log"

echo "========================================================================"
echo "Sa2VA模型后台上传 - 开始时间: $(date)"
echo "========================================================================"

cd /home/ubuntu/Sa2VA

# 检查登录状态
echo "检查HuggingFace登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ 错误：未登录HuggingFace"
    echo "请先运行: huggingface-cli login"
    exit 1
fi

echo "✅ 已登录为: $(huggingface-cli whoami 2>&1 | grep -v Warning)"
echo ""

# ============================================================================
# 上传模型1: sa2va-vessel-hf (iter_12192)
# ============================================================================
echo "========================================================================"
echo "模型1: ly17/sa2va-vessel-hf"
echo "========================================================================"
echo "源目录: models/sa2va_vessel_hf"
echo "大小: $(du -sh models/sa2va_vessel_hf 2>/dev/null | cut -f1 || echo '未知')"
echo "开始时间: $(date)"
echo ""

if [ -d "models/sa2va_vessel_hf" ]; then
    echo "开始上传模型1..."
    
    huggingface-cli upload \
        ly17/sa2va-vessel-hf \
        models/sa2va_vessel_hf \
        --repo-type model \
        --commit-message "Upload Sa2VA vessel model (iter_12192)"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型1上传成功!"
        echo "完成时间: $(date)"
        echo "地址: https://huggingface.co/ly17/sa2va-vessel-hf"
        echo ""
    else
        echo ""
        echo "❌ 模型1上传失败!"
        echo "失败时间: $(date)"
        exit 1
    fi
else
    echo "⚠️  跳过模型1：目录不存在 models/sa2va_vessel_hf"
    echo ""
fi

# ============================================================================
# 上传模型2: sa2va-vessel-iter3672-hf (iter_3672)
# ============================================================================
echo "========================================================================"
echo "模型2: ly17/sa2va-vessel-iter3672-hf"
echo "========================================================================"
echo "源目录: models/sa2va_vessel_iter3672_hf"
echo "大小: $(du -sh models/sa2va_vessel_iter3672_hf 2>/dev/null | cut -f1 || echo '未知')"
echo "开始时间: $(date)"
echo ""

if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
    echo "开始上传模型2..."
    
    huggingface-cli upload \
        ly17/sa2va-vessel-iter3672-hf \
        models/sa2va_vessel_iter3672_hf \
        --repo-type model \
        --commit-message "Upload Sa2VA vessel model (iter_3672)"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型2上传成功!"
        echo "完成时间: $(date)"
        echo "地址: https://huggingface.co/ly17/sa2va-vessel-iter3672-hf"
        echo ""
    else
        echo ""
        echo "❌ 模型2上传失败!"
        echo "失败时间: $(date)"
        exit 1
    fi
else
    echo "⚠️  跳过模型2：目录不存在 models/sa2va_vessel_iter3672_hf"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo "========================================================================"
echo "✅ 所有模型上传完成！"
echo "========================================================================"
echo "完成时间: $(date)"
echo ""
echo "模型地址:"
echo "  - https://huggingface.co/ly17/sa2va-vessel-hf"
echo "  - https://huggingface.co/ly17/sa2va-vessel-iter3672-hf"
echo ""
echo "日志文件: $LOG_FILE"
echo ""
echo "下一步："
echo "  1. 访问HuggingFace编辑模型README"
echo "  2. 添加标签和描述"
echo "  3. 验证模型可以下载"
echo ""
echo "========================================================================"
