#!/bin/bash

echo "========================================================================"
echo "完整分析：100张图片评估 + 转换过程检查"
echo "========================================================================"

# 设置micromamba路径
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export PATH="/home/ubuntu/micromamba/micromamba/bin:$PATH"

# 初始化micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"

# 激活环境
echo "激活topo-sarl环境..."
micromamba activate topo-sarl

if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

echo "✅ 环境激活成功"
echo ""

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "第1步：检查转换过程"
echo "========================================================================"
echo ""

python check_conversion_process.py 2>&1 | tee conversion_check.log

echo ""
echo "========================================================================"
echo "第2步：100张图片大规模评估"
echo "========================================================================"
echo ""
echo "⚠️  这将需要较长时间（预计20-30分钟）"
echo ""

python evaluate_100_samples.py 2>&1 | tee evaluation_100_samples.log

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  1. 转换检查: cat conversion_check.log"
echo "  2. 100张评估: cat evaluation_100_samples.log"
echo "  3. 详细结果: cat comparison_100_samples/comparison_100_samples.json"
echo ""
