#!/bin/bash

# 激活环境并运行预测
source ~/.bashrc
eval "$(/home/ubuntu/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

echo "环境激活成功"
python3 --version
echo ""

cd /home/ubuntu/Sa2VA
python3 predict_with_trained_model.py
