#!/bin/bash

echo "========================================================================"
echo "清理磁盘空间以完成模型转换"
echo "========================================================================"

echo "当前磁盘使用情况:"
df -h /home/ubuntu | tail -1
echo ""

# 1. 删除中间checkpoint（保留最终的iter_3672.pth）
echo "步骤1: 删除中间训练checkpoint..."
echo "  保留: iter_3672.pth (最终checkpoint)"
echo "  删除: iter_2000.pth, iter_2500.pth, iter_3000.pth, iter_3500.pth"
echo ""

rm -f /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_2000.pth
rm -f /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_2500.pth
rm -f /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3000.pth
rm -f /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3500.pth

echo "✅ 删除中间checkpoint完成 (释放约10GB)"
df -h /home/ubuntu | tail -1
echo ""

# 2. 删除部分转换的HF模型
echo "步骤2: 删除部分转换的HF模型..."
rm -rf /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation_hf/

echo "✅ 删除部分转换的HF模型完成 (释放约1GB)"
df -h /home/ubuntu | tail -1
echo ""

# 3. 删除之前错误的推理结果
echo "步骤3: 删除之前错误的推理结果..."
rm -rf /home/ubuntu/Sa2VA/fixed_sa2va_inference_results/
rm -rf /home/ubuntu/Sa2VA/final_working_inference_results/
rm -rf /home/ubuntu/Sa2VA/simple_sa2va_inference_results/
rm -rf /home/ubuntu/Sa2VA/real_sa2va_inference_results/
rm -rf /home/ubuntu/Sa2VA/real_inference_fixed_results/
rm -rf /home/ubuntu/Sa2VA/multi_gpu_inference_results/
rm -rf /home/ubuntu/Sa2VA/evaluation_results/
rm -rf /home/ubuntu/Sa2VA/dataset_samples_visualization/
rm -rf /home/ubuntu/Sa2VA/real_hf_inference_results/

echo "✅ 删除推理结果完成 (释放约50MB)"
df -h /home/ubuntu | tail -1
echo ""

# 4. 删除训练日志目录
echo "步骤4: 删除旧的训练日志目录..."
rm -rf /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/20251123_*

echo "✅ 删除训练日志完成"
df -h /home/ubuntu | tail -1
echo ""

# 5. 删除tmp目录
echo "步骤5: 清理临时文件..."
rm -rf /home/ubuntu/Sa2VA/tmp/
rm -f /home/ubuntu/Sa2VA/*.tar.gz

echo "✅ 清理临时文件完成"
df -h /home/ubuntu | tail -1
echo ""

echo "========================================================================"
echo "清理完成！"
echo "========================================================================"
echo ""
echo "最终磁盘使用情况:"
df -h /home/ubuntu | tail -1
echo ""
echo "释放的空间应该足够完成HuggingFace模型转换了！"
echo ""
echo "下一步: bash convert_to_hf.sh"
