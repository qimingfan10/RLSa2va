"""
测试现有的HF模型是否可用
"""
import os
import sys
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODEL_PATH = "/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
TEST_IMAGE = "/home/ubuntu/Sa2VA/data/merged_vessel_data/images/Chen_Fang_0000103366__1-4_1_04B2D3CF_frame_000034.jpg"

print(f"测试HF模型: {HF_MODEL_PATH}")
print(f"测试图片: {TEST_IMAGE}")
print()

# 检查模型文件
if not os.path.exists(HF_MODEL_PATH):
    print("❌ 模型不存在")
    exit(1)

if not os.path.exists(TEST_IMAGE):
    print("❌ 测试图片不存在")
    exit(1)

try:
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载成功")
    
    print("\n2. 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ 模型加载成功")
    
    print("\n3. 加载测试图片...")
    image = Image.open(TEST_IMAGE).convert('RGB')
    print(f"✅ 图片加载成功: {image.size}")
    
    print("\n4. 进行推理...")
    text = "<image>Please segment the blood vessel."
    
    result = model.predict_forward(
        image=image,
        text=text,
        tokenizer=tokenizer,
        processor=None
    )
    
    print("✅ 推理成功！")
    print(f"\n预测文本: {result.get('prediction', 'N/A')}")
    
    if 'prediction_masks' in result:
        print(f"预测mask数量: {len(result['prediction_masks'])}")
        print("\n✅✅✅ 这个HF模型可以使用！")
    else:
        print("\n⚠️  没有预测mask")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n这个HF模型可能有问题，需要重新转换")
