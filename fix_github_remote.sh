#!/bin/bash

echo "========================================================================"
echo "修复GitHub远程仓库配置"
echo "========================================================================"
echo ""

cd /home/ubuntu/Sa2VA

# 步骤1: 移除旧的远程仓库
echo "步骤1: 移除旧的远程仓库..."
git remote remove origin

if [ $? -eq 0 ]; then
    echo "✅ 旧的远程仓库已移除"
else
    echo "⚠️  没有找到旧的远程仓库（可能已经删除）"
fi

echo ""

# 步骤2: 添加新的远程仓库
echo "步骤2: 添加新的远程仓库..."
git remote add origin https://github.com/qimingfan10/RLSa2va.git

echo "✅ 新的远程仓库已添加"
echo ""

# 步骤3: 验证
echo "步骤3: 验证远程仓库配置..."
git remote -v

echo ""
echo "========================================================================"
echo "✅ 远程仓库配置已更新！"
echo "========================================================================"
echo ""
echo "下一步: 推送代码"
echo ""
echo "选择推送方式:"
echo ""
echo "方式A: 使用Personal Access Token (推荐)"
echo "  1. 访问: https://github.com/settings/tokens"
echo "  2. 生成token (勾选 repo 权限)"
echo "  3. 运行: git push -u origin main"
echo "  4. 用户名: qimingfan10"
echo "  5. 密码: 输入你的token (不是GitHub密码)"
echo ""
echo "方式B: 使用SSH密钥"
echo "  运行: bash setup_ssh_key.sh"
echo ""
read -p "选择方式 (A/B): " method

if [[ $method == "A" || $method == "a" ]]; then
    echo ""
    echo "请按照以下步骤操作:"
    echo "1. 打开浏览器访问: https://github.com/settings/tokens"
    echo "2. 点击 'Generate new token' > 'Generate new token (classic)'"
    echo "3. 勾选 'repo' 权限"
    echo "4. 生成并复制token"
    echo ""
    read -p "准备好了吗？按Enter继续推送..."
    
    git push -u origin main
    
elif [[ $method == "B" || $method == "b" ]]; then
    echo ""
    echo "正在设置SSH密钥..."
    bash /home/ubuntu/Sa2VA/setup_ssh_key.sh
else
    echo ""
    echo "已取消。请手动运行:"
    echo "  git push -u origin main"
fi
