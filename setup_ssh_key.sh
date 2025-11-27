#!/bin/bash

echo "========================================================================"
echo "设置GitHub SSH密钥"
echo "========================================================================"
echo ""

# 检查是否已有SSH密钥
if [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "✅ 已存在SSH密钥"
    echo ""
    echo "您的公钥内容:"
    echo "----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "----------------------------------------"
    echo ""
elif [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✅ 已存在SSH密钥 (RSA)"
    echo ""
    echo "您的公钥内容:"
    echo "----------------------------------------"
    cat ~/.ssh/id_rsa.pub
    echo "----------------------------------------"
    echo ""
else
    echo "未找到SSH密钥，正在生成..."
    echo ""
    
    read -p "输入您的GitHub邮箱: " email
    
    ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""
    
    echo ""
    echo "✅ SSH密钥已生成"
    echo ""
    echo "您的公钥内容:"
    echo "----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "----------------------------------------"
    echo ""
fi

# 添加到ssh-agent
echo "添加密钥到ssh-agent..."
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519 2>/dev/null || ssh-add ~/.ssh/id_rsa 2>/dev/null

echo ""
echo "========================================================================"
echo "下一步: 添加SSH密钥到GitHub"
echo "========================================================================"
echo ""
echo "1. 复制上面的公钥内容"
echo "2. 访问: https://github.com/settings/ssh/new"
echo "3. Title: Sa2VA Ubuntu Server"
echo "4. Key: 粘贴公钥内容"
echo "5. 点击 'Add SSH key'"
echo ""
read -p "完成后按Enter继续..."

# 测试SSH连接
echo ""
echo "测试SSH连接..."
ssh -T git@github.com

echo ""
echo "========================================================================"
echo "更新远程仓库为SSH URL"
echo "========================================================================"

cd /home/ubuntu/Sa2VA

# 移除HTTPS远程仓库
git remote remove origin

# 添加SSH远程仓库
git remote add origin git@github.com:qimingfan10/RLSa2va.git

echo "✅ 远程仓库已更新为SSH"
echo ""
git remote -v

echo ""
echo "========================================================================"
echo "推送代码到GitHub"
echo "========================================================================"
echo ""
read -p "现在推送代码? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================================================"
        echo "✅ 代码已成功推送到GitHub!"
        echo "========================================================================"
        echo ""
        echo "仓库地址: https://github.com/qimingfan10/RLSa2va"
    else
        echo ""
        echo "❌ 推送失败，请检查SSH密钥是否正确配置"
    fi
fi
