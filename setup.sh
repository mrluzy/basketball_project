#!/bin/bash

# 机器人篮球投篮控制器 - 一键安装和运行脚本
# Author: Basketball AI Team
# Date: 2024

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 主函数
main() {
    echo "="*60
    echo "🏀 机器人篮球投篮控制器 - 一键安装和运行"
    echo "="*60
    echo

    # 检查conda是否安装
    if ! command_exists conda; then
        print_error "Conda未安装，请先安装Anaconda或Miniconda"
        print_info "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    print_success "检测到Conda已安装"

    # 检查Python是否安装
    if ! command_exists python; then
        print_error "Python未安装，请先安装Python 3.8+"
        exit 1
    fi

    print_success "检测到Python已安装"

    # 获取当前目录
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$PROJECT_DIR"

    print_info "项目目录: $PROJECT_DIR"

    # 检查环境是否已存在
    if conda env list | grep -q "basketball_ai"; then
        print_warning "环境 'basketball_ai' 已存在"
        read -p "是否要重新创建环境? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "删除现有环境..."
            conda env remove -n basketball_ai -y
        else
            print_info "使用现有环境"
        fi
    fi

    # 创建conda环境
    if ! conda env list | grep -q "basketball_ai"; then
        print_info "创建Conda环境 'basketball_ai'..."
        conda env create -f environment.yml
        print_success "环境创建完成"
    fi

    # 激活环境并安装依赖
    print_info "激活环境并安装依赖..."
    
    # 使用conda run来在指定环境中运行命令
    conda run -n basketball_ai pip install -r requirements.txt
    print_success "依赖安装完成"

    # 创建必要的目录结构
    print_info "创建项目目录结构..."
    mkdir -p data/{raw,processed,models}
    mkdir -p results/{figures,logs}
    mkdir -p tests
    mkdir -p src
    print_success "目录结构创建完成"

    # 运行主程序
    print_info "启动篮球投篮控制器..."
    echo
    print_warning "注意: 程序运行过程中会弹出多个图表窗口，请不要关闭"
    echo
    
    # 在conda环境中运行主程序
    conda run -n basketball_ai python main.py

    print_success "程序运行完成！"
    echo
    print_info "结果文件保存在以下位置:"
    print_info "  - 训练好的模型: data/models/"
    print_info "  - 结果图表: results/figures/"
    print_info "  - 日志文件: results/logs/"
    echo
    print_info "如需重新运行，请执行:"
    print_info "  conda activate basketball_ai"
    print_info "  python main.py"
}

# 错误处理
trap 'print_error "脚本执行过程中发生错误，请检查上述输出信息"' ERR

# 运行主函数
main "$@"