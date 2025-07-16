# 机器人篮球投篮控制器

基于神经网络的机器人篮球投篮控制系统，能够根据机器人位置和篮筐位置预测最优投篮参数。

## 项目概述

本项目实现了一个智能投篮控制器，通过神经网络模型预测最优的投篮参数：
- **输入**: 机器人位置(x, y)、篮筐位置(x₀, y₀)、篮筐高度h
- **输出**: 初速度v₀、仰角θ_pitch、偏向角θ_yaw

## 项目结构

```
basketball_project/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── environment.yml          # Conda环境配置
├── setup.sh                 # 一键环境创建和运行脚本
├── main.py                  # 主程序入口
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── physics_model.py     # 物理建模模块
│   ├── data_generator.py    # 数据生成模块
│   ├── neural_network.py    # 神经网络模型
│   ├── trainer.py           # 训练模块
│   ├── evaluator.py         # 评估模块
│   └── visualizer.py        # 可视化模块
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── models/              # 训练好的模型
├── results/                 # 结果输出目录
│   ├── figures/             # 图表
│   └── logs/                # 日志文件
└── tests/                   # 测试代码
    └── test_model.py
```

## 快速开始

### 方法一：一键运行（推荐）

```bash
# 给脚本执行权限并运行
chmod +x setup.sh
./setup.sh
```

### 方法二：手动安装

1. **创建Conda环境**
```bash
conda env create -f environment.yml
conda activate basketball_ai
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行程序**
```bash
python main.py
```

## 功能特性

- 🎯 **物理建模**: 基于抛物线运动的投篮物理模型
- 🧠 **神经网络**: 轻量级前馈神经网络，适合嵌入式部署
- 📊 **数据生成**: 自动生成带噪声的训练数据
- 📈 **可视化**: 实时显示训练过程和结果分析
- 🔧 **模型优化**: 支持模型剪枝和量化
- 📱 **嵌入式友好**: 优化的模型结构，支持实时推理

## 技术栈

- **Python 3.8+**
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化
- **Pandas**: 数据处理
- **Scikit-learn**: 机器学习工具

## 模型架构

采用轻量级前馈神经网络：
- 输入层: 5个神经元 (x, y, x₀, y₀, h)
- 隐藏层: 2层，每层64个神经元
- 输出层: 3个神经元 (v₀, θ_pitch, θ_yaw)
- 激活函数: ReLU (隐藏层), Linear (输出层)

## 性能指标

- **模型大小**: < 50KB
- **推理时间**: < 1ms (CPU)
- **精度**: 平均误差 < 5%
- **内存占用**: < 10MB

## 使用说明

运行程序后，系统将自动：
1. 生成训练数据
2. 训练神经网络模型
3. 评估模型性能
4. 显示可视化结果
5. 保存训练好的模型

所有重要结果都会通过弹窗图表展示，包括：
- 训练损失曲线
- 测试误差分布
- 模型预测vs真实值对比
- 不同场景下的性能分析

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License

## 联系方式

如有问题，请通过Issue联系。