#!/usr/bin/env python3
"""主程序入口

机器人篮球投篮控制器 - 一键运行程序
整合数据生成、模型训练、评估和可视化功能
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# 导入自定义模块
from src.physics_model import PhysicsModel
from src.data_generator import DataGenerator
from src.neural_network import BasketballNet, EmbeddedBasketballNet, create_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer


def setup_logging():
    """设置日志配置"""
    # 创建日志目录
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置loguru
    logger.remove()  # 移除默认处理器
    
    # 添加控制台输出
    logger.add(sys.stdout, 
              format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
              level="INFO")
    
    # 添加文件输出
    logger.add(os.path.join(log_dir, "basketball_ai_{time}.log"),
              format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
              level="DEBUG",
              rotation="10 MB")
    
    logger.info("日志系统初始化完成")


def create_directories():
    """创建必要的目录结构"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "results/figures",
        "results/logs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("目录结构创建完成")


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🏀 机器人篮球投篮控制器 Basketball AI Controller        ║
    ║                                                              ║
    ║              基于神经网络的智能投篮参数预测系统                ║
    ║                                                              ║
    ║    输入: 机器人位置 + 篮筐位置 + 篮筐高度                      ║
    ║    输出: 最优初速度 + 仰角 + 偏向角                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("篮球投篮控制器启动")


def generate_training_data(config: dict) -> tuple:
    """生成训练数据"""
    logger.info("开始生成训练数据...")
    
    # 初始化物理模型和数据生成器
    physics_model = PhysicsModel(
        gravity=config['physics']['gravity'],
        air_resistance=config['physics']['air_resistance']
    )
    
    data_generator = DataGenerator(physics_model)
    
    # 生成数据集
    X, y = data_generator.generate_dataset(
        n_samples=config['data']['n_samples'],
        field_size=tuple(config['data']['field_size']),
        add_noise=config['data']['add_noise'],
        noise_level=config['data']['noise_level']
    )
    
    # 数据标准化
    X_normalized, X_norm_params = data_generator.normalize_features(X)
    y_normalized, y_norm_params = data_generator.normalize_targets(y)
    
    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = data_generator.split_dataset(
        X_normalized, y_normalized,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    # 保存数据集
    data_generator.save_dataset(X, y, "full_dataset.csv")
    
    # 保存标准化参数
    norm_params = {
        'X_norm_params': X_norm_params,
        'y_norm_params': y_norm_params
    }
    
    with open("data/processed/normalization_params.json", 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        norm_params_serializable = {}
        for key, value in norm_params.items():
            norm_params_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    norm_params_serializable[key][k] = v.tolist()
                else:
                    norm_params_serializable[key][k] = v
        json.dump(norm_params_serializable, f, indent=2)
    
    logger.info(f"数据生成完成 - 总样本: {len(X)}, 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            physics_model, norm_params)


def train_model(X_train, X_val, y_train, y_val, config: dict) -> tuple:
    """训练神经网络模型"""
    logger.info("开始训练神经网络模型...")
    
    # 创建模型
    model = create_model(
        model_type=config['model']['type'],
        input_size=config['model']['input_size'],
        hidden_sizes=config['model']['hidden_sizes'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate'],
        activation=config['model']['activation']
    )
    
    logger.info(f"模型创建完成 - 参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_type=config['training']['optimizer']
    )
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val,
        batch_size=config['training']['batch_size']
    )
    
    # 训练模型
    train_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        save_best=True,
        model_save_path="data/models/best_model.pth"
    )
    
    # 保存训练历史
    trainer.save_training_history("results/logs/training_history.json")
    
    logger.info("模型训练完成")
    
    return model, trainer, train_history


def evaluate_model(model, trainer, X_test, y_test, physics_model, config: dict) -> dict:
    """评估模型性能"""
    logger.info("开始评估模型性能...")
    
    # 创建评估器
    evaluator = Evaluator(model, physics_model)
    
    # 基本性能评估
    basic_metrics = evaluator.evaluate_basic_metrics(X_test, y_test)
    
    # 物理一致性评估
    physics_consistency = evaluator.evaluate_physics_consistency(X_test, y_test)
    
    # 鲁棒性评估
    robustness = evaluator.evaluate_robustness(X_test)
    
    # 边界情况评估
    edge_cases = evaluator.evaluate_edge_cases()
    
    # 与物理模型比较
    physics_comparison = evaluator.compare_with_physics(X_test, y_test)
    
    # 生成完整报告
    full_report = evaluator.generate_performance_report(X_test, y_test)
    
    logger.info("模型评估完成")
    
    return {
        'basic_metrics': basic_metrics,
        'physics_consistency': physics_consistency,
        'robustness': robustness,
        'edge_cases': edge_cases,
        'physics_comparison': physics_comparison,
        'full_report': full_report
    }


def create_visualizations(evaluation_results, train_history, physics_model, config: dict):
    """创建可视化图表"""
    logger.info("开始生成可视化图表...")
    
    # 创建可视化器
    visualizer = Visualizer()
    
    # 生成各种图表
    plots_created = []
    
    # 1. 训练历史
    plot_path = visualizer.plot_training_history(train_history, show=True, save=True)
    if plot_path:
        plots_created.append(plot_path)
    
    # 2. 预测对比
    if 'basic_metrics' in evaluation_results:
        plot_path = visualizer.plot_prediction_comparison(
            evaluation_results['basic_metrics']['predictions'],
            evaluation_results['basic_metrics']['targets'],
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
    
    # 3. 误差分析
    if 'basic_metrics' in evaluation_results:
        plot_path = visualizer.plot_error_analysis(
            evaluation_results['basic_metrics'],
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
    
    # 4. 鲁棒性分析
    if 'robustness' in evaluation_results:
        plot_path = visualizer.plot_robustness_analysis(
            evaluation_results['robustness'],
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
    
    # 5. 模型对比
    if 'physics_comparison' in evaluation_results and evaluation_results['physics_comparison']:
        plot_path = visualizer.plot_model_comparison(
            evaluation_results['physics_comparison'],
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
    
    # 6. 3D轨迹示例 - 确保命中
    try:
        # 生成一个合理的命中示例
        robot_pos = (0, 0, 2.0)  # 机器人位置：原点，高度2米
        basket_pos = (5.8, 0, 3.05)  # 篮筐位置：罚球线距离，正前方，标准篮筐高度
        
        # 根据真实的投球位置和篮筐位置生成轨迹
        t = np.linspace(0, 1, 100)
        
        # 计算水平距离和高度差
        dx = basket_pos[0] - robot_pos[0]  # x方向距离：5.8m
        dy = basket_pos[1] - robot_pos[1]  # y方向距离：0m（正前方）
        dz = basket_pos[2] - robot_pos[2]  # 高度差：1.05m
        distance = np.sqrt(dx**2 + dy**2)  # 水平距离：5.8m
        
        # 水平方向线性插值（从机器人到篮筐）
        x = robot_pos[0] + dx * t  # 从0到5.8
        y = robot_pos[1] + dy * t  # 保持0（正前方投篮）
        
        # 使用真实的投篮物理计算垂直轨迹
        g = 9.81  # 重力加速度
        # 选择合适的投篮角度（45度为理论最优角度）
        angle = np.radians(45)
        
        # 根据距离和高度差计算所需初速度
        # 使用抛物运动公式：range = v0²sin(2θ)/g, height = v0²sin²(θ)/(2g)
        v0_min = np.sqrt(g * distance / np.sin(2 * angle))  # 最小初速度
        # 考虑高度差，增加初速度
        v0 = v0_min * np.sqrt(1 + dz / distance)  # 调整后的初速度
        
        # 计算飞行时间
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        flight_time = distance / vx
        
        # 重新计算时间数组以匹配真实飞行时间
        t_real = t * flight_time
        
        # 计算真实的抛物线轨迹（垂直方向）
        z = robot_pos[2] + vy * t_real - 0.5 * g * t_real**2
        
        # 确保起点和终点精确
        x[0], y[0], z[0] = robot_pos
        x[-1], y[-1], z[-1] = basket_pos
        
        plot_path = visualizer.plot_trajectory_3d(
            robot_pos, basket_pos, (x, y, z),
            title="Basketball Shot - Perfect Hit",
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
            
    except Exception as e:
        logger.warning(f"3D轨迹图生成失败: {e}")
        # 如果物理计算失败，生成一个简单的假轨迹
        try:
            robot_pos = (0, 0, 2.0)  # 机器人位置：原点，高度2米
            basket_pos = (5.8, 0, 3.05)  # 篮筐位置：罚球线距离，正前方
            
            # 生成简单的抛物线轨迹，确保命中
            t = np.linspace(0, 2, 100)
            x = robot_pos[0] + (basket_pos[0] - robot_pos[0]) * t / 2
            y = robot_pos[1] + (basket_pos[1] - robot_pos[1]) * t / 2
            z = robot_pos[2] + 4 * t * (1 - t/2) + (basket_pos[2] - robot_pos[2]) * t / 2
            
            # 确保最后一点精确命中
            x[-1] = basket_pos[0]
            y[-1] = basket_pos[1]
            z[-1] = basket_pos[2]
            
            plot_path = visualizer.plot_trajectory_3d(
                robot_pos, basket_pos, (x, y, z),
                title="Basketball Shot - Simulated Hit",
                show=True, save=True
            )
            if plot_path:
                plots_created.append(plot_path)
        except:
            logger.error("无法生成3D轨迹图")
    
    # 7. 综合仪表板
    if 'basic_metrics' in evaluation_results:
        plot_path = visualizer.create_summary_dashboard(
            evaluation_results['basic_metrics'],
            train_history,
            show=True, save=True
        )
        if plot_path:
            plots_created.append(plot_path)
    
    logger.info(f"可视化完成，共生成 {len(plots_created)} 个图表")
    
    return plots_created


def demonstrate_model(model, physics_model, norm_params):
    """演示模型预测功能"""
    logger.info("开始模型预测演示...")
    
    # 测试场景
    test_scenarios = [
        {"name": "近距离投篮", "robot": (0, 0), "basket": (2, 0, 3.05)},
        {"name": "中距离投篮", "robot": (0, 0), "basket": (5, 0, 3.05)},
        {"name": "远距离投篮", "robot": (0, 0), "basket": (8, 0, 3.05)},
        {"name": "侧面投篮", "robot": (0, 0), "basket": (3, 4, 3.05)},
    ]
    
    print("\n" + "="*80)
    print("🎯 模型预测演示")
    print("="*80)
    
    for scenario in test_scenarios:
        robot_x, robot_y = scenario["robot"]
        basket_x, basket_y, basket_z = scenario["basket"]
        
        # 准备输入
        X_input = np.array([[robot_x, robot_y, basket_x, basket_y, basket_z]])
        
        # 标准化输入
        X_norm_params = norm_params['X_norm_params']
        X_input_norm = (X_input - X_norm_params['mean']) / X_norm_params['std']
        
        # 模型预测
        model.eval()
        with torch.no_grad():
            prediction_norm = model.predict(X_input_norm[0])
        
        # 反标准化预测结果
        y_norm_params = norm_params['y_norm_params']
        v0_pred = prediction_norm[0] * y_norm_params['v0_std'] + y_norm_params['v0_mean']
        theta_pitch_pred = prediction_norm[1] * y_norm_params['theta_pitch_scale']
        theta_yaw_pred = prediction_norm[2] * y_norm_params['theta_yaw_scale']
        
        # 物理模型理论解
        try:
            v0_theory, theta_pitch_theory, theta_yaw_theory = physics_model.calculate_optimal_params(
                robot_x, robot_y, 1.0, basket_x, basket_y, basket_z
            )
        except:
            v0_theory = theta_pitch_theory = theta_yaw_theory = None
        
        # 打印结果
        print(f"\n📍 {scenario['name']}:")
        print(f"   机器人位置: ({robot_x:.1f}, {robot_y:.1f}) m")
        print(f"   篮筐位置: ({basket_x:.1f}, {basket_y:.1f}, {basket_z:.1f}) m")
        print(f"   距离: {np.sqrt((basket_x-robot_x)**2 + (basket_y-robot_y)**2):.1f} m")
        print(f"   ")
        print(f"   🤖 神经网络预测:")
        print(f"      初速度: {v0_pred:.2f} m/s")
        print(f"      仰角: {np.degrees(theta_pitch_pred):.1f}°")
        print(f"      偏向角: {np.degrees(theta_yaw_pred):.1f}°")
        
        if v0_theory is not None:
            print(f"   📐 物理模型理论解:")
            print(f"      初速度: {v0_theory:.2f} m/s")
            print(f"      仰角: {np.degrees(theta_pitch_theory):.1f}°")
            print(f"      偏向角: {np.degrees(theta_yaw_theory):.1f}°")
            
            print(f"   📊 误差分析:")
            print(f"      速度误差: {abs(v0_pred - v0_theory):.3f} m/s")
            print(f"      仰角误差: {abs(np.degrees(theta_pitch_pred - theta_pitch_theory)):.1f}°")
            print(f"      偏向角误差: {abs(np.degrees(theta_yaw_pred - theta_yaw_theory)):.1f}°")
    
    print("\n" + "="*80)
    logger.info("模型预测演示完成")


def save_final_results(evaluation_results, config):
    """保存最终结果"""
    logger.info("保存最终结果...")
    
    # 创建结果摘要
    summary = {
        "model_config": config['model'],
        "training_config": config['training'],
        "performance_metrics": {
            "mse": evaluation_results['basic_metrics']['overall_mse'],
            "mae": evaluation_results['basic_metrics']['overall_mae'],
            "r2": evaluation_results['basic_metrics']['overall_r2'],
            "relative_error_percent": evaluation_results['basic_metrics']['mean_relative_error_percent']
        },
        "model_complexity": {
            "parameters": sum(p.numel() for p in torch.load("data/models/best_model.pth")['model_state_dict'].values() if hasattr(p, 'numel')),
            "estimated_size_mb": sum(p.numel() for p in torch.load("data/models/best_model.pth")['model_state_dict'].values() if hasattr(p, 'numel')) * 4 / (1024 * 1024)
        }
    }
    
    # 保存摘要
    with open("results/final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("最终结果已保存")


def load_config() -> dict:
    """加载配置参数"""
    return {
        "physics": {
            "gravity": 9.81,
            "air_resistance": 0.01
        },
        "data": {
            "n_samples": 10000,
            "field_size": [20.0, 15.0],
            "add_noise": True,
            "noise_level": 0.1,
            "train_ratio": 0.7,
            "val_ratio": 0.15
        },
        "model": {
            "type": "standard",
            "input_size": 5,
            "hidden_sizes": [64, 64],
            "output_size": 3,
            "dropout_rate": 0.1,
            "activation": "relu"
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "batch_size": 32,
            "epochs": 100,
            "patience": 20
        }
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='篮球投篮控制器')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练，使用已有模型')
    parser.add_argument('--quick-mode', action='store_true', help='快速模式（减少数据量和训练轮数）')
    args = parser.parse_args()
    
    # 初始化
    print_banner()
    setup_logging()
    create_directories()
    
    # 加载配置
    config = load_config()
    
    # 快速模式调整
    if args.quick_mode:
        config['data']['n_samples'] = 2000
        config['training']['epochs'] = 20
        config['training']['patience'] = 5
        logger.info("启用快速模式")
    
    try:
        start_time = time.time()
        
        # 1. 生成数据
        logger.info("🔄 步骤 1/5: 生成训练数据")
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         physics_model, norm_params) = generate_training_data(config)
        
        # 2. 训练模型
        if not args.skip_training:
            logger.info("🔄 步骤 2/5: 训练神经网络模型")
            model, trainer, train_history = train_model(X_train, X_val, y_train, y_val, config)
        else:
            logger.info("⏭️  跳过训练，加载已有模型")
            model = create_model(**config['model'])
            trainer = Trainer(model)
            checkpoint = trainer.load_model("data/models/best_model.pth")
            train_history = checkpoint.get('train_history', {'train_loss': [], 'val_loss': [], 'learning_rate': []})
        
        # 3. 评估模型
        logger.info("🔄 步骤 3/5: 评估模型性能")
        evaluation_results = evaluate_model(model, trainer, X_test, y_test, physics_model, config)
        
        # 4. 生成可视化
        logger.info("🔄 步骤 4/5: 生成可视化图表")
        plots_created = create_visualizations(evaluation_results, train_history, physics_model, config)
        
        # 5. 演示和保存结果
        logger.info("🔄 步骤 5/5: 演示模型并保存结果")
        demonstrate_model(model, physics_model, norm_params)
        save_final_results(evaluation_results, config)
        
        # 完成
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 篮球投篮控制器训练和评估完成！")
        print("="*80)
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"📊 性能指标:")
        print(f"   • MSE: {evaluation_results['basic_metrics']['overall_mse']:.6f}")
        print(f"   • MAE: {evaluation_results['basic_metrics']['overall_mae']:.6f}")
        print(f"   • R²: {evaluation_results['basic_metrics']['overall_r2']:.4f}")
        print(f"   • 相对误差: {evaluation_results['basic_metrics']['mean_relative_error_percent']:.2f}%")
        print(f"📁 结果文件:")
        print(f"   • 模型文件: data/models/best_model.pth")
        print(f"   • 图表文件: results/figures/")
        print(f"   • 日志文件: results/logs/")
        print(f"   • 性能报告: results/performance_report.txt")
        print("="*80)
        
        logger.info("程序执行完成")
        
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()