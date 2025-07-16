#!/usr/bin/env python3
"""快速测试脚本

用于验证篮球投篮控制器项目的基本功能
"""

import os
import sys
import numpy as np
import torch

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.physics_model import PhysicsModel
from src.data_generator import DataGenerator
from src.neural_network import BasketballNet, create_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer


def test_physics_model():
    """测试物理模型"""
    print("🔬 测试物理模型...")
    
    physics = PhysicsModel()
    
    # 测试轨迹计算
    robot_pos = (0, 0, 1.0)
    basket_pos = (5, 0, 3.05)
    
    try:
        v0, theta_pitch, theta_yaw = physics.calculate_optimal_params(*robot_pos, *basket_pos)
        print(f"   ✅ 最优参数计算成功: v0={v0:.2f}, pitch={np.degrees(theta_pitch):.1f}°, yaw={np.degrees(theta_yaw):.1f}°")
        
        t, x, y, z = physics.calculate_trajectory(*robot_pos, v0, theta_pitch, theta_yaw)
        print(f"   ✅ 轨迹计算成功: {len(t)} 个时间点")
        
        success = physics.check_shot_success(*robot_pos, v0, theta_pitch, theta_yaw, *basket_pos)
        print(f"   ✅ 投篮成功检查: {'成功' if success else '失败'}")
        
    except Exception as e:
        print(f"   ❌ 物理模型测试失败: {e}")
        return False
    
    return True


def test_data_generation():
    """测试数据生成"""
    print("📊 测试数据生成...")
    
    try:
        physics = PhysicsModel()
        generator = DataGenerator(physics)
        
        # 生成小量数据
        X, y = generator.generate_dataset(n_samples=100, field_size=(10, 10))
        print(f"   ✅ 数据生成成功: X.shape={X.shape}, y.shape={y.shape}")
        
        # 测试标准化
        X_norm, X_params = generator.normalize_features(X)
        y_norm, y_params = generator.normalize_targets(y)
        print(f"   ✅ 数据标准化成功")
        
        # 测试数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = generator.split_dataset(X_norm, y_norm)
        print(f"   ✅ 数据划分成功: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        
    except Exception as e:
        print(f"   ❌ 数据生成测试失败: {e}")
        return False
    
    return True


def test_neural_network():
    """测试神经网络"""
    print("🧠 测试神经网络...")
    
    try:
        # 创建模型
        model = create_model(
            model_type="standard",
            input_size=5,
            hidden_sizes=[32, 32],
            output_size=3
        )
        print(f"   ✅ 模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 测试前向传播
        x = torch.randn(10, 5)
        y = model(x)
        print(f"   ✅ 前向传播成功: 输入{x.shape} -> 输出{y.shape}")
        
        # 测试预测
        pred = model.predict(x[0].numpy())
        print(f"   ✅ 单样本预测成功: {pred}")
        
    except Exception as e:
        print(f"   ❌ 神经网络测试失败: {e}")
        return False
    
    return True


def test_training():
    """测试训练过程"""
    print("🏋️ 测试训练过程...")
    
    try:
        # 准备数据
        physics = PhysicsModel()
        generator = DataGenerator(physics)
        X, y = generator.generate_dataset(n_samples=200)
        X_norm, _ = generator.normalize_features(X)
        y_norm, _ = generator.normalize_targets(y)
        X_train, X_val, _, y_train, y_val, _ = generator.split_dataset(X_norm, y_norm)
        
        # 创建模型和训练器
        model = create_model(
            model_type="standard",
            input_size=5,
            hidden_sizes=[16, 16],
            output_size=3
        )
        trainer = Trainer(model, learning_rate=0.01)
        
        # 准备数据加载器
        train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_val, y_val, batch_size=16)
        print(f"   ✅ 数据加载器创建成功")
        
        # 短暂训练
        history = trainer.train(train_loader, val_loader, epochs=3, patience=10)
        print(f"   ✅ 训练完成: {len(history['train_loss'])} 轮")
        
    except Exception as e:
        print(f"   ❌ 训练测试失败: {e}")
        return False
    
    return True


def test_evaluation():
    """测试评估功能"""
    print("📈 测试评估功能...")
    
    try:
        # 准备数据和模型
        physics = PhysicsModel()
        generator = DataGenerator(physics)
        X, y = generator.generate_dataset(n_samples=100)
        X_norm, _ = generator.normalize_features(X)
        y_norm, _ = generator.normalize_targets(y)
        
        model = create_model(
            model_type="standard",
            input_size=5,
            hidden_sizes=[16, 16],
            output_size=3
        )
        evaluator = Evaluator(model, physics)
        
        # 基本评估
        metrics = evaluator.evaluate_basic_metrics(X_norm, y_norm)
        print(f"   ✅ 基本评估完成: MSE={metrics['overall_mse']:.6f}")
        
        # 物理一致性评估
        consistency = evaluator.evaluate_physics_consistency(X_norm, y_norm)
        print(f"   ✅ 物理一致性评估完成")
        
    except Exception as e:
        print(f"   ❌ 评估测试失败: {e}")
        return False
    
    return True


def test_visualization():
    """测试可视化功能"""
    print("📊 测试可视化功能...")
    
    try:
        visualizer = Visualizer()
        
        # 创建测试数据
        train_history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
            'learning_rate': [0.001] * 5
        }
        
        # 测试训练历史图
        plot_path = visualizer.plot_training_history(train_history, show=False, save=False)
        print(f"   ✅ 训练历史图创建成功")
        
        # 测试3D轨迹图
        robot_pos = (0, 0, 1)
        basket_pos = (5, 0, 3.05)
        trajectory = (np.linspace(0, 5, 50), np.zeros(50), np.linspace(1, 3.05, 50))
        
        plot_path = visualizer.plot_trajectory_3d(
            robot_pos, basket_pos, trajectory,
            show=False, save=False
        )
        print(f"   ✅ 3D轨迹图创建成功")
        
    except Exception as e:
        print(f"   ❌ 可视化测试失败: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("🧪 篮球投篮控制器 - 快速功能测试")
    print("=" * 50)
    
    tests = [
        ("物理模型", test_physics_model),
        ("数据生成", test_data_generation),
        ("神经网络", test_neural_network),
        ("训练过程", test_training),
        ("评估功能", test_evaluation),
        ("可视化", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} 测试通过\n")
            else:
                print(f"❌ {name} 测试失败\n")
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}\n")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)