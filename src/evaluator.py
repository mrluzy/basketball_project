"""评估模块

评估神经网络模型的性能，包括误差分析、可视化等功能
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import os

from .physics_model import PhysicsModel
from .neural_network import BasketballNet


class Evaluator:
    """模型评估器
    
    评估篮球投篮控制模型的性能
    """
    
    def __init__(self, 
                 model: nn.Module,
                 physics_model: PhysicsModel = None,
                 device: str = 'auto'):
        """
        初始化评估器
        
        Args:
            model: 训练好的神经网络模型
            physics_model: 物理模型实例
            device: 计算设备
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.physics_model = physics_model or PhysicsModel()
        
        logger.info(f"评估器初始化完成，使用设备: {self.device}")
    
    def evaluate_basic_metrics(self, 
                              X_test: np.ndarray, 
                              y_test: np.ndarray) -> Dict:
        """
        计算基本评估指标
        
        Args:
            X_test: 测试输入
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred_tensor = self.model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()
        
        # 计算各种指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # 计算R²分数
        r2 = r2_score(y_test, y_pred)
        
        # 分别计算每个输出的指标
        output_names = ['v0', 'theta_pitch', 'theta_yaw']
        individual_metrics = {}
        
        for i, name in enumerate(output_names):
            individual_metrics[f'{name}_mse'] = mean_squared_error(y_test[:, i], y_pred[:, i])
            individual_metrics[f'{name}_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
            individual_metrics[f'{name}_rmse'] = np.sqrt(individual_metrics[f'{name}_mse'])
            individual_metrics[f'{name}_r2'] = r2_score(y_test[:, i], y_pred[:, i])
        
        # 计算相对误差
        relative_errors = np.abs((y_pred - y_test) / (y_test + 1e-8)) * 100
        mean_relative_error = np.mean(relative_errors)
        
        results = {
            'overall_mse': mse,
            'overall_mae': mae,
            'overall_rmse': rmse,
            'overall_r2': r2,
            'mean_relative_error_percent': mean_relative_error,
            'predictions': y_pred,
            'targets': y_test,
            **individual_metrics
        }
        
        logger.info(f"基本指标计算完成 - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        
        return results
    
    def evaluate_physics_consistency(self, 
                                   X_test: np.ndarray, 
                                   y_test: np.ndarray,
                                   n_samples: int = 1000) -> Dict:
        """
        评估模型预测的物理一致性
        
        Args:
            X_test: 测试输入
            y_test: 测试目标
            n_samples: 评估样本数量
            
        Returns:
            物理一致性评估结果
        """
        # 随机选择样本
        if len(X_test) > n_samples:
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        else:
            X_sample = X_test
            y_sample = y_test
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sample).to(self.device)
            y_pred_tensor = self.model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()
        
        success_rates = []
        trajectory_errors = []
        
        for i in range(len(X_sample)):
            # 提取位置信息
            robot_x, robot_y, basket_x, basket_y, basket_z = X_sample[i]
            robot_z = 1.0  # 假设机器人高度
            
            # 模型预测的参数
            v0_pred, theta_pitch_pred, theta_yaw_pred = y_pred[i]
            
            # 真实的参数
            v0_true, theta_pitch_true, theta_yaw_true = y_sample[i]
            
            # 计算预测轨迹
            try:
                t_pred, x_pred, y_pred_traj, z_pred = self.physics_model.calculate_trajectory(
                    robot_x, robot_y, robot_z, v0_pred, theta_pitch_pred, theta_yaw_pred
                )
                
                # 检查是否成功投篮
                success_pred = self.physics_model.check_trajectory_success(
                    x_pred, y_pred_traj, z_pred, basket_x, basket_y, basket_z
                )
                
                # 计算真实轨迹
                t_true, x_true, y_true_traj, z_true = self.physics_model.calculate_trajectory(
                    robot_x, robot_y, robot_z, v0_true, theta_pitch_true, theta_yaw_true
                )
                
                success_true = self.physics_model.check_trajectory_success(
                    x_true, y_true_traj, z_true, basket_x, basket_y, basket_z
                )
                
                success_rates.append(1 if success_pred == success_true else 0)
                
                # 计算轨迹误差（在篮筐高度处）
                if len(z_pred) > 1 and len(z_true) > 1:
                    # 找到最接近篮筐高度的点
                    idx_pred = np.argmin(np.abs(z_pred - basket_z))
                    idx_true = np.argmin(np.abs(z_true - basket_z))
                    
                    error = math.sqrt((x_pred[idx_pred] - x_true[idx_true])**2 + 
                                    (y_pred_traj[idx_pred] - y_true_traj[idx_true])**2)
                    trajectory_errors.append(error)
                
            except Exception as e:
                logger.warning(f"轨迹计算失败: {e}")
                continue
        
        results = {
            'success_rate_consistency': np.mean(success_rates) if success_rates else 0,
            'mean_trajectory_error': np.mean(trajectory_errors) if trajectory_errors else float('inf'),
            'std_trajectory_error': np.std(trajectory_errors) if trajectory_errors else 0,
            'valid_trajectories': len(trajectory_errors)
        }
        
        logger.info(f"物理一致性评估完成 - 成功率一致性: {results['success_rate_consistency']:.3f}")
        
        return results
    
    def evaluate_robustness(self, 
                           X_test: np.ndarray, 
                           noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """
        评估模型的鲁棒性（对输入噪声的敏感性）
        
        Args:
            X_test: 测试输入
            noise_levels: 噪声水平列表
            
        Returns:
            鲁棒性评估结果
        """
        self.model.eval()
        
        # 原始预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_original = self.model(X_tensor).cpu().numpy()
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            prediction_variations = []
            
            # 多次添加噪声并预测
            for _ in range(10):  # 10次重复
                # 添加高斯噪声
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise
                
                # 预测
                with torch.no_grad():
                    X_noisy_tensor = torch.FloatTensor(X_noisy).to(self.device)
                    y_noisy = self.model(X_noisy_tensor).cpu().numpy()
                
                # 计算与原始预测的差异
                variation = np.mean(np.abs(y_noisy - y_original))
                prediction_variations.append(variation)
            
            robustness_results[f'noise_{noise_level}'] = {
                'mean_variation': np.mean(prediction_variations),
                'std_variation': np.std(prediction_variations),
                'max_variation': np.max(prediction_variations)
            }
        
        logger.info(f"鲁棒性评估完成，测试了 {len(noise_levels)} 个噪声水平")
        
        return robustness_results
    
    def evaluate_edge_cases(self, field_size: Tuple[float, float] = (20.0, 15.0)) -> Dict:
        """
        评估边界情况下的模型性能
        
        Args:
            field_size: 场地大小
            
        Returns:
            边界情况评估结果
        """
        edge_cases = [
            # 极近距离
            {'robot_pos': (0, 0, 1.0), 'basket_pos': (1, 0, 3.05), 'name': '极近距离'},
            # 极远距离
            {'robot_pos': (0, 0, 1.0), 'basket_pos': (15, 0, 3.05), 'name': '极远距离'},
            # 极大角度
            {'robot_pos': (0, 0, 1.0), 'basket_pos': (2, 8, 3.05), 'name': '极大角度'},
            # 高篮筐
            {'robot_pos': (0, 0, 1.0), 'basket_pos': (5, 0, 4.0), 'name': '高篮筐'},
            # 低篮筐
            {'robot_pos': (0, 0, 1.0), 'basket_pos': (5, 0, 2.0), 'name': '低篮筐'},
            # 场地边缘
            {'robot_pos': (-field_size[0]/2+1, -field_size[1]/2+1, 1.0), 
             'basket_pos': (field_size[0]/2-1, field_size[1]/2-1, 3.05), 'name': '场地边缘'}
        ]
        
        results = {}
        
        for case in edge_cases:
            robot_x, robot_y, robot_z = case['robot_pos']
            basket_x, basket_y, basket_z = case['basket_pos']
            
            # 准备输入
            X_input = np.array([[robot_x, robot_y, basket_x, basket_y, basket_z]])
            
            # 模型预测
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_input).to(self.device)
                y_pred = self.model(X_tensor).cpu().numpy()[0]
            
            # 物理模型计算理论值
            try:
                v0_theory, theta_pitch_theory, theta_yaw_theory = \
                    self.physics_model.calculate_optimal_params(
                        robot_x, robot_y, robot_z, basket_x, basket_y, basket_z
                    )
                
                # 计算误差
                error_v0 = abs(y_pred[0] - v0_theory)
                error_pitch = abs(y_pred[1] - theta_pitch_theory)
                error_yaw = abs(y_pred[2] - theta_yaw_theory)
                
                # 检查预测的物理合理性
                is_reasonable = (
                    y_pred[0] > 0 and y_pred[0] < 50 and  # 速度合理
                    y_pred[1] > 0 and y_pred[1] < math.pi/2 and  # 仰角合理
                    abs(y_pred[2]) < math.pi  # 偏向角合理
                )
                
                results[case['name']] = {
                    'prediction': y_pred.tolist(),
                    'theory': [v0_theory, theta_pitch_theory, theta_yaw_theory],
                    'errors': [error_v0, error_pitch, error_yaw],
                    'is_reasonable': is_reasonable
                }
                
            except Exception as e:
                results[case['name']] = {
                    'prediction': y_pred.tolist(),
                    'theory': None,
                    'errors': None,
                    'is_reasonable': False,
                    'error_message': str(e)
                }
        
        logger.info(f"边界情况评估完成，测试了 {len(edge_cases)} 个场景")
        
        return results
    
    def compare_with_physics(self, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray,
                           n_samples: int = 500) -> Dict:
        """
        与物理模型的理论解进行比较
        
        Args:
            X_test: 测试输入
            y_test: 测试目标
            n_samples: 比较样本数量
            
        Returns:
            比较结果
        """
        # 随机选择样本
        if len(X_test) > n_samples:
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test[indices]
        else:
            X_sample = X_test
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sample).to(self.device)
            y_pred = self.model(X_tensor).cpu().numpy()
        
        # 物理模型计算
        y_physics = []
        valid_indices = []
        
        for i, x in enumerate(X_sample):
            robot_x, robot_y, basket_x, basket_y, basket_z = x
            robot_z = 1.0  # 假设机器人高度
            
            try:
                v0, theta_pitch, theta_yaw = self.physics_model.calculate_optimal_params(
                    robot_x, robot_y, robot_z, basket_x, basket_y, basket_z
                )
                y_physics.append([v0, theta_pitch, theta_yaw])
                valid_indices.append(i)
            except Exception:
                continue
        
        if not y_physics:
            logger.warning("没有有效的物理模型计算结果")
            return {}
        
        y_physics = np.array(y_physics)
        y_pred_valid = y_pred[valid_indices]
        
        # 计算比较指标
        physics_comparison = {
            'correlation_v0': np.corrcoef(y_pred_valid[:, 0], y_physics[:, 0])[0, 1],
            'correlation_pitch': np.corrcoef(y_pred_valid[:, 1], y_physics[:, 1])[0, 1],
            'correlation_yaw': np.corrcoef(y_pred_valid[:, 2], y_physics[:, 2])[0, 1],
            'mse_vs_physics': mean_squared_error(y_physics, y_pred_valid),
            'mae_vs_physics': mean_absolute_error(y_physics, y_pred_valid),
            'predictions': y_pred_valid,
            'physics_solutions': y_physics
        }
        
        logger.info(f"物理模型比较完成，有效样本: {len(y_physics)}")
        
        return physics_comparison
    
    def generate_performance_report(self, 
                                  X_test: np.ndarray, 
                                  y_test: np.ndarray,
                                  save_path: str = "results/performance_report.txt") -> Dict:
        """
        生成完整的性能评估报告
        
        Args:
            X_test: 测试输入
            y_test: 测试目标
            save_path: 报告保存路径
            
        Returns:
            完整评估结果
        """
        logger.info("开始生成性能评估报告...")
        
        # 各项评估
        basic_metrics = self.evaluate_basic_metrics(X_test, y_test)
        physics_consistency = self.evaluate_physics_consistency(X_test, y_test)
        robustness = self.evaluate_robustness(X_test)
        edge_cases = self.evaluate_edge_cases()
        physics_comparison = self.compare_with_physics(X_test, y_test)
        
        # 汇总结果
        full_report = {
            'basic_metrics': basic_metrics,
            'physics_consistency': physics_consistency,
            'robustness': robustness,
            'edge_cases': edge_cases,
            'physics_comparison': physics_comparison
        }
        
        # 保存报告
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("篮球投篮控制器性能评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本指标
            f.write("1. 基本性能指标\n")
            f.write("-" * 20 + "\n")
            f.write(f"总体MSE: {basic_metrics['overall_mse']:.6f}\n")
            f.write(f"总体MAE: {basic_metrics['overall_mae']:.6f}\n")
            f.write(f"总体R²: {basic_metrics['overall_r2']:.4f}\n")
            f.write(f"平均相对误差: {basic_metrics['mean_relative_error_percent']:.2f}%\n\n")
            
            # 各输出指标
            f.write("各输出参数性能:\n")
            for param in ['v0', 'theta_pitch', 'theta_yaw']:
                f.write(f"  {param} - MSE: {basic_metrics[f'{param}_mse']:.6f}, ")
                f.write(f"R²: {basic_metrics[f'{param}_r2']:.4f}\n")
            f.write("\n")
            
            # 物理一致性
            f.write("2. 物理一致性\n")
            f.write("-" * 20 + "\n")
            f.write(f"成功率一致性: {physics_consistency['success_rate_consistency']:.3f}\n")
            f.write(f"平均轨迹误差: {physics_consistency['mean_trajectory_error']:.3f}m\n\n")
            
            # 鲁棒性
            f.write("3. 鲁棒性分析\n")
            f.write("-" * 20 + "\n")
            for noise_level, metrics in robustness.items():
                f.write(f"噪声水平 {noise_level}: 平均变化 {metrics['mean_variation']:.6f}\n")
            f.write("\n")
            
            # 边界情况
            f.write("4. 边界情况分析\n")
            f.write("-" * 20 + "\n")
            for case_name, case_result in edge_cases.items():
                f.write(f"{case_name}: 合理性 {case_result['is_reasonable']}\n")
            f.write("\n")
        
        logger.info(f"性能评估报告已保存到: {save_path}")
        
        return full_report