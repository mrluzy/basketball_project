"""可视化模块

生成训练过程和结果的可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from loguru import logger
import math

# 设置中文字体和样式
# 基于系统检测的可用字体
import platform
system = platform.system()

# 使用检测到的中文字体
available_fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = available_fonts
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:
    """可视化工具类
    
    生成各种图表用于分析和展示结果
    """
    
    def __init__(self, save_dir: str = "results/figures"):
        """
        初始化可视化器
        
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 重新设置字体配置，确保中文显示正常
        # 使用检测到的可用中文字体
        available_fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans']
        plt.rcParams['font.sans-serif'] = available_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 清除matplotlib字体缓存
        try:
            import matplotlib.font_manager as fm
            fm._rebuild()
        except:
            pass
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        logger.info(f"可视化器初始化完成，图片将保存到: {save_dir}")
    
    def plot_training_history(self, 
                            train_history: Dict,
                            show: bool = True,
                            save: bool = True) -> str:
        """
        绘制训练历史曲线
        
        Args:
            train_history: 训练历史数据
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Process Analysis', fontsize=16, fontweight='bold')
        
        # 训练和验证损失
        axes[0, 0].plot(train_history['train_loss'], label='Training Loss', color=self.colors[0], linewidth=2)
        axes[0, 0].plot(train_history['val_loss'], label='Validation Loss', color=self.colors[1], linewidth=2)
        axes[0, 0].set_title('Loss Function Changes')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[0, 1].plot(train_history['learning_rate'], color=self.colors[2], linewidth=2)
        axes[0, 1].set_title('Learning Rate Changes')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 损失函数对数图
        axes[1, 0].semilogy(train_history['train_loss'], label='Training Loss', color=self.colors[0], linewidth=2)
        axes[1, 0].semilogy(train_history['val_loss'], label='Validation Loss', color=self.colors[1], linewidth=2)
        axes[1, 0].set_title('Loss Function Changes (Log Scale)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 过拟合分析
        if len(train_history['train_loss']) > 10:
            train_smooth = pd.Series(train_history['train_loss']).rolling(window=5).mean()
            val_smooth = pd.Series(train_history['val_loss']).rolling(window=5).mean()
            
            axes[1, 1].plot(train_smooth, label='Training Loss (Smooth)', color=self.colors[0], linewidth=2)
            axes[1, 1].plot(val_smooth, label='Validation Loss (Smooth)', color=self.colors[1], linewidth=2)
            
            # 计算过拟合指标
            if len(val_smooth.dropna()) > 0:
                overfitting = val_smooth.dropna().iloc[-1] - train_smooth.dropna().iloc[-1]
                axes[1, 1].text(0.05, 0.95, f'Overfitting: {overfitting:.6f}', 
                               transform=axes[1, 1].transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1, 1].set_title('Smoothed Loss Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'training_history.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Training history chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def plot_prediction_comparison(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 show: bool = True,
                                 save: bool = True) -> str:
        """
        绘制预测值与真实值的比较
        
        Args:
            predictions: 预测值
            targets: 真实值
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        param_names = ['Initial Velocity v₀ (m/s)', 'Pitch Angle θ_pitch (rad)', 'Yaw Angle θ_yaw (rad)']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Prediction vs Ground Truth Comparison', fontsize=16, fontweight='bold')
        
        for i, param_name in enumerate(param_names):
            # 散点图
            axes[0, i].scatter(targets[:, i], predictions[:, i], alpha=0.6, s=20, color=self.colors[i])
            
            # 理想线 (y=x)
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Line')
            
            # 计算R²
            correlation = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
            r_squared = correlation ** 2
            
            axes[0, i].set_title(f'{param_name}\nR² = {r_squared:.4f}')
            axes[0, i].set_xlabel('Ground Truth')
            axes[0, i].set_ylabel('Prediction')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 误差分布直方图
            errors = predictions[:, i] - targets[:, i]
            axes[1, i].hist(errors, bins=50, alpha=0.7, color=self.colors[i], edgecolor='black')
            axes[1, i].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, i].set_title(f'{param_name} Error Distribution\nMean: {np.mean(errors):.6f}, Std: {np.std(errors):.6f}')
            axes[1, i].set_xlabel('Error (Prediction - Ground Truth)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'prediction_comparison.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction comparison chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def plot_error_analysis(self, 
                          evaluation_results: Dict,
                          show: bool = True,
                          save: bool = True) -> str:
        """
        绘制误差分析图表
        
        Args:
            evaluation_results: 评估结果
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        # 各参数MSE对比
        params = ['v0', 'theta_pitch', 'theta_yaw']
        param_names = ['Initial Velocity', 'Pitch Angle', 'Yaw Angle']
        mse_values = [evaluation_results[f'{param}_mse'] for param in params]
        
        bars = axes[0, 0].bar(param_names, mse_values, color=self.colors[:3])
        axes[0, 0].set_title('MSE Comparison by Parameter')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, mse_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                           f'{value:.6f}', ha='center', va='bottom')
        
        # R²分数对比
        r2_values = [evaluation_results[f'{param}_r2'] for param in params]
        bars = axes[0, 1].bar(param_names, r2_values, color=self.colors[3:6])
        axes[0, 1].set_title('R² Score Comparison by Parameter')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, r2_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # 误差热力图
        if 'predictions' in evaluation_results and 'targets' in evaluation_results:
            predictions = evaluation_results['predictions']
            targets = evaluation_results['targets']
            errors = np.abs(predictions - targets)
            
            # 计算误差的百分位数
            percentiles = [50, 75, 90, 95, 99]
            error_percentiles = np.percentile(errors.flatten(), percentiles)
            
            axes[1, 0].bar(range(len(percentiles)), error_percentiles, color=self.colors[6:6+len(percentiles)])
            axes[1, 0].set_title('Error Percentile Distribution')
            axes[1, 0].set_xlabel('Percentile')
            axes[1, 0].set_ylabel('Absolute Error')
            axes[1, 0].set_xticks(range(len(percentiles)))
            axes[1, 0].set_xticklabels([f'{p}%' for p in percentiles])
            axes[1, 0].grid(True, alpha=0.3)
        
        # 性能指标雷达图
        metrics = ['MSE', 'MAE', 'R²']
        values = [
            1 - min(evaluation_results['overall_mse'], 1),  # 归一化MSE
            1 - min(evaluation_results['overall_mae'], 1),  # 归一化MAE
            evaluation_results['overall_r2']  # R²
        ]
        
        # 雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
        axes[1, 1].plot(angles, values, 'o-', linewidth=2, color=self.colors[0])
        axes[1, 1].fill(angles, values, alpha=0.25, color=self.colors[0])
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Comprehensive Performance Radar Chart', pad=20)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'error_analysis.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def plot_trajectory_3d(self, 
                          robot_pos: Tuple[float, float, float],
                          basket_pos: Tuple[float, float, float],
                          trajectory: Tuple[np.ndarray, np.ndarray, np.ndarray],
                          title: str = "Basketball Shot Trajectory",
                          show: bool = True,
                          save: bool = True) -> str:
        """
        绘制简化的3D轨迹图
        
        Args:
            robot_pos: 机器人位置
            basket_pos: 篮筐位置
            trajectory: 轨迹坐标 (x, y, z)
            title: 图表标题
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x_traj, y_traj, z_traj = trajectory
        
        # 绘制简单的轨迹线
        ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')
        
        # 绘制起点和终点
        ax.scatter(*robot_pos, color='green', s=80, label='Start')
        ax.scatter(*basket_pos, color='red', s=80, label='Target')
        
        # 在轨迹上显示篮球位置（每隔几个点显示一个篮球）
        ball_indices = range(0, len(x_traj), max(1, len(x_traj)//8))  # 显示8个篮球位置
        for i in ball_indices[1:-1]:  # 跳过起点和终点
            ax.scatter(x_traj[i], y_traj[i], z_traj[i], color='orange', s=30, alpha=0.7)
        
        # 移除篮筐圆圈，只保留抛物线轨迹
        
        # 简化的标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # 简化的坐标轴设置
        ax.set_xlim(min(x_traj)-1, max(x_traj)+1)
        ax.set_ylim(min(y_traj)-1, max(y_traj)+1)
        ax.set_zlim(0, max(z_traj)+1)
        
        # 简单网格
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'trajectory_3d.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"3D trajectory chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def plot_robustness_analysis(self, 
                               robustness_results: Dict,
                               show: bool = True,
                               save: bool = True) -> str:
        """
        绘制鲁棒性分析图表
        
        Args:
            robustness_results: 鲁棒性评估结果
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Robustness Analysis', fontsize=16, fontweight='bold')
        
        # 提取数据
        noise_levels = []
        mean_variations = []
        std_variations = []
        
        for key, value in robustness_results.items():
            if key.startswith('noise_'):
                noise_level = float(key.split('_')[1])
                noise_levels.append(noise_level)
                mean_variations.append(value['mean_variation'])
                std_variations.append(value['std_variation'])
        
        # 排序
        sorted_indices = np.argsort(noise_levels)
        noise_levels = np.array(noise_levels)[sorted_indices]
        mean_variations = np.array(mean_variations)[sorted_indices]
        std_variations = np.array(std_variations)[sorted_indices]
        
        # 绘制均值变化
        axes[0].errorbar(noise_levels, mean_variations, yerr=std_variations, 
                        marker='o', linewidth=2, markersize=8, capsize=5, color=self.colors[0])
        axes[0].set_title('Prediction Variation vs Input Noise')
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Prediction Variation')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # 绘制鲁棒性指标
        robustness_score = 1 / (1 + mean_variations)  # 简单的鲁棒性评分
        axes[1].plot(noise_levels, robustness_score, 'o-', linewidth=2, markersize=8, color=self.colors[1])
        axes[1].set_title('Robustness Score')
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Robustness Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'robustness_analysis.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Robustness analysis chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def plot_model_comparison(self, 
                            physics_comparison: Dict,
                            show: bool = True,
                            save: bool = True) -> str:
        """
        绘制模型与物理解的比较
        
        Args:
            physics_comparison: 物理模型比较结果
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        if not physics_comparison:
            logger.warning("No physics model comparison data available")
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Neural Network vs Physics Model Comparison', fontsize=16, fontweight='bold')
        
        predictions = physics_comparison['predictions']
        physics_solutions = physics_comparison['physics_solutions']
        param_names = ['Initial Velocity v₀ (m/s)', 'Pitch Angle θ_pitch (rad)', 'Yaw Angle θ_yaw (rad)']
        
        for i, param_name in enumerate(param_names):
            # 散点图对比
            axes[0, i].scatter(physics_solutions[:, i], predictions[:, i], 
                             alpha=0.6, s=20, color=self.colors[i])
            
            # 理想线
            min_val = min(physics_solutions[:, i].min(), predictions[:, i].min())
            max_val = max(physics_solutions[:, i].max(), predictions[:, i].max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Line')
            
            # 计算相关系数
            correlation = physics_comparison[f'correlation_{["v0", "pitch", "yaw"][i]}']
            
            axes[0, i].set_title(f'{param_names[i]}\nCorrelation: {correlation:.4f}')
            axes[0, i].set_xlabel('Physics Model Solution')
            axes[0, i].set_ylabel('Neural Network Prediction')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 误差分析
            errors = predictions[:, i] - physics_solutions[:, i]
            axes[1, i].hist(errors, bins=30, alpha=0.7, color=self.colors[i], edgecolor='black')
            axes[1, i].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, i].set_title(f'{param_names[i]} Error Distribution\nMean: {np.mean(errors):.6f}')
            axes[1, i].set_xlabel('Error (NN - Physics)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'model_comparison.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison chart saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath
    
    def create_summary_dashboard(self, 
                               evaluation_results: Dict,
                               train_history: Dict,
                               show: bool = True,
                               save: bool = True) -> str:
        """
        创建综合仪表板
        
        Args:
            evaluation_results: 评估结果
            train_history: 训练历史
            show: 是否显示图表
            save: 是否保存图表
            
        Returns:
            保存的文件路径
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Basketball Shooting Controller - Comprehensive Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 训练损失
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(train_history['train_loss'], label='Training', color=self.colors[0])
        ax1.plot(train_history['val_loss'], label='Validation', color=self.colors[1])
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 性能指标
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['MSE', 'MAE', 'R²']
        values = [evaluation_results['overall_mse'], 
                 evaluation_results['overall_mae'],
                 evaluation_results['overall_r2']]
        bars = ax2.bar(metrics, values, color=self.colors[:3])
        ax2.set_title('Key Metrics')
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 3. 各参数R²对比
        ax3 = fig.add_subplot(gs[0, 2])
        params = ['v0', 'theta_pitch', 'theta_yaw']
        param_names = ['Initial Velocity', 'Pitch Angle', 'Yaw Angle']
        r2_values = [evaluation_results[f'{param}_r2'] for param in params]
        ax3.bar(param_names, r2_values, color=self.colors[3:6])
        ax3.set_title('R² Score by Parameter')
        ax3.set_ylim(0, 1)
        
        # 4. 学习率变化
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(train_history['learning_rate'], color=self.colors[6])
        ax4.set_title('Learning Rate Changes')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # 5-7. 预测vs真实值散点图
        if 'predictions' in evaluation_results and 'targets' in evaluation_results:
            predictions = evaluation_results['predictions']
            targets = evaluation_results['targets']
            
            for i, param_name in enumerate(['Initial Velocity', 'Pitch Angle', 'Yaw Angle']):
                ax = fig.add_subplot(gs[1, i])
                ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10, color=self.colors[i])
                
                min_val = min(targets[:, i].min(), predictions[:, i].min())
                max_val = max(targets[:, i].max(), predictions[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                correlation = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
                ax.set_title(f'{param_name}\nR={correlation:.3f}')
                ax.set_xlabel('Ground Truth')
                ax.set_ylabel('Prediction')
        
        # 8. 模型复杂度信息
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        info_text = f"""
Model Information:
• Total Parameters: {evaluation_results.get('total_parameters', 'N/A')}
• MSE: {evaluation_results['overall_mse']:.6f}
• MAE: {evaluation_results['overall_mae']:.6f}
• R²: {evaluation_results['overall_r2']:.4f}

        """
        ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 9-11. 误差分布直方图
        if 'predictions' in evaluation_results and 'targets' in evaluation_results:
            for i, param_name in enumerate(['Initial Velocity Error', 'Pitch Angle Error', 'Yaw Angle Error']):
                ax = fig.add_subplot(gs[2, i])
                errors = predictions[:, i] - targets[:, i]
                ax.hist(errors, bins=30, alpha=0.7, color=self.colors[i], edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{param_name}\nμ={np.mean(errors):.4f}')
                ax.set_xlabel('Error')
                ax.set_ylabel('Frequency')
        
        # 12. 综合评分
        ax12 = fig.add_subplot(gs[2, 3])
        # 计算综合评分
        score = (evaluation_results['overall_r2'] * 0.4 + 
                (1 - min(evaluation_results['overall_mse'], 1)) * 0.3 +
                (1 - min(evaluation_results['overall_mae'], 1)) * 0.3) * 100
        
        # 绘制评分表盘
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        ax12 = plt.subplot(gs[2, 3], projection='polar')
        ax12.plot(theta, r, 'k-', linewidth=3)
        ax12.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # 评分指针
        score_angle = np.pi * (1 - score/100)
        ax12.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=5)
        ax12.set_ylim(0, 1)
        ax12.set_theta_zero_location('N')
        ax12.set_theta_direction(-1)
        ax12.set_title(f'Overall Score\n{score:.1f}/100', pad=20)
        
        # 保存图片
        filepath = None
        if save:
            filepath = os.path.join(self.save_dir, 'summary_dashboard.png')
            # 强制使用中文字体保存
            with plt.rc_context({'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']}):
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Summary dashboard saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filepath


def create_visualizer(save_dir: str = "results/figures") -> Visualizer:
    """
    创建可视化器的工厂函数
    
    Args:
        save_dir: 保存目录
        
    Returns:
        可视化器实例
    """
    return Visualizer(save_dir)