"""数据生成模块

生成用于训练神经网络的数据集，包括输入特征和目标输出
"""

import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
from tqdm import tqdm
import os

from .physics_model import PhysicsModel


class DataGenerator:
    """数据生成器
    
    生成机器人篮球投篮的训练数据
    """
    
    def __init__(self, physics_model: PhysicsModel = None):
        """
        初始化数据生成器
        
        Args:
            physics_model: 物理模型实例
        """
        self.physics_model = physics_model or PhysicsModel()
        
    def generate_dataset(self, 
                        n_samples: int = 10000,
                        field_size: Tuple[float, float] = (20.0, 15.0),
                        robot_height_range: Tuple[float, float] = (0.5, 1.5),
                        basket_height_range: Tuple[float, float] = (2.8, 3.2),
                        add_noise: bool = True,
                        noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据集
        
        Args:
            n_samples: 样本数量
            field_size: 场地大小 (长, 宽)
            robot_height_range: 机器人高度范围
            basket_height_range: 篮筐高度范围
            add_noise: 是否添加噪声
            noise_level: 噪声水平
            
        Returns:
            输入特征数组, 目标输出数组
        """
        print(f"生成 {n_samples} 个训练样本...")
        
        # 初始化数据数组
        X = np.zeros((n_samples, 5))  # [robot_x, robot_y, basket_x, basket_y, basket_height]
        y = np.zeros((n_samples, 3))  # [v0, theta_pitch, theta_yaw]
        
        valid_samples = 0
        attempts = 0
        max_attempts = n_samples * 3  # 最大尝试次数
        
        with tqdm(total=n_samples, desc="生成数据") as pbar:
            while valid_samples < n_samples and attempts < max_attempts:
                attempts += 1
                
                # 随机生成机器人位置
                robot_x = np.random.uniform(-field_size[0]/2, field_size[0]/2)
                robot_y = np.random.uniform(-field_size[1]/2, field_size[1]/2)
                robot_z = np.random.uniform(*robot_height_range)
                
                # 随机生成篮筐位置
                basket_x = np.random.uniform(-field_size[0]/2, field_size[0]/2)
                basket_y = np.random.uniform(-field_size[1]/2, field_size[1]/2)
                basket_z = np.random.uniform(*basket_height_range)
                
                # 确保机器人和篮筐不在同一位置
                distance = math.sqrt((robot_x - basket_x)**2 + (robot_y - basket_y)**2)
                if distance < 1.0:  # 最小距离1米
                    continue
                    
                # 计算理论最优参数
                try:
                    v0, theta_pitch, theta_yaw = self.physics_model.calculate_optimal_params(
                        robot_x, robot_y, robot_z, basket_x, basket_y, basket_z
                    )
                    
                    # 检查参数是否合理
                    if v0 > 50 or v0 < 1:  # 速度范围检查
                        continue
                        
                    if theta_pitch > math.pi/2 or theta_pitch < 0:  # 角度范围检查
                        continue
                    
                    # 添加噪声（如果需要）
                    if add_noise:
                        v0, theta_pitch, theta_yaw = self.physics_model.add_noise(
                            v0, theta_pitch, theta_yaw, 
                            v0_noise=noise_level, angle_noise=noise_level*0.1
                        )
                    
                    # 验证轨迹是否成功
                    t, traj_x, traj_y, traj_z = self.physics_model.calculate_trajectory(
                        robot_x, robot_y, robot_z, v0, theta_pitch, theta_yaw
                    )
                    
                    success = self.physics_model.check_trajectory_success(
                        traj_x, traj_y, traj_z, basket_x, basket_y, basket_z
                    )
                    
                    # 只保留成功的样本（可选：也可以保留失败样本用于对比学习）
                    if success or np.random.random() < 0.3:  # 30%概率保留失败样本
                        # 保存数据
                        X[valid_samples] = [robot_x, robot_y, basket_x, basket_y, basket_z]
                        y[valid_samples] = [v0, theta_pitch, theta_yaw]
                        
                        valid_samples += 1
                        pbar.update(1)
                        
                except Exception as e:
                    # 跳过计算失败的样本
                    continue
        
        if valid_samples < n_samples:
            print(f"警告: 只生成了 {valid_samples} 个有效样本，目标是 {n_samples} 个")
            X = X[:valid_samples]
            y = y[:valid_samples]
            
        return X, y
    
    def generate_test_scenarios(self) -> List[Dict]:
        """
        生成特定的测试场景
        
        Returns:
            测试场景列表
        """
        scenarios = [
            {
                'name': '近距离投篮',
                'robot_pos': (0, 0, 1.0),
                'basket_pos': (2, 0, 3.05),
                'description': '机器人距离篮筐2米的近距离投篮'
            },
            {
                'name': '中距离投篮', 
                'robot_pos': (0, 0, 1.0),
                'basket_pos': (5, 0, 3.05),
                'description': '机器人距离篮筐5米的中距离投篮'
            },
            {
                'name': '远距离投篮',
                'robot_pos': (0, 0, 1.0), 
                'basket_pos': (8, 0, 3.05),
                'description': '机器人距离篮筐8米的远距离投篮'
            },
            {
                'name': '侧面投篮',
                'robot_pos': (0, 0, 1.0),
                'basket_pos': (3, 4, 3.05),
                'description': '机器人从侧面角度投篮'
            },
            {
                'name': '高篮筐投篮',
                'robot_pos': (0, 0, 1.0),
                'basket_pos': (4, 0, 3.5),
                'description': '投篮到较高的篮筐'
            },
            {
                'name': '低篮筐投篮',
                'robot_pos': (0, 0, 1.0),
                'basket_pos': (4, 0, 2.5),
                'description': '投篮到较低的篮筐'
            }
        ]
        
        return scenarios
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        标准化输入特征
        
        Args:
            X: 输入特征数组
            
        Returns:
            标准化后的特征数组, 标准化参数字典
        """
        # 计算均值和标准差
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        # 避免除零
        std = np.where(std == 0, 1, std)
        
        # 标准化
        X_normalized = (X - mean) / std
        
        # 保存标准化参数
        norm_params = {
            'mean': mean,
            'std': std
        }
        
        return X_normalized, norm_params
    
    def normalize_targets(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        标准化目标输出
        
        Args:
            y: 目标输出数组
            
        Returns:
            标准化后的目标数组, 标准化参数字典
        """
        # 对不同的输出使用不同的标准化策略
        y_normalized = np.copy(y)
        
        # 速度标准化 (v0)
        v0_mean = np.mean(y[:, 0])
        v0_std = np.std(y[:, 0])
        y_normalized[:, 0] = (y[:, 0] - v0_mean) / v0_std
        
        # 角度标准化 (theta_pitch, theta_yaw)
        # 角度通常不需要标准化，但可以归一化到[-1, 1]
        y_normalized[:, 1] = y[:, 1] / (math.pi/2)  # 仰角归一化到[0, 1]
        y_normalized[:, 2] = y[:, 2] / math.pi      # 偏向角归一化到[-1, 1]
        
        norm_params = {
            'v0_mean': v0_mean,
            'v0_std': v0_std,
            'theta_pitch_scale': math.pi/2,
            'theta_yaw_scale': math.pi
        }
        
        return y_normalized, norm_params
    
    def _normalize_targets_with_params(self, y: np.ndarray, norm_params: Dict) -> np.ndarray:
        """
        使用已有参数标准化目标输出
        
        Args:
            y: 目标输出数组
            norm_params: 标准化参数字典
            
        Returns:
            标准化后的目标数组
        """
        y_normalized = np.copy(y)
        
        # 速度标准化 (v0)
        y_normalized[:, 0] = (y[:, 0] - norm_params['v0_mean']) / norm_params['v0_std']
        
        # 角度标准化 (theta_pitch, theta_yaw)
        y_normalized[:, 1] = y[:, 1] / norm_params['theta_pitch_scale']
        y_normalized[:, 2] = y[:, 2] / norm_params['theta_yaw_scale']
        
        return y_normalized
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, 
                    filename: str, data_dir: str = "data/processed"):
        """
        保存数据集到文件
        
        Args:
            X: 输入特征
            y: 目标输出
            filename: 文件名
            data_dir: 数据目录
        """
        os.makedirs(data_dir, exist_ok=True)
        
        # 合并数据
        data = np.hstack([X, y])
        
        # 创建DataFrame
        columns = ['robot_x', 'robot_y', 'basket_x', 'basket_y', 'basket_height',
                  'v0', 'theta_pitch', 'theta_yaw']
        df = pd.DataFrame(data, columns=columns)
        
        # 保存为CSV
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"数据集已保存到: {filepath}")
        
    def load_dataset(self, filename: str, data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray]:
        """
        从文件加载数据集
        
        Args:
            filename: 文件名
            data_dir: 数据目录
            
        Returns:
            输入特征, 目标输出
        """
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        
        X = df[['robot_x', 'robot_y', 'basket_x', 'basket_y', 'basket_height']].values
        y = df[['v0', 'theta_pitch', 'theta_yaw']].values
        
        return X, y
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        划分数据集为训练集、验证集和测试集
        
        Args:
            X: 输入特征
            y: 目标输出
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 计算分割点
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 分割数据
        X_train = X_shuffled[:train_end]
        X_val = X_shuffled[train_end:val_end]
        X_test = X_shuffled[val_end:]
        
        y_train = y_shuffled[:train_end]
        y_val = y_shuffled[train_end:val_end]
        y_test = y_shuffled[val_end:]
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test