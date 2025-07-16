"""物理建模模块

实现篮球投篮的物理模型，包括抛物线运动、空气阻力等因素
"""

import numpy as np
import math
from typing import Tuple, Optional


class PhysicsModel:
    """篮球投篮物理模型
    
    基于经典力学的抛物线运动模型，考虑重力和空气阻力
    """
    
    def __init__(self, 
                 gravity: float = 9.81,
                 air_resistance: float = 0.01,
                 ball_radius: float = 0.12,
                 basket_radius: float = 0.225,
                 basket_height: float = 3.05):
        """
        初始化物理模型参数
        
        Args:
            gravity: 重力加速度 (m/s²)
            air_resistance: 空气阻力系数
            ball_radius: 篮球半径 (m)
            basket_radius: 篮筐半径 (m) 
            basket_height: 标准篮筐高度 (m)
        """
        self.g = gravity
        self.air_resistance = air_resistance
        self.ball_radius = ball_radius
        self.basket_radius = basket_radius
        self.standard_basket_height = basket_height
        
    def calculate_trajectory(self, 
                           x0: float, y0: float, z0: float,
                           v0: float, theta_pitch: float, theta_yaw: float,
                           t_max: float = 5.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算篮球的飞行轨迹
        
        Args:
            x0, y0, z0: 初始位置 (m)
            v0: 初始速度 (m/s)
            theta_pitch: 仰角 (弧度)
            theta_yaw: 偏向角 (弧度)
            t_max: 最大飞行时间 (s)
            dt: 时间步长 (s)
            
        Returns:
            时间数组, x坐标数组, y坐标数组, z坐标数组
        """
        # 初始速度分量
        vx0 = v0 * math.cos(theta_pitch) * math.cos(theta_yaw)
        vy0 = v0 * math.cos(theta_pitch) * math.sin(theta_yaw)
        vz0 = v0 * math.sin(theta_pitch)
        
        # 时间数组
        t = np.arange(0, t_max, dt)
        n_points = len(t)
        
        # 初始化位置和速度数组
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)
        vx = np.zeros(n_points)
        vy = np.zeros(n_points)
        vz = np.zeros(n_points)
        
        # 设置初始条件
        x[0], y[0], z[0] = x0, y0, z0
        vx[0], vy[0], vz[0] = vx0, vy0, vz0
        
        # 数值积分求解轨迹
        for i in range(1, n_points):
            # 当前速度大小
            v_mag = math.sqrt(vx[i-1]**2 + vy[i-1]**2 + vz[i-1]**2)
            
            # 空气阻力
            if v_mag > 0:
                drag_x = -self.air_resistance * vx[i-1] * v_mag
                drag_y = -self.air_resistance * vy[i-1] * v_mag
                drag_z = -self.air_resistance * vz[i-1] * v_mag
            else:
                drag_x = drag_y = drag_z = 0
            
            # 更新速度（考虑重力和空气阻力）
            vx[i] = vx[i-1] + drag_x * dt
            vy[i] = vy[i-1] + drag_y * dt
            vz[i] = vz[i-1] + (-self.g + drag_z) * dt
            
            # 更新位置
            x[i] = x[i-1] + vx[i] * dt
            y[i] = y[i-1] + vy[i] * dt
            z[i] = z[i-1] + vz[i] * dt
            
            # 如果球落地，停止计算
            if z[i] < 0:
                t = t[:i+1]
                x = x[:i+1]
                y = y[:i+1]
                z = z[:i+1]
                break
                
        return t, x, y, z
    
    def calculate_optimal_params(self, 
                               robot_x: float, robot_y: float, robot_z: float,
                               basket_x: float, basket_y: float, basket_z: float) -> Tuple[float, float, float]:
        """
        计算理论最优投篮参数（不考虑空气阻力的解析解）
        
        Args:
            robot_x, robot_y, robot_z: 机器人位置
            basket_x, basket_y, basket_z: 篮筐位置
            
        Returns:
            最优初速度, 最优仰角, 最优偏向角
        """
        # 计算水平距离和高度差
        dx = basket_x - robot_x
        dy = basket_y - robot_y
        dz = basket_z - robot_z
        
        # 水平距离
        horizontal_distance = math.sqrt(dx**2 + dy**2)
        
        # 偏向角（水平方向）
        theta_yaw = math.atan2(dy, dx)
        
        # 使用抛物线运动公式计算最优仰角和初速度
        # 假设45度角附近为最优（可以进一步优化）
        theta_pitch_candidates = np.linspace(math.radians(30), math.radians(60), 100)
        
        best_v0 = float('inf')
        best_theta_pitch = math.radians(45)
        
        for theta_pitch in theta_pitch_candidates:
            # 根据抛物线运动公式计算所需初速度
            cos_theta = math.cos(theta_pitch)
            sin_theta = math.sin(theta_pitch)
            tan_theta = math.tan(theta_pitch)
            
            # 避免除零错误
            if abs(cos_theta) < 1e-6:
                continue
                
            # 抛物线运动方程求解初速度
            discriminant = (sin_theta**2) - (2 * self.g * dz) / (horizontal_distance**2 / cos_theta**2)
            
            if discriminant >= 0:
                v0_squared = (self.g * horizontal_distance**2) / (cos_theta**2 * (horizontal_distance * tan_theta - dz))
                
                if v0_squared > 0:
                    v0 = math.sqrt(v0_squared)
                    if v0 < best_v0:
                        best_v0 = v0
                        best_theta_pitch = theta_pitch
        
        # 如果没有找到合理解，使用经验值
        if best_v0 == float('inf'):
            best_v0 = math.sqrt(self.g * horizontal_distance)  # 经验公式
            best_theta_pitch = math.radians(45)
            
        return best_v0, best_theta_pitch, theta_yaw
    
    def check_trajectory_success(self, 
                               trajectory_x: np.ndarray, 
                               trajectory_y: np.ndarray, 
                               trajectory_z: np.ndarray,
                               basket_x: float, basket_y: float, basket_z: float,
                               tolerance: float = 0.1) -> bool:
        """
        检查轨迹是否成功投篮
        
        Args:
            trajectory_x, trajectory_y, trajectory_z: 轨迹坐标
            basket_x, basket_y, basket_z: 篮筐位置
            tolerance: 允许误差范围
            
        Returns:
            是否成功投篮
        """
        # 找到轨迹中最接近篮筐高度的点
        height_diff = np.abs(trajectory_z - basket_z)
        min_height_idx = np.argmin(height_diff)
        
        # 检查该点是否在篮筐范围内
        closest_x = trajectory_x[min_height_idx]
        closest_y = trajectory_y[min_height_idx]
        
        distance_to_basket = math.sqrt((closest_x - basket_x)**2 + (closest_y - basket_y)**2)
        
        return distance_to_basket <= (self.basket_radius - self.ball_radius + tolerance)
    
    def add_noise(self, 
                  v0: float, theta_pitch: float, theta_yaw: float,
                  v0_noise: float = 0.1, angle_noise: float = 0.05) -> Tuple[float, float, float]:
        """
        为投篮参数添加噪声（模拟传感器误差和执行器误差）
        
        Args:
            v0, theta_pitch, theta_yaw: 原始参数
            v0_noise: 速度噪声标准差
            angle_noise: 角度噪声标准差（弧度）
            
        Returns:
            带噪声的参数
        """
        noisy_v0 = v0 + np.random.normal(0, v0_noise)
        noisy_theta_pitch = theta_pitch + np.random.normal(0, angle_noise)
        noisy_theta_yaw = theta_yaw + np.random.normal(0, angle_noise)
        
        # 确保参数在合理范围内
        noisy_v0 = max(0.1, noisy_v0)  # 速度不能为负
        noisy_theta_pitch = np.clip(noisy_theta_pitch, 0, math.pi/2)  # 仰角在0-90度
        
        return noisy_v0, noisy_theta_pitch, noisy_theta_yaw
    
    def check_shot_success(self, 
                          robot_x: float, robot_y: float, robot_z: float,
                          v0: float, theta_pitch: float, theta_yaw: float,
                          basket_x: float, basket_y: float, basket_z: float) -> bool:
        """
        检查投篮是否成功
        
        Args:
            robot_x, robot_y, robot_z: 机器人位置
            v0: 初始速度
            theta_pitch: 仰角
            theta_yaw: 偏向角
            basket_x, basket_y, basket_z: 篮筐位置
            
        Returns:
            是否成功投篮
        """
        # 计算轨迹
        t, x, y, z = self.calculate_trajectory(robot_x, robot_y, robot_z, v0, theta_pitch, theta_yaw)
        
        # 检查轨迹是否成功
        return self.check_trajectory_success(x, y, z, basket_x, basket_y, basket_z)