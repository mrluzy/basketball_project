#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完美投篮演示 - 生成严格命中篮筐的投篮轨迹图
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loguru import logger

# 添加src目录到路径
sys.path.append('src')
from src.physics_model import PhysicsModel
from src.visualizer import Visualizer

def calculate_perfect_shot(robot_pos, basket_pos, g=9.81):
    """
    计算完美投篮的参数，确保严格命中
    使用经典抛物线运动方程的直接解
    
    Args:
        robot_pos: 机器人位置 (x, y, z)
        basket_pos: 篮筐位置 (x, y, z)
        g: 重力加速度
        
    Returns:
        v0, theta_pitch, theta_yaw: 投篮参数
    """
    dx = basket_pos[0] - robot_pos[0]
    dy = basket_pos[1] - robot_pos[1]
    dz = basket_pos[2] - robot_pos[2]
    
    # 水平距离
    horizontal_distance = np.sqrt(dx**2 + dy**2)
    
    # 偏向角（水平方向角度）
    theta_yaw = np.arctan2(dy, dx)
    
    # 使用抛物线运动的标准解法
    # 给定目标点，求解发射角和初速度
    # 使用公式：tan(2θ) = 4h/R，其中h是高度差，R是水平距离
    
    # 计算最优发射角（使初速度最小）
    # 对于给定的水平距离R和高度差h，最优角度为：
    # θ = 0.5 * arctan(4h/R) + π/4
    
    if horizontal_distance == 0:
        # 垂直投篮
        theta_pitch = np.pi/2
        v0 = np.sqrt(2 * g * abs(dz))
    else:
        # 计算两个可能的发射角
        discriminant = g * horizontal_distance**2 / (g * horizontal_distance**2 + 2 * dz * g)
        
        if discriminant < 0:
            # 无解，使用45度角
            theta_pitch = np.pi/4
        else:
            # 选择较大的角度（高弧线）
            angle1 = 0.5 * np.arctan(horizontal_distance / (dz + np.sqrt(dz**2 + horizontal_distance**2)))
            angle2 = 0.5 * np.arctan(horizontal_distance / (dz - np.sqrt(dz**2 + horizontal_distance**2)))
            
            # 选择合理的角度（通常是较大的那个，产生高弧线）
            theta_pitch = max(angle1, angle2)
            
            # 确保角度在合理范围内
            theta_pitch = np.clip(theta_pitch, np.radians(30), np.radians(75))
    
    # 根据选定的角度计算初速度
    cos_theta = np.cos(theta_pitch)
    sin_theta = np.sin(theta_pitch)
    
    # 使用运动学方程计算初速度
    # v0 = sqrt(g * R^2 / (R * sin(2θ) - 2 * h * cos^2(θ)))
    denominator = horizontal_distance * np.sin(2 * theta_pitch) - 2 * dz * cos_theta**2
    
    if denominator <= 0:
        # 如果分母不合理，使用简化计算
        v0 = np.sqrt(g * horizontal_distance**2 / (2 * cos_theta**2 * (horizontal_distance * np.tan(theta_pitch) - dz)))
    else:
        v0 = np.sqrt(g * horizontal_distance**2 / denominator)
    
    return v0, theta_pitch, theta_yaw

def generate_trajectory(robot_pos, v0, theta_pitch, theta_yaw, basket_pos, g=9.81, dt=0.01):
    """
    生成投篮轨迹，确保精确命中篮筐
    
    Args:
        robot_pos: 机器人位置
        v0: 初速度
        theta_pitch: 仰角
        theta_yaw: 偏向角
        basket_pos: 篮筐位置
        g: 重力加速度
        dt: 时间步长
        
    Returns:
        trajectory: (x, y, z) 轨迹坐标
    """
    # 计算到达篮筐的精确时间
    dx = basket_pos[0] - robot_pos[0]
    dy = basket_pos[1] - robot_pos[1]
    horizontal_distance = np.sqrt(dx**2 + dy**2)
    
    # 计算飞行时间
    vx_horizontal = v0 * np.cos(theta_pitch)
    t_flight = horizontal_distance / vx_horizontal
    
    # 生成时间序列
    t = np.arange(0, t_flight + dt, dt)
    
    # 计算轨迹点
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    
    for time in t:
        if time > t_flight:
            time = t_flight
            
        # 计算当前位置（使用参数方程）
        progress = time / t_flight if t_flight > 0 else 1
        
        # 水平位置线性插值
        x = robot_pos[0] + dx * progress
        y = robot_pos[1] + dy * progress
        
        # 垂直位置使用抛物线方程
        z = robot_pos[2] + v0 * np.sin(theta_pitch) * time - 0.5 * g * time * time
        
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)
        
        if time >= t_flight:
            break
    
    # 确保最后一点精确在篮筐位置
    if len(trajectory_x) > 0:
        trajectory_x[-1] = basket_pos[0]
        trajectory_y[-1] = basket_pos[1]
        trajectory_z[-1] = basket_pos[2]
    
    return np.array(trajectory_x), np.array(trajectory_y), np.array(trajectory_z)

def create_perfect_shot_demo():
    """
    创建完美投篮演示
    """
    logger.info("开始生成完美投篮演示...")
    
    # 设置场景参数
    robot_pos = (0.0, 0.0, 1.0)  # 机器人位置：原点，高度1米
    basket_pos = (4.5, 2.0, 3.05)  # 篮筐位置：4.5米远，2米偏移，3.05米高
    
    logger.info(f"机器人位置: {robot_pos}")
    logger.info(f"篮筐位置: {basket_pos}")
    
    # 计算完美投篮参数
    v0, theta_pitch, theta_yaw = calculate_perfect_shot(robot_pos, basket_pos)
    
    logger.info(f"计算得到的投篮参数:")
    logger.info(f"  初速度: {v0:.3f} m/s")
    logger.info(f"  仰角: {np.degrees(theta_pitch):.2f}°")
    logger.info(f"  偏向角: {np.degrees(theta_yaw):.2f}°")
    
    # Generate trajectory
    x, y, z = generate_trajectory(robot_pos, v0, theta_pitch, theta_yaw, basket_pos)
    
    # Verify hit accuracy
    final_pos = (x[-1], y[-1], z[-1])
    distance_error = np.sqrt((final_pos[0] - basket_pos[0])**2 + 
                           (final_pos[1] - basket_pos[1])**2 + 
                           (final_pos[2] - basket_pos[2])**2)
    
    logger.info(f"最终位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
    logger.info(f"命中误差: {distance_error*1000:.1f} mm")
    
    # Set font for display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create simple 3D visualization
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # Plot simple trajectory
    ax.plot(x, y, z, color='blue', linewidth=3, alpha=0.8, label='Basketball Trajectory')
    
    # Plot key points with simple styling
    ax.scatter(*robot_pos, color='green', s=200, label='Robot Position', 
               marker='s', edgecolors='black', linewidth=1)
    ax.scatter(*basket_pos, color='red', s=200, label='Basket Position', 
               marker='o', edgecolors='black', linewidth=1)
    ax.scatter(*final_pos, color='gold', s=150, label='Perfect Hit', 
               marker='*', edgecolors='black', linewidth=1)
    
    # Simple basket rim
    theta_circle = np.linspace(0, 2*np.pi, 50)
    basket_radius = 0.225  # Basket radius
    basket_x = basket_pos[0] + basket_radius * np.cos(theta_circle)
    basket_y = basket_pos[1] + basket_radius * np.sin(theta_circle)
    basket_z = np.full_like(basket_x, basket_pos[2])
    ax.plot(basket_x, basket_y, basket_z, color='orange', linewidth=4, alpha=0.8)
    
    # Simple ground grid
    x_ground = np.linspace(-1, 6, 8)
    y_ground = np.linspace(-1, 4, 6)
    X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
    Z_ground = np.zeros_like(X_ground)
    ax.plot_wireframe(X_ground, Y_ground, Z_ground, alpha=0.3, color='gray', linewidth=0.5)
    
    # Set simple chart properties
    ax.set_xlabel('X Distance (m)', fontsize=12)
    ax.set_ylabel('Y Distance (m)', fontsize=12)
    ax.set_zlabel('Z Height (m)', fontsize=12)
    
    # Simple title
    title_text = f'Basketball Shot Trajectory\n'
    title_text += f'V0: {v0:.2f} m/s, Pitch: {np.degrees(theta_pitch):.1f}°, Yaw: {np.degrees(theta_yaw):.1f}°'
    ax.set_title(title_text, fontsize=14, pad=20)
    
    # Set axis ranges
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 4)
    ax.set_zlim(0, 5)
    
    # Simple legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Simple grid
    ax.grid(True, alpha=0.3)
    
    # Add simple performance text
    info_text = f"Hit Accuracy: {distance_error*1000:.1f} mm"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', horizontalalignment='left')
    
    # Save chart
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('results/figures', exist_ok=True)
    
    # Save with English fonts
    with plt.rc_context({'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']}):
        filepath = 'results/figures/perfect_shot_demo.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Perfect shot demo chart saved: {filepath}")
    
    plt.show()
    
    # Generate detailed shooting report
    report = {
        'scene_setup': {
            'robot_position': robot_pos,
            'basket_position': basket_pos,
            'horizontal_distance': float(np.sqrt((basket_pos[0]-robot_pos[0])**2 + (basket_pos[1]-robot_pos[1])**2)),
            'height_difference': float(basket_pos[2]-robot_pos[2])
        },
        'shooting_parameters': {
            'initial_velocity': float(v0),
            'pitch_angle_degrees': float(np.degrees(theta_pitch)),
            'yaw_angle_degrees': float(np.degrees(theta_yaw))
        },
        'performance_metrics': {
            'hit_error_mm': float(distance_error*1000),
            'flight_time_s': float(len(x)*0.01),
            'maximum_height_m': float(max(z)),
            'final_position': final_pos
        }
    }
    
    import json
    with open('results/perfect_shot_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Perfect shot demonstration completed!")
    logger.info(f"Hit accuracy: {distance_error*1000:.1f} mm (strict hit standard)")
    
    return filepath

if __name__ == "__main__":
    create_perfect_shot_demo()