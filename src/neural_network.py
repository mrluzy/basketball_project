"""神经网络模块

实现轻量级的前馈神经网络，适合嵌入式部署
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math


class BasketballNet(nn.Module):
    """篮球投篮控制神经网络
    
    轻量级前馈神经网络，用于预测最优投篮参数
    """
    
    def __init__(self, 
                 input_size: int = 5,
                 hidden_sizes: list = [64, 64],
                 output_size: int = 3,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度
            dropout_rate: Dropout比率
            activation: 激活函数类型
        """
        super(BasketballNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_size]
            
        Returns:
            输出张量 [batch_size, output_size]
        """
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测函数（用于推理）
        
        Args:
            x: 输入数组 [batch_size, input_size] 或 [input_size]
            
        Returns:
            预测结果数组
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_tensor = torch.FloatTensor(x)
            
            # 预测
            output = self.forward(x_tensor)
            
            # 转换回numpy数组
            result = output.cpu().numpy()
            
            # 如果输入是单个样本，返回一维数组
            if result.shape[0] == 1:
                result = result.squeeze(0)
                
            return result
    
    def get_model_size(self) -> Dict[str, int]:
        """
        获取模型大小信息
        
        Returns:
            模型大小统计字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算模型大小（字节）
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def prune_model(self, pruning_ratio: float = 0.2) -> 'BasketballNet':
        """
        模型剪枝（简单的权重剪枝）
        
        Args:
            pruning_ratio: 剪枝比例
            
        Returns:
            剪枝后的模型
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # 计算权重的阈值
                    weight_abs = torch.abs(module.weight)
                    threshold = torch.quantile(weight_abs, pruning_ratio)
                    
                    # 创建掩码
                    mask = weight_abs > threshold
                    
                    # 应用剪枝
                    module.weight.data *= mask.float()
        
        return self
    
    def quantize_model(self) -> 'BasketballNet':
        """
        模型量化（简单的8位量化）
        
        Returns:
            量化后的模型
        """
        # 这里实现简单的权重量化
        # 实际应用中可以使用PyTorch的量化工具
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # 计算量化参数
                    weight_min = module.weight.min()
                    weight_max = module.weight.max()
                    
                    # 量化到8位
                    scale = (weight_max - weight_min) / 255
                    zero_point = -weight_min / scale
                    
                    # 量化和反量化
                    quantized = torch.round(module.weight / scale + zero_point)
                    quantized = torch.clamp(quantized, 0, 255)
                    module.weight.data = (quantized - zero_point) * scale
        
        return self


class EmbeddedBasketballNet(nn.Module):
    """嵌入式优化版本的篮球投篮网络
    
    专为嵌入式设备设计的超轻量级网络
    """
    
    def __init__(self, input_size: int = 5, output_size: int = 3):
        """
        初始化嵌入式网络
        
        Args:
            input_size: 输入特征维度
            output_size: 输出维度
        """
        super(EmbeddedBasketballNet, self).__init__()
        
        # 极简网络结构
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        
        # 使用更简单的激活函数
        self.activation = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测函数
        
        Args:
            x: 输入数组
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_tensor = torch.FloatTensor(x)
            output = self.forward(x_tensor)
            result = output.cpu().numpy()
            if result.shape[0] == 1:
                result = result.squeeze(0)
            return result


class CustomLoss(nn.Module):
    """自定义损失函数
    
    结合MSE损失和物理约束
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 physics_weight: float = 0.1,
                 angle_weight: float = 0.5):
        """
        初始化损失函数
        
        Args:
            mse_weight: MSE损失权重
            physics_weight: 物理约束权重
            angle_weight: 角度约束权重
        """
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.angle_weight = angle_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 预测值 [batch_size, 3] (v0, theta_pitch, theta_yaw)
            targets: 目标值 [batch_size, 3]
            
        Returns:
            总损失
        """
        # 基本MSE损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 物理约束损失（速度应该为正）
        v0_pred = predictions[:, 0]
        physics_loss = torch.mean(torch.relu(-v0_pred))  # 惩罚负速度
        
        # 角度约束损失（仰角应该在0-π/2之间）
        theta_pitch_pred = predictions[:, 1]
        angle_loss = torch.mean(torch.relu(-theta_pitch_pred)) + \
                    torch.mean(torch.relu(theta_pitch_pred - math.pi/2))
        
        # 总损失
        total_loss = (self.mse_weight * mse_loss + 
                     self.physics_weight * physics_loss + 
                     self.angle_weight * angle_loss)
        
        return total_loss


def create_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    创建模型的工厂函数
    
    Args:
        model_type: 模型类型 ('standard', 'embedded')
        **kwargs: 模型参数
        
    Returns:
        神经网络模型
    """
    if model_type == 'standard':
        return BasketballNet(**kwargs)
    elif model_type == 'embedded':
        return EmbeddedBasketballNet(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> Dict:
    """
    获取模型摘要信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型摘要字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(model_size_mb, 3),
        'architecture': str(model)
    }