"""训练模块

实现神经网络的训练过程，包括训练循环、验证、早停等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import json
from loguru import logger

from .neural_network import BasketballNet, CustomLoss


class Trainer:
    """神经网络训练器
    
    负责训练篮球投篮控制神经网络
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 optimizer_type: str = 'adam'):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            optimizer_type: 优化器类型
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # 设置优化器
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), 
                                      lr=learning_rate, 
                                      weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), 
                                     lr=learning_rate, 
                                     momentum=0.9,
                                     weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 设置损失函数
        self.criterion = CustomLoss()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 早停参数
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    def prepare_data(self, 
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            batch_size: 批次大小
            
        Returns:
            训练数据加载器, 验证数据加载器
        """
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"数据准备完成 - 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="训练", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with tqdm(val_loader, desc="验证", leave=False) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 100,
             patience: int = 20,
             save_best: bool = True,
             model_save_path: str = "data/models/best_model.pth") -> Dict:
        """
        完整的训练过程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            patience: 早停耐心值
            save_best: 是否保存最佳模型
            model_save_path: 模型保存路径
            
        Returns:
            训练历史字典
        """
        logger.info(f"开始训练，总轮数: {epochs}")
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(current_lr)
            
            # 打印进度
            logger.info(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 学习率: {current_lr:.2e}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_best:
                    self.save_model(model_save_path, epoch, val_loss)
                    logger.info(f"保存最佳模型到: {model_save_path}")
            else:
                self.patience_counter += 1
                
            # 早停
            if self.patience_counter >= patience:
                logger.info(f"验证损失连续 {patience} 轮未改善，提前停止训练")
                break
            
            # 学习率过小时停止
            if current_lr < 1e-7:
                logger.info("学习率过小，停止训练")
                break
        
        logger.info(f"训练完成，最佳验证损失: {self.best_val_loss:.6f}")
        
        return self.train_history
    
    def save_model(self, filepath: str, epoch: int, val_loss: float):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            epoch: 当前轮数
            val_loss: 验证损失
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history
        }, filepath)
    
    def load_model(self, filepath: str) -> Dict:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            检查点信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        logger.info(f"模型加载完成，验证损失: {checkpoint['val_loss']:.6f}")
        
        return checkpoint
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        # 合并所有预测和目标
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        # 计算各种指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # 分别计算每个输出的误差
        v0_mse = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
        theta_pitch_mse = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
        theta_yaw_mse = np.mean((predictions[:, 2] - targets[:, 2]) ** 2)
        
        results = {
            'test_loss': total_loss / len(test_loader),
            'mse': mse,
            'mae': mae,
            'v0_mse': v0_mse,
            'theta_pitch_mse': theta_pitch_mse,
            'theta_yaw_mse': theta_yaw_mse,
            'predictions': predictions,
            'targets': targets
        }
        
        logger.info(f"模型评估完成 - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        return results
    
    def save_training_history(self, filepath: str):
        """
        保存训练历史
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info(f"训练历史已保存到: {filepath}")
    
    def get_model_complexity(self) -> Dict:
        """
        获取模型复杂度信息
        
        Returns:
            模型复杂度字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 估算FLOPs（浮点运算次数）
        flops = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
        
        # 估算模型大小
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_flops': flops,
            'model_size_mb': model_size_mb
        }


def create_trainer(model: nn.Module, **kwargs) -> Trainer:
    """
    创建训练器的工厂函数
    
    Args:
        model: 神经网络模型
        **kwargs: 训练器参数
        
    Returns:
        训练器实例
    """
    return Trainer(model, **kwargs)