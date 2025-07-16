"""机器人篮球投篮控制器

基于神经网络的智能投篮控制系统
"""

__version__ = "1.0.0"
__author__ = "Basketball AI Team"
__email__ = "basketball.ai@example.com"

from .physics_model import PhysicsModel
from .data_generator import DataGenerator
from .neural_network import BasketballNet
from .trainer import Trainer
from .evaluator import Evaluator
from .visualizer import Visualizer

__all__ = [
    "PhysicsModel",
    "DataGenerator", 
    "BasketballNet",
    "Trainer",
    "Evaluator",
    "Visualizer"
]