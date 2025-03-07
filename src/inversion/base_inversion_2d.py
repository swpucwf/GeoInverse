import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from ..model.model_2d import EMModel2D, Grid2D, EMParameters2D
from ..forward.forward_2d import ForwardSolver2D

class BaseInversion2D(ABC):
    """二维电磁反演基类
    
    为二维电磁反演算法提供基础框架，定义了反演过程中所需的基本接口和通用功能。
    具体的反演算法（如Occam反演、贝叶斯反演等）需要继承此类并实现相应的抽象方法。
    
    Attributes:
        forward_solver (ForwardSolver2D): 二维正演求解器
        model (EMModel2D): 二维电磁场模型
        data (np.ndarray): 观测数据
        data_std (np.ndarray): 观测数据的标准差
    """
    
    def __init__(self, forward_solver: ForwardSolver2D, model: EMModel2D,
                 data: np.ndarray, data_std: Optional[np.ndarray] = None):
        """初始化反演器
        
        Args:
            forward_solver: 二维正演求解器
            model: 二维电磁场模型
            data: 观测数据数组
            data_std: 观测数据的标准差，用于数据加权
        """
        self.forward_solver = forward_solver
        self.model = model
        self.data = data
        self.data_std = data_std if data_std is not None else np.ones_like(data)
        
    @abstractmethod
    def objective_function(self, model_params: np.ndarray) -> float:
        """计算目标函数值
        
        Args:
            model_params: 模型参数数组
            
        Returns:
            float: 目标函数值
        """
        pass
    
    @abstractmethod
    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """计算目标函数对模型参数的梯度
        
        Args:
            model_params: 模型参数数组
            
        Returns:
            np.ndarray: 梯度数组
        """
        pass
    
    @abstractmethod
    def invert(self, initial_model: Optional[np.ndarray] = None,
              max_iterations: int = 100) -> Tuple[np.ndarray, dict]:
        """执行反演
        
        Args:
            initial_model: 初始模型参数，如果为None则使用默认值
            max_iterations: 最大迭代次数
            
        Returns:
            Tuple[np.ndarray, dict]: 反演结果和收敛信息
        """
        pass
    
    def data_misfit(self, predicted_data: np.ndarray) -> float:
        """计算数据拟合误差
        
        Args:
            predicted_data: 正演计算的预测数据
            
        Returns:
            float: 加权均方根误差
        """
        residuals = (predicted_data - self.data) / self.data_std
        return np.sqrt(np.mean(residuals**2))
    
    def regularization(self, model_params: np.ndarray) -> float:
        """计算正则化项
        
        Args:
            model_params: 模型参数数组
            
        Returns:
            float: 正则化项值
        """
        # 默认使用一阶差分作为平滑约束
        dx = np.diff(model_params.reshape(self.model.grid.z.size, -1), axis=1)
        dz = np.diff(model_params.reshape(self.model.grid.z.size, -1), axis=0)
        return np.sum(dx**2) + np.sum(dz**2)
    
    def update_model(self, model_params: np.ndarray) -> None:
        """更新模型参数
        
        Args:
            model_params: 新的模型参数数组
        """
        self.model.parameters.conductivity = model_params.reshape(
            self.model.grid.z.size, self.model.grid.x.size)