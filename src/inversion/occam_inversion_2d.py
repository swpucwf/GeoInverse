import numpy as np
from typing import List, Tuple, Optional, Union
from .base_inversion_2d import BaseInversion2D
from ..model.model_2d import EMModel2D
from ..forward.forward_2d import ForwardSolver2D

class OccamInversion2D(BaseInversion2D):
    """基于Occam算法的二维电磁反演类
    
    实现Occam反演算法，通过迭代优化寻找满足数据拟合要求的最平滑模型。
    该算法在每次迭代中调整正则化参数，以平衡数据拟合和模型平滑度。
    
    Attributes:
        target_misfit (float): 目标数据拟合误差
        min_lambda (float): 最小正则化参数
        max_lambda (float): 最大正则化参数
        lambda_factor (float): 正则化参数调整因子
    """
    
    def __init__(self, forward_solver: ForwardSolver2D, model: EMModel2D,
                 data: np.ndarray, data_std: Optional[np.ndarray] = None,
                 target_misfit: float = 1.0, min_lambda: float = 1e-10,
                 max_lambda: float = 1e10, lambda_factor: float = 10.0):
        """初始化Occam反演器
        
        Args:
            forward_solver: 二维正演求解器
            model: 二维电磁场模型
            data: 观测数据数组
            data_std: 观测数据的标准差
            target_misfit: 目标数据拟合误差
            min_lambda: 最小正则化参数
            max_lambda: 最大正则化参数
            lambda_factor: 正则化参数调整因子
        """
        super().__init__(forward_solver, model, data, data_std)
        self.target_misfit = target_misfit
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.lambda_factor = lambda_factor
        self._current_lambda = max_lambda
    
    def objective_function(self, model_params: np.ndarray) -> float:
        """计算目标函数值
        
        目标函数由数据拟合项和正则化项组成：
        Φ = Φ_d + λΦ_m
        其中Φ_d为数据拟合项，Φ_m为模型正则化项，λ为正则化参数
        
        Args:
            model_params: 模型参数数组
            
        Returns:
            float: 目标函数值
        """
        # 更新模型参数
        self.update_model(model_params)
        
        # 计算正演响应
        predicted_data = self.forward_solver.forward(self.model)
        
        # 计算数据拟合项
        data_misfit = self.data_misfit(predicted_data)
        
        # 计算正则化项
        reg_term = self.regularization(model_params)
        
        return data_misfit + self._current_lambda * reg_term
    
    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """计算目标函数对模型参数的梯度
        
        使用伴随状态法计算梯度，避免直接计算Jacobian矩阵
        
        Args:
            model_params: 模型参数数组
            
        Returns:
            np.ndarray: 梯度数组
        """
        # 计算数据拟合项的梯度
        self.update_model(model_params)
        predicted_data = self.forward_solver.forward(self.model)
        residuals = (predicted_data - self.data) / self.data_std**2
        
        # 使用伴随状态法计算敏感性
        sensitivity = self.forward_solver.compute_sensitivity(self.model, residuals)
        
        # 计算正则化项的梯度
        model_shape = (self.model.grid.z.size, self.model.grid.x.size)
        reg_grad = np.zeros_like(model_params)
        
        # 添加一阶差分正则化的梯度
        model_2d = model_params.reshape(model_shape)
        dx = np.diff(model_2d, axis=1)
        dz = np.diff(model_2d, axis=0)
        
        # 计算x方向的梯度
        reg_grad_x = np.zeros_like(model_2d)
        reg_grad_x[:, 1:-1] = -2 * dx[:, :-1] + 2 * dx[:, 1:]
        reg_grad_x[:, 0] = 2 * dx[:, 0]
        reg_grad_x[:, -1] = -2 * dx[:, -1]
        
        # 计算z方向的梯度
        reg_grad_z = np.zeros_like(model_2d)
        reg_grad_z[1:-1, :] = -2 * dz[:-1, :] + 2 * dz[1:, :]
        reg_grad_z[0, :] = 2 * dz[0, :]
        reg_grad_z[-1, :] = -2 * dz[-1, :]
        
        reg_grad = (reg_grad_x + reg_grad_z).flatten()
        
        return sensitivity + self._current_lambda * reg_grad
    
    def invert(self, initial_model: Optional[np.ndarray] = None,
              max_iterations: int = 100) -> Tuple[np.ndarray, dict]:
        """执行反演
        
        使用共轭梯度法优化目标函数，同时动态调整正则化参数
        
        Args:
            initial_model: 初始模型参数，如果为None则使用均匀半空间
            max_iterations: 最大迭代次数
            
        Returns:
            Tuple[np.ndarray, dict]: 反演结果和收敛信息
        """
        # 初始化模型参数
        if initial_model is None:
            initial_model = np.ones(self.model.grid.x.size * self.model.grid.z.size)
        
        current_model = initial_model.copy()
        best_model = current_model.copy()
        best_misfit = float('inf')
        
        # 记录收敛信息
        convergence_info = {
            'iterations': [],
            'misfits': [],
            'regularization_params': [],
            'objective_values': []
        }
        
        for iteration in range(max_iterations):
            # 计算目标函数值和梯度
            obj_value = self.objective_function(current_model)
            grad = self.gradient(current_model)
            
            # 使用共轭梯度法更新模型
            if iteration == 0:
                search_direction = -grad
            else:
                beta = np.sum(grad * (grad - prev_grad)) / np.sum(prev_grad * prev_grad)
                search_direction = -grad + beta * prev_search_direction
            
            # 线搜索确定步长
            alpha = self._line_search(current_model, search_direction)
            
            # 更新模型
            current_model = current_model + alpha * search_direction
            
            # 计算数据拟合误差
            self.update_model(current_model)
            predicted_data = self.forward_solver.forward(self.model)
            current_misfit = self.data_misfit(predicted_data)
            
            # 更新最佳模型
            if current_misfit < best_misfit:
                best_misfit = current_misfit
                best_model = current_model.copy()
            
            # 记录收敛信息
            convergence_info['iterations'].append(iteration)
            convergence_info['misfits'].append(current_misfit)
            convergence_info['regularization_params'].append(self._current_lambda)
            convergence_info['objective_values'].append(obj_value)
            
            # 调整正则化参数
            if current_misfit > self.target_misfit:
                self._current_lambda /= self.lambda_factor
            else:
                self._current_lambda *= self.lambda_factor
            
            self._current_lambda = np.clip(self._current_lambda,
                                          self.min_lambda, self.max_lambda)
            
            # 保存当前梯度和搜索方向
            prev_grad = grad
            prev_search_direction = search_direction
            
            # 检查收敛条件
            if abs(current_misfit - self.target_misfit) < 0.1:
                break
        
        return best_model, convergence_info
    
    def _line_search(self, model: np.ndarray, direction: np.ndarray) -> float:
        """执行线搜索以确定最优步长
        
        使用简单的回溯线搜索算法
        
        Args:
            model: 当前模型参数
            direction: 搜索方向
            
        Returns:
            float: 最优步长
        """
        alpha = 1.0
        c = 0.5
        tau = 0.5
        
        initial_obj = self.objective_function(model)
        initial_grad = self.gradient(model)
        slope = np.sum(initial_grad * direction)
        
        while True:
            new_model = model + alpha * direction
            new_obj = self.objective_function(new_model)
            
            if new_obj <= initial_obj + c * alpha * slope:
                break
            
            alpha *= tau
            
            if alpha < 1e-10:
                break
        
        return alpha