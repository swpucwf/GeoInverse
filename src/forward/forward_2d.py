import numpy as np
from typing import List, Tuple, Optional
from ..model.model_2d import EMModel2D, Grid2D, EMParameters2D

class ForwardSolver2D:
    """二维频域电磁场正演求解器类
    
    实现基于有限差分法的二维频域电磁场正演模拟，支持任意电导率分布和多频率计算。
    采用TE模式（横向电场模式），电场垂直于xz平面。
    
    Attributes:
        frequencies (np.ndarray): 工作频率数组，单位为赫兹(Hz)
        source_positions (np.ndarray): 发射源位置数组，形状为(n_sources, 2)，每行为[x, z]坐标
        receiver_positions (np.ndarray): 接收器位置数组，形状为(n_receivers, 2)，每行为[x, z]坐标
    """
    
    def __init__(self, frequencies: np.ndarray,
                 source_positions: np.ndarray,
                 receiver_positions: np.ndarray):
        """初始化求解器
        
        Args:
            frequencies: 工作频率数组
            source_positions: 发射源位置数组
            receiver_positions: 接收器位置数组
        """
        self.frequencies = frequencies
        self.source_positions = source_positions
        self.receiver_positions = receiver_positions
        
    def _build_coefficient_matrix(self, grid: Grid2D, parameters: EMParameters2D,
                                frequency: float) -> np.ndarray:
        """构建有限差分系数矩阵
        
        Args:
            grid: 二维网格参数
            parameters: 电磁场参数
            frequency: 当前频率
            
        Returns:
            np.ndarray: 系数矩阵，形状为(N, N)，其中N为网格节点总数
        """
        nx = len(grid.x)
        nz = len(grid.z)
        N = nx * nz
        
        # 构建稀疏矩阵
        omega = 2 * np.pi * frequency
        mu0 = 4 * np.pi * 1e-7
        epsilon0 = 8.854e-12
        
        # 初始化系数矩阵
        A = np.zeros((N, N), dtype=complex)
        
        # 填充系数矩阵（简化版本，实际应用中需要考虑边界条件和PML等）
        for i in range(nz):
            for j in range(nx):
                idx = i * nx + j
                
                # 获取当前节点的材料参数
                sigma = parameters.conductivity[i, j]
                epsilon_r = parameters.relative_permittivity[i, j]
                mu_r = parameters.relative_permeability[i, j]
                
                # 计算系数
                dx = grid.dx[min(j, nx-2)]
                dz = grid.dz[min(i, nz-2)]
                k = np.sqrt(omega**2 * mu0 * mu_r * epsilon0 * epsilon_r - \
                          1j * omega * mu0 * mu_r * sigma)
                
                # 确保网格间距不为零
                if dx > 0 and dz > 0:
                    # 填充对角元素
                    A[idx, idx] = -2/dx**2 - 2/dz**2 + k**2
                    
                    # 填充非对角元素
                    if j > 0:  # 左侧节点
                        A[idx, idx-1] = 1/dx**2
                    if j < nx-1:  # 右侧节点
                        A[idx, idx+1] = 1/dx**2
                    if i > 0:  # 上方节点
                        A[idx, idx-nx] = 1/dz**2
                    if i < nz-1:  # 下方节点
                        A[idx, idx+nx] = 1/dz**2
                else:
                    # 如果网格间距为零，使用边界条件
                    A[idx, idx] = 1.0
        
        return A
    
    def _solve_field(self, A: np.ndarray, source: np.ndarray) -> np.ndarray:
        """求解电场分布
        
        Args:
            A: 系数矩阵
            source: 源项向量
            
        Returns:
            np.ndarray: 电场分布
        """
        return np.linalg.solve(A, source)
    
    def compute_response(self, model: EMModel2D) -> np.ndarray:
        """计算电磁场响应
        
        Args:
            model: 二维电磁场模型
            
        Returns:
            np.ndarray: 电磁场响应，形状为(n_frequencies, n_receivers)
        """
        n_freq = len(self.frequencies)
        n_rec = len(self.receiver_positions)
        response = np.zeros((n_freq, n_rec), dtype=complex)
        
        for i, freq in enumerate(self.frequencies):
            # 构建系数矩阵
            A = self._build_coefficient_matrix(model.grid, model.parameters, freq)
            
            # 对每个发射源计算场响应
            for src_pos in self.source_positions:
                # 构建源项（简化版本，实际应用中需要考虑源的具体形式）
                source = np.zeros(len(model.grid.x) * len(model.grid.z), dtype=complex)
                src_idx = self._get_nearest_node_index(src_pos, model.grid)
                source[src_idx] = 1.0
                
                # 求解场分布
                field = self._solve_field(A, source)
                
                # 提取接收点处的响应
                for j, rec_pos in enumerate(self.receiver_positions):
                    rec_idx = self._get_nearest_node_index(rec_pos, model.grid)
                    response[i, j] += field[rec_idx]
        
        return response
    
    def _get_nearest_node_index(self, position: np.ndarray, grid: Grid2D) -> int:
        """获取最近网格节点的索引
        
        Args:
            position: 位置坐标[x, z]
            grid: 二维网格参数
            
        Returns:
            int: 网格节点的一维索引
        """
        x, z = position
        j = np.abs(grid.x - x).argmin()
        i = np.abs(grid.z - z).argmin()
        return i * len(grid.x) + j