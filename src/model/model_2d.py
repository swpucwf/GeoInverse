import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from src.model.base_model import BaseModel


@dataclass
class Grid2D:
    """二维网格参数类
    
    用于描述二维模型的网格剖分参数，包括x方向和z方向的网格节点坐标。
    
    Attributes:
        x (np.ndarray): x方向网格节点坐标，单位为米(m)
        z (np.ndarray): z方向网格节点坐标，单位为米(m)
        dx (np.ndarray): x方向网格间距，单位为米(m)
        dz (np.ndarray): z方向网格间距，单位为米(m)
    """
    x: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    dz: np.ndarray
    
    def plot(self, ax=None, show_grid=True, **kwargs):
        """绘制网格
        
        Args:
            ax: matplotlib轴对象
            show_grid: 是否显示网格线
            **kwargs: 其他绘图参数
        """
        if ax is None:
            _, ax = plt.subplots()
            
        X, Z = np.meshgrid(self.x, self.z)
        
        if show_grid:
            # 绘制垂直网格线
            for x in self.x:
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            # 绘制水平网格线
            for z in self.z:
                ax.axhline(z, color='gray', linestyle=':', alpha=0.5)
        
        # 绘制网格点
        ax.plot(X.flatten(), Z.flatten(), 'k.', markersize=2)
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.grid(False)
        
        return ax

@dataclass
class EMParameters2D:
    """二维电磁场参数类
    
    用于描述二维电磁场模型的物理参数分布，包括电导率、相对介电常数和相对磁导率。
    
    Attributes:
        conductivity (np.ndarray): 电导率分布，形状为(nz, nx)，单位为西门子/米(S/m)
        relative_permittivity (np.ndarray): 相对介电常数分布，形状为(nz, nx)，无量纲
        relative_permeability (np.ndarray): 相对磁导率分布，形状为(nz, nx)，无量纲
    """
    conductivity: np.ndarray
    relative_permittivity: np.ndarray = None
    relative_permeability: np.ndarray = None

class EMModel2D(BaseModel):
    """二维频域电磁场正演模型类
    
    实现基于有限差分法的二维频域电磁场正演模拟，支持任意电导率分布和多频率计算。
    模型采用TE模式（横向电场模式），电场垂直于xz平面。
    
    Attributes:
        grid (Grid2D): 二维网格参数
        parameters (EMParameters2D): 电磁场参数
        frequencies (np.ndarray): 工作频率数组，单位为赫兹(Hz)
        name (str): 模型名称
    """
    
    def __init__(self, grid: Grid2D, parameters: EMParameters2D,
                 frequencies: np.ndarray, name: str = "EMModel2D"):
        """初始化二维频域电磁场模型
        
        Args:
            grid: 二维网格参数
            parameters: 电磁场参数
            frequencies: 工作频率数组
            name: 模型名称
        """
        super().__init__(name)
        self.grid = grid
        self.parameters = parameters
        self.frequencies = frequencies
        
        # 设置默认的相对介电常数和磁导率
        if parameters.relative_permittivity is None:
            parameters.relative_permittivity = np.ones_like(parameters.conductivity)
        if parameters.relative_permeability is None:
            parameters.relative_permeability = np.ones_like(parameters.conductivity)
            
        self._validate_dimensions()
    
    def _validate_dimensions(self) -> None:
        """验证网格和参数维度的一致性
        
        Raises:
            ValueError: 当网格和参数维度不匹配时
        """
        nx = len(self.grid.x)
        nz = len(self.grid.z)
        if self.parameters.conductivity.shape != (nz, nx):
            raise ValueError(f"电导率数组维度{self.parameters.conductivity.shape}与网格维度({nz}, {nx})不匹配")
    
    def get_parameters(self) -> Tuple[np.ndarray, ...]:
        """获取模型参数
        
        Returns:
            Tuple[np.ndarray, ...]: 包含电导率、相对介电常数和相对磁导率的元组
        """
        return (self.parameters.conductivity,
                self.parameters.relative_permittivity,
                self.parameters.relative_permeability)
    
    def plot(self, ax=None, parameter='conductivity', **kwargs) -> None:
        """绘制模型参数分布
        
        Args:
            ax: matplotlib轴对象
            parameter: 要绘制的参数，可选'conductivity'、'permittivity'或'permeability'
            **kwargs: 其他绘图参数
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots()
            
        if parameter == 'conductivity':
            data = 1.0 / self.parameters.conductivity  # 转换为电阻率
            title = '电阻率 (Ω·m)'
        elif parameter == 'permittivity':
            data = self.parameters.relative_permittivity
            title = '相对介电常数'
        else:
            data = self.parameters.relative_permeability
            title = '相对磁导率'
            
        X, Z = np.meshgrid(self.grid.x, self.grid.z)
        im = ax.pcolormesh(X, Z, data, shading='auto', **kwargs)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(title)
    
    def validate(self) -> bool:
        """验证模型参数的有效性
        
        Returns:
            bool: 如果所有参数都有效则返回True
        """
        # 验证电导率为正值
        if np.any(self.parameters.conductivity <= 0):
            return False
        
        # 验证相对介电常数和磁导率为正值
        if np.any(self.parameters.relative_permittivity <= 0) or \
           np.any(self.parameters.relative_permeability <= 0):
            return False
        
        return True