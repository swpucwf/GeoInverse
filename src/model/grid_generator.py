import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

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

class GridGenerator:
    """网格生成器类
    
    提供多种网格剖分方法，包括均匀剖分和非均匀剖分。
    """
    
    @staticmethod
    def uniform_grid(x_range: Tuple[float, float], z_range: Tuple[float, float],
                    nx: int, nz: int) -> Grid2D:
        """生成均匀网格
        
        Args:
            x_range: x方向范围，格式为(x_min, x_max)
            z_range: z方向范围，格式为(z_min, z_max)
            nx: x方向网格点数
            nz: z方向网格点数
            
        Returns:
            Grid2D: 均匀网格对象
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        z = np.linspace(z_range[0], z_range[1], nz)
        dx = np.diff(x)
        dz = np.diff(z)
        
        return Grid2D(x=x, z=z, dx=dx, dz=dz)
    
    @staticmethod
    def geometric_grid(x_range: Tuple[float, float], z_range: Tuple[float, float],
                      nx: int, nz: int, x_ratio: float = 1.1,
                      z_ratio: float = 1.1) -> Grid2D:
        """生成几何级数网格
        
        网格间距按几何级数变化，可用于边界加密。
        
        Args:
            x_range: x方向范围，格式为(x_min, x_max)
            z_range: z方向范围，格式为(z_min, z_max)
            nx: x方向网格点数
            nz: z方向网格点数
            x_ratio: x方向相邻网格间距比例
            z_ratio: z方向相邻网格间距比例
            
        Returns:
            Grid2D: 非均匀网格对象
        """
        # 生成x方向网格
        if x_ratio == 1.0:
            x = np.linspace(x_range[0], x_range[1], nx)
        else:
            # 计算初始间距
            L = x_range[1] - x_range[0]
            a = x_ratio
            n = nx - 1
            dx0 = L * (a - 1) / (a**n - 1)
            
            # 生成网格点
            x = [x_range[0]]
            for i in range(n):
                x.append(x[-1] + dx0 * x_ratio**i)
            x = np.array(x)
        
        # 生成z方向网格
        if z_ratio == 1.0:
            z = np.linspace(z_range[0], z_range[1], nz)
        else:
            # 计算初始间距
            L = z_range[1] - z_range[0]
            a = z_ratio
            n = nz - 1
            dz0 = L * (a - 1) / (a**n - 1)
            
            # 生成网格点
            z = [z_range[0]]
            for i in range(n):
                z.append(z[-1] + dz0 * z_ratio**i)
            z = np.array(z)
        
        dx = np.diff(x)
        dz = np.diff(z)
        
        return Grid2D(x=x, z=z, dx=dx, dz=dz)
    
    @staticmethod
    def refined_grid(x_range: Tuple[float, float], z_range: Tuple[float, float],
                     nx: int, nz: int, refined_regions: List[Tuple[Tuple[float, float],
                     Tuple[float, float], int, int]] = None) -> Grid2D:
        """生成局部加密网格
        
        在指定区域进行网格加密。
        
        Args:
            x_range: x方向范围，格式为(x_min, x_max)
            z_range: z方向范围，格式为(z_min, z_max)
            nx: x方向基础网格点数
            nz: z方向基础网格点数
            refined_regions: 加密区域列表，每个元素为(x_range, z_range, nx_local, nz_local)
                表示在指定区域内插入额外的网格点
            
        Returns:
            Grid2D: 局部加密网格对象
        """
        # 生成基础均匀网格点
        x = list(np.linspace(x_range[0], x_range[1], nx))
        z = list(np.linspace(z_range[0], z_range[1], nz))
        
        # 在加密区域插入额外的网格点
        if refined_regions:
            for region in refined_regions:
                x_local = np.linspace(region[0][0], region[0][1], region[2])
                z_local = np.linspace(region[1][0], region[1][1], region[3])
                x.extend(x_local)
                z.extend(z_local)
            
            # 去除重复点并排序
            x = np.unique(x)
            z = np.unique(z)
        
        dx = np.diff(x)
        dz = np.diff(z)
        
        return Grid2D(x=x, z=z, dx=dx, dz=dz)