import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.model.model import LayeredEarthModel

__all__ = ['plot_model', 'plot_response']

def plot_model(model: LayeredEarthModel, ax=None, depth_range: Optional[Tuple[float, float]] = None,
            show_parameters: bool = True) -> None:
    """
    绘制地层模型
    
    Args:
        model: 分层地球模型对象
        ax: matplotlib轴对象，如果为None则创建新的图形
        depth_range: 绘图的深度范围，格式为(min_depth, max_depth)
        show_parameters: 是否显示地层参数
    """
    model.plot(ax=ax, depth_range=depth_range, show_parameters=show_parameters)


def plot_response(response: np.ndarray, frequencies: List[float], offsets: List[float],
                 plot_type: str = 'amplitude', ax: Optional[Axes] = None,
                 title: Optional[str] = None) -> Axes:
    """
    绘制电磁场响应
    
    Args:
        response: 电磁场响应数组，形状为(频率数, 偏移距数)
        frequencies: 频率列表 (Hz)
        offsets: 偏移距列表 (m)
        plot_type: 绘图类型，可选'amplitude'或'phase'
        ax: matplotlib轴对象，如果为None则创建新的图形
        title: 图形标题
        
    Returns:
        Axes: 绘图轴对象
    """
    return EMVisualization.plot_field_response(
        np.array(frequencies), np.array(offsets), response,
        plot_type=plot_type, ax=ax, title=title
    )


class EMVisualization:
    """电磁场响应可视化类
    
    提供统一的可视化接口，用于绘制电磁场响应、地层模型等
    支持多种绘图类型和布局方式
    """
    
    @staticmethod
    def plot_field_response(frequencies: np.ndarray,
                           offsets: np.ndarray,
                           response: np.ndarray,
                           plot_type: str = 'both',
                           ax: Optional[Union[Axes, Tuple[Axes, Axes]]] = None,
                           title: Optional[str] = None) -> Union[Axes, Tuple[Axes, Axes]]:
        """绘制电磁场响应
        
        Args:
            frequencies: 频率数组 (Hz)
            offsets: 偏移距数组 (m)
            response: 电磁场响应数组，形状为(频率数, 偏移距数)
            plot_type: 绘图类型，可选'amplitude'、'phase'或'both'
            ax: matplotlib轴对象，如果为None则创建新的图形
            title: 图形标题
            
        Returns:
            Union[Axes, Tuple[Axes, Axes]]: 绘图轴对象
        """
        if plot_type not in ['amplitude', 'phase', 'both']:
            raise ValueError("plot_type必须是'amplitude'、'phase'或'both'之一")
        
        if plot_type == 'both':
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            else:
                ax1, ax2 = ax
                
            EMVisualization.plot_field_response(frequencies, offsets, response,
                                              'amplitude', ax1)
            EMVisualization.plot_field_response(frequencies, offsets, response,
                                              'phase', ax2)
            
            if title:
                ax1.figure.suptitle(title)
            
            return ax1, ax2
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        for i, freq in enumerate(frequencies):
            if plot_type == 'amplitude':
                ax.semilogx(offsets, np.abs(response[i, :]),
                           label=f'{freq:.1f} Hz')
                ax.set_ylabel('幅值 (V/m)')
            else:
                ax.semilogx(offsets, np.angle(response[i, :], deg=True),
                           label=f'{freq:.1f} Hz')
                ax.set_ylabel('相位 (度)')
        
        ax.set_xlabel('偏移距 (m)')
        ax.grid(True)
        ax.legend()
        
        if title:
            ax.set_title(title)
        
        return ax
    
    @staticmethod
    def plot_field_map(frequencies: np.ndarray,
                       offsets: np.ndarray,
                       response: np.ndarray,
                       plot_type: str = 'amplitude',
                       ax: Optional[Axes] = None,
                       title: Optional[str] = None) -> Axes:
        """绘制电磁场响应的二维分布图
        
        Args:
            frequencies: 频率数组 (Hz)
            offsets: 偏移距数组 (m)
            response: 电磁场响应数组，形状为(频率数, 偏移距数)
            plot_type: 绘图类型，'amplitude'或'phase'
            ax: matplotlib轴对象，如果为None则创建新的图形
            title: 图形标题
            
        Returns:
            Axes: 绘图轴对象
        """
        if plot_type not in ['amplitude', 'phase']:
            raise ValueError("plot_type必须是'amplitude'或'phase'之一")
            
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        X, Y = np.meshgrid(offsets, frequencies)
        if plot_type == 'amplitude':
            Z = np.abs(response)
            label = '幅值 (V/m)'
        else:
            Z = np.angle(response, deg=True)
            label = '相位 (度)'
            
        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        ax.set_xlabel('偏移距 (m)')
        ax.set_ylabel('频率 (Hz)')
        plt.colorbar(im, ax=ax, label=label)
        
        if title:
            ax.set_title(title)
            
        return ax