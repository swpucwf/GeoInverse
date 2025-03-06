import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.model.model import LayeredEarthModel

__all__ = ['plot_model', 'plot_response']

def plot_model(model: LayeredEarthModel, ax=None, depth_range: Optional[Tuple[float, float]] = None,
            show_parameters: bool = True) -> None:
    """绘制地层模型
    
    将LayeredEarthModel对象可视化为二维剖面图，显示各层的电阻率分布和物理参数。
    
    Args:
        model: 分层地球模型对象，包含各层的物理参数信息
        ax: matplotlib轴对象，如果为None则创建新的图形
        depth_range: 绘图的深度范围，格式为(min_depth, max_depth)，如果为None则自动设置
        show_parameters: 是否在图中显示地层参数（电阻率、介电常数等）
        
    Example:
        >>> model = LayeredEarthModel.from_depths_resistivities(
        ...     [0, 10, 30, np.inf], [100, 50, 500]
        ... )
        >>> plot_model(model, depth_range=(0, 50))
    """
    model.plot(ax=ax, depth_range=depth_range, show_parameters=show_parameters)


def plot_response(response: np.ndarray, frequencies: List[float], offsets: List[float],
                 plot_type: str = 'amplitude', ax: Optional[Axes] = None,
                 title: Optional[str] = None) -> Axes:
    """绘制电磁场响应
    
    将计算得到的电磁场响应数据可视化，支持绘制幅值或相位随偏移距的变化曲线。
    
    Args:
        response: 电磁场响应数组，形状为(频率数, 偏移距数)，包含复数值
        frequencies: 频率列表，单位为Hz
        offsets: 偏移距列表，单位为米
        plot_type: 绘图类型，可选'amplitude'（幅值）或'phase'（相位）
        ax: matplotlib轴对象，如果为None则创建新的图形
        title: 图形标题
        
    Returns:
        Axes: matplotlib轴对象，包含绘制的图形
        
    Example:
        >>> response = forward_model.calculate_response(frequencies, offsets)
        >>> plot_response(response, frequencies, offsets, plot_type='amplitude',
        ...              title='电磁场响应幅值')
    """
    return EMVisualization.plot_field_response(
        np.array(frequencies), np.array(offsets), response,
        plot_type=plot_type, ax=ax, title=title
    )


class EMVisualization:
    """电磁场响应可视化类
    
    提供统一的可视化接口，用于绘制电磁场响应、地层模型等。
    支持多种绘图类型和布局方式，包括：
    1. 响应曲线图（幅值/相位随偏移距变化）
    2. 响应分布图（频率-偏移距二维分布）
    
    Example:
        >>> # 绘制响应曲线
        >>> EMVisualization.plot_field_response(
        ...     frequencies, offsets, response, plot_type='both'
        ... )
        >>> # 绘制响应分布图
        >>> EMVisualization.plot_field_map(
        ...     frequencies, offsets, response, plot_type='amplitude'
        ... )
    """
    
    @staticmethod
    def plot_field_response(frequencies: np.ndarray,
                           offsets: np.ndarray,
                           response: np.ndarray,
                           plot_type: str = 'both',
                           ax: Optional[Union[Axes, Tuple[Axes, Axes]]] = None,
                           title: Optional[str] = None) -> Union[Axes, Tuple[Axes, Axes]]:
        """绘制电磁场响应曲线
        
        将电磁场响应数据绘制为曲线图，可选择绘制幅值、相位或两者都绘制。
        
        Args:
            frequencies: 频率数组，单位为Hz
            offsets: 偏移距数组，单位为米
            response: 电磁场响应数组，形状为(频率数, 偏移距数)，包含复数值
            plot_type: 绘图类型，可选'amplitude'、'phase'或'both'
            ax: matplotlib轴对象，如果为None则创建新的图形
            title: 图形标题
            
        Returns:
            Union[Axes, Tuple[Axes, Axes]]: 一个或两个matplotlib轴对象
            
        Raises:
            ValueError: 当plot_type不是有效值时
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
        
        将电磁场响应数据绘制为频率-偏移距二维分布图，使用颜色表示响应值的大小。
        
        Args:
            frequencies: 频率数组，单位为Hz
            offsets: 偏移距数组，单位为米
            response: 电磁场响应数组，形状为(频率数, 偏移距数)，包含复数值
            plot_type: 绘图类型，'amplitude'（幅值）或'phase'（相位）
            ax: matplotlib轴对象，如果为None则创建新的图形
            title: 图形标题
            
        Returns:
            Axes: matplotlib轴对象
            
        Raises:
            ValueError: 当plot_type不是有效值时
            
        Example:
            >>> EMVisualization.plot_field_map(
            ...     frequencies, offsets, response,
            ...     plot_type='amplitude',
            ...     title='电磁场响应幅值分布'
            ... )
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