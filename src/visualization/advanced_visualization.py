import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.model.model import LayeredEarthModel
from src.model.model_2d import EMModel2D, Grid2D

__all__ = ['plot_model_comparison', 'plot_grid_analysis', 'plot_forward_results']

def plot_model_comparison(models: List[LayeredEarthModel], titles: List[str] = None,
                       depth_range: Optional[Tuple[float, float]] = None,
                       show_parameters: bool = True) -> Figure:
    """绘制多个地层模型的对比图
    
    将多个LayeredEarthModel对象并排显示，方便比较不同模型的结构和参数。
    
    Args:
        models: 地层模型列表
        titles: 每个模型的标题列表
        depth_range: 绘图的深度范围，格式为(min_depth, max_depth)
        show_parameters: 是否显示地层参数
        
    Returns:
        Figure: matplotlib图形对象
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model, ax) in enumerate(zip(models, axes)):
        model.plot(ax=ax, depth_range=depth_range, show_parameters=show_parameters)
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    plt.tight_layout()
    return fig

def plot_grid_analysis(grid: Grid2D, show_statistics: bool = True) -> Figure:
    """绘制网格分析图
    
    详细展示网格的分布特征，包括网格间距统计和网格密度分布。
    
    Args:
        grid: 二维网格对象
        show_statistics: 是否显示网格统计信息
        
    Returns:
        Figure: matplotlib图形对象
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 网格分布图
    ax1 = plt.subplot(131)
    grid.plot(ax=ax1, show_grid=True)
    ax1.set_title('网格分布')
    
    # x方向网格间距分布
    ax2 = plt.subplot(132)
    ax2.hist(grid.dx, bins=20, alpha=0.7)
    ax2.set_xlabel('x方向网格间距 (m)')
    ax2.set_ylabel('频数')
    ax2.set_title('x方向网格间距分布')
    
    # z方向网格间距分布
    ax3 = plt.subplot(133)
    ax3.hist(grid.dz, bins=20, alpha=0.7)
    ax3.set_xlabel('z方向网格间距 (m)')
    ax3.set_ylabel('频数')
    ax3.set_title('z方向网格间距分布')
    
    if show_statistics:
        stats_text = f'网格统计信息:\n'
        stats_text += f'x方向: {len(grid.x)}个节点\n'
        stats_text += f'最小间距: {grid.dx.min():.2f}m\n'
        stats_text += f'最大间距: {grid.dx.max():.2f}m\n'
        stats_text += f'z方向: {len(grid.z)}个节点\n'
        stats_text += f'最小间距: {grid.dz.min():.2f}m\n'
        stats_text += f'最大间距: {grid.dz.max():.2f}m'
        
        fig.text(0.98, 0.98, stats_text, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_forward_results(model: EMModel2D, response: np.ndarray,
                       frequencies: np.ndarray, offsets: np.ndarray,
                       show_model: bool = True) -> Figure:
    if show_model:
        fig = plt.figure(figsize=(15, 12))
        
        # 模型结构
        ax1 = plt.subplot(221)
        model.plot(ax=ax1, parameter='conductivity')
        ax1.set_title('模型电导率分布')
        
        # 幅值曲线
        ax2 = plt.subplot(222)
        for i, freq in enumerate(frequencies):
            ax2.semilogx(offsets, np.abs(response[i, :]),
                        label=f'{freq:.1f} Hz')
        ax2.set_xlabel('偏移距 (m)')
        ax2.set_ylabel('幅值 (V/m)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('幅值-偏移距曲线')
        
        # 相位曲线
        ax3 = plt.subplot(223)
        for i, freq in enumerate(frequencies):
            ax3.semilogx(offsets, np.angle(response[i, :], deg=True),
                        label=f'{freq:.1f} Hz')
        ax3.set_xlabel('偏移距 (m)')
        ax3.set_ylabel('相位 (度)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('相位-偏移距曲线')
        
        # 幅值分布图
        ax4 = plt.subplot(224)
        X, Y = np.meshgrid(offsets, frequencies)
        im = ax4.pcolormesh(X, Y, np.abs(response),
                           shading='auto', cmap='viridis')
        ax4.set_xlabel('偏移距 (m)')
        ax4.set_ylabel('频率 (Hz)')
        plt.colorbar(im, ax=ax4, label='幅值 (V/m)')
        ax4.set_title('幅值分布图')
        
        # 调整子图间距
        plt.subplots_adjust(wspace=0.4, hspace=0.284)
    
    else:
        fig = plt.figure(figsize=(15, 5))
        
        # 幅值曲线
        ax1 = plt.subplot(131)
        for i, freq in enumerate(frequencies):
            ax1.semilogx(offsets, np.abs(response[i, :]),
                        label=f'{freq:.1f} Hz')
        ax1.set_xlabel('偏移距 (m)')
        ax1.set_ylabel('幅值 (V/m)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('幅值-偏移距曲线')
        
        # 相位曲线
        ax2 = plt.subplot(132)
        for i, freq in enumerate(frequencies):
            ax2.semilogx(offsets, np.angle(response[i, :], deg=True),
                        label=f'{freq:.1f} Hz')
        ax2.set_xlabel('偏移距 (m)')
        ax2.set_ylabel('相位 (度)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('相位-偏移距曲线')
        
        # 幅值分布图
        ax3 = plt.subplot(133)
        X, Y = np.meshgrid(offsets, frequencies)
        im = ax3.pcolormesh(X, Y, np.abs(response),
                           shading='auto', cmap='viridis')
        ax3.set_xlabel('偏移距 (m)')
        ax3.set_ylabel('频率 (Hz)')
        plt.colorbar(im, ax=ax3, label='幅值 (V/m)')
        ax3.set_title('幅值分布图')
        
        # 调整子图间距
        plt.subplots_adjust(wspace=0.4)
    
    plt.tight_layout()
    return fig