import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class Layer:
    """地层模型的单层参数类
    
    Attributes:
        thickness: 层厚度 (m)，对于半无限空间可以为None
        resistivity: 电阻率 (Ω·m)
        relative_permittivity: 相对介电常数
        relative_permeability: 相对磁导率
    """
    thickness: Optional[float]
    resistivity: float
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0

class LayeredEarthModel:
    """分层地球模型类
    
    用于定义和管理分层地球模型的参数，支持参数化描述和可视化
    
    Attributes:
        layers: 地层列表，从上到下排列
    """
    
    def __init__(self, layers: List[Layer]):
        """初始化分层地球模型
        
        Args:
            layers: 地层参数列表，从上到下排列
                   最后一层的thickness应为None，表示半无限空间
        """
        if not layers:
            raise ValueError("地层列表不能为空")
        if layers[-1].thickness is not None:
            raise ValueError("最后一层必须是半无限空间（thickness=None）")
        
        self.layers = layers
        
    @property
    def n_layers(self) -> int:
        """地层数量"""
        return len(self.layers)
    
    @property
    def depths(self) -> np.ndarray:
        """计算每层顶部的深度"""
        depths = [0.0]  # 第一层顶部深度为0
        current_depth = 0.0
        
        for layer in self.layers[:-1]:  # 不包括最后一层（半无限空间）
            if layer.thickness is not None:
                current_depth += layer.thickness
                depths.append(current_depth)
        
        return np.array(depths)
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取模型参数数组
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                thicknesses: 各层厚度数组
                resistivities: 各层电阻率数组
                permittivities: 各层相对介电常数数组
                permeabilities: 各层相对磁导率数组
        """
        thicknesses = []
        resistivities = []
        permittivities = []
        permeabilities = []
        
        for layer in self.layers:
            thicknesses.append(layer.thickness if layer.thickness is not None else np.inf)
            resistivities.append(layer.resistivity)
            permittivities.append(layer.relative_permittivity)
            permeabilities.append(layer.relative_permeability)
        
        return (np.array(thicknesses), np.array(resistivities),
                np.array(permittivities), np.array(permeabilities))
    
    def plot(self, ax=None, depth_range: Optional[Tuple[float, float]] = None,
            show_parameters: bool = True) -> None:
        """绘制地层模型
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新的图形
            depth_range: 绘图的深度范围，格式为(min_depth, max_depth)
            show_parameters: 是否显示地层参数
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        depths = self.depths
        if depth_range is None:
            if len(depths) > 1:
                max_depth = depths[-1] * 1.2  # 延伸20%以显示最后一层
            else:
                max_depth = 100  # 默认深度范围
            depth_range = (0, max_depth)
        
        # 绘制地层界面
        for i, depth in enumerate(depths):
            if depth <= depth_range[1]:
                ax.axhline(y=depth, color='k', linestyle='-', linewidth=0.5)
        
        # 填充地层颜色
        cmap = plt.cm.viridis  # 使用viridis颜色映射
        resistivities = np.array([layer.resistivity for layer in self.layers])
        norm = plt.Normalize(vmin=np.log10(min(resistivities)),
                           vmax=np.log10(max(resistivities)))
        
        for i in range(len(depths)):
            y_top = depths[i]
            y_bottom = depths[i+1] if i < len(depths)-1 else depth_range[1]
            color = cmap(norm(np.log10(self.layers[i].resistivity)))
            ax.fill_between([-1, 1], y_top, y_bottom, color=color, alpha=0.5)
            
            # 添加参数标注
            if show_parameters:
                y_text = (y_top + y_bottom) / 2
                text = f"ρ={self.layers[i].resistivity:.1f} Ω·m"
                if self.layers[i].relative_permittivity != 1.0:
                    text += f"\nεr={self.layers[i].relative_permittivity:.1f}"
                if self.layers[i].relative_permeability != 1.0:
                    text += f"\nμr={self.layers[i].relative_permeability:.1f}"
                ax.text(0, y_text, text, ha='center', va='center')
        
        # 设置坐标轴
        ax.set_ylim(depth_range[1], depth_range[0])  # 反转y轴方向
        ax.set_xlim(-1, 1)
        ax.set_ylabel('深度 (m)')
        ax.set_xticks([])
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='log10(电阻率 Ω·m)')