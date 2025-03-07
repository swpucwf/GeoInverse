import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from .base_model import BaseModel

@dataclass
class Layer:
    """地层模型的单层参数类
    
    用于描述分层地球模型中单个地层的物理参数，包括厚度、电阻率、相对介电常数和相对磁导率。
    最后一层（半无限空间）的厚度应设置为None。
    
    Attributes:
        thickness (Optional[float]): 层厚度，单位为米(m)。对于半无限空间（最后一层）应设置为None
        resistivity (float): 地层电阻率，单位为欧姆·米(Ω·m)，必须大于0
        relative_permittivity (float): 地层相对介电常数，无量纲，默认为1.0，必须大于0
        relative_permeability (float): 地层相对磁导率，无量纲，默认为1.0，必须大于0
        
    Example:
        >>> # 创建一个20米厚、电阻率为100的地层
        >>> layer1 = Layer(thickness=20.0, resistivity=100.0)
        >>> # 创建一个半无限空间（最后一层），电阻率为500
        >>> layer2 = Layer(thickness=None, resistivity=500.0)
    """
    thickness: Optional[float]
    resistivity: float
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0

class LayeredEarthModel(BaseModel):
    """分层地球模型类
    
    用于定义和管理分层地球模型的参数，支持参数化描述和可视化。模型由多个地层组成，
    每个地层具有其独特的物理参数（厚度、电阻率等）。最后一层必须是半无限空间（厚度为None）。
    
    Attributes:
        layers (List[Layer]): 地层列表，从上到下排列，最后一层必须是半无限空间
        name (str): 模型名称，用于标识不同的模型实例
        
    Example:
        >>> # 创建一个三层地球模型
        >>> layer1 = Layer(thickness=10.0, resistivity=100.0)
        >>> layer2 = Layer(thickness=20.0, resistivity=50.0)
        >>> layer3 = Layer(thickness=None, resistivity=500.0)  # 半无限空间
        >>> model = LayeredEarthModel([layer1, layer2, layer3])
        
        >>> # 使用深度和电阻率列表创建模型
        >>> depths = [0.0, 10.0, 30.0, np.inf]  # 层界面深度
        >>> resistivities = [100.0, 50.0, 500.0]  # 各层电阻率
        >>> model = LayeredEarthModel.from_depths_resistivities(depths, resistivities)
    """
    
    def __init__(self, layers: List[Layer], name: str = "LayeredEarth"):
        """初始化分层地球模型
        
        Args:
            layers (List[Layer]): 地层参数列表，从上到下排列。最后一层的thickness必须为None，
                                表示半无限空间
            name (str): 模型名称，默认为"LayeredEarth"
            
        Raises:
            ValueError: 当地层列表为空或最后一层不是半无限空间时
        """
        super().__init__(name)
        if not layers:
            raise ValueError("地层列表不能为空")
        if layers[-1].thickness is not None:
            raise ValueError("最后一层必须是半无限空间（thickness=None）")
        
        self.layers = layers
        
    @classmethod
    def from_depths_resistivities(cls, depths: List[float], resistivities: List[float]):
        """从深度列表和电阻率列表创建地层模型
        
        Args:
            depths (List[float]): 层界面深度列表，从上到下排列，第一个元素应为0.0，
                                最后一个元素应为np.inf
            resistivities (List[float]): 各层电阻率列表，长度应比depths少1
            
        Returns:
            LayeredEarthModel: 创建的地层模型对象
            
        Raises:
            ValueError: 当深度列表长度不正确或最后一层深度不是无限时
            
        Example:
            >>> depths = [0.0, 10.0, 30.0, np.inf]
            >>> resistivities = [100.0, 50.0, 500.0]
            >>> model = LayeredEarthModel.from_depths_resistivities(depths, resistivities)
        """
        if len(depths) != len(resistivities) + 1:
            raise ValueError("深度列表长度应比电阻率列表长度多1")
        if depths[-1] != np.inf:
            raise ValueError("最后一层深度应为无限(np.inf)")
            
        layers = []
        for i in range(len(resistivities)):
            thickness = depths[i+1] - depths[i] if depths[i+1] != np.inf else None
            layers.append(Layer(thickness=thickness, resistivity=resistivities[i]))
            
        return cls(layers)
        
    @property
    def n_layers(self) -> int:
        """地层数量
        
        Returns:
            int: 模型中的地层总数
        """
        return len(self.layers)
    
    @property
    def depths(self) -> np.ndarray:
        """计算每层顶部的深度
        
        Returns:
            np.ndarray: 各层顶部深度数组，从上到下排列，第一个元素为0.0
        """
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
                - thicknesses: 各层厚度数组，最后一层为np.inf
                - resistivities: 各层电阻率数组
                - permittivities: 各层相对介电常数数组
                - permeabilities: 各层相对磁导率数组
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
        
        # 使用更鲜明的颜色映射
        cmap = plt.cm.Set3  # 使用Set3颜色映射，颜色更加分明
        resistivities = np.array([layer.resistivity for layer in self.layers])
        
        # 绘制地层填充
        for i in range(len(depths)):
            y_top = depths[i]
            y_bottom = depths[i+1] if i < len(depths)-1 else depth_range[1]
            color = cmap(i % cmap.N)  # 循环使用颜色映射中的颜色
            ax.fill_between([-60, 100], y_top, y_bottom, color=color, alpha=0.7)
            
            # 添加参数标注
            if show_parameters:
                y_text = (y_top + y_bottom) / 2
                text = f"ρ={self.layers[i].resistivity:.1f} Ω·m"
                if self.layers[i].relative_permittivity != 1.0:
                    text += f"\nεr={self.layers[i].relative_permittivity:.1f}"
                if self.layers[i].relative_permeability != 1.0:
                    text += f"\nμr={self.layers[i].relative_permeability:.1f}"
                ax.text(-40, y_text, text, ha='left', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 绘制地层界面
        for i, depth in enumerate(depths):
            if depth <= depth_range[1]:
                ax.axhline(y=depth, color='black', linestyle='-', linewidth=1.0)
        
        # 设置坐标轴
        ax.set_ylim(depth_range[1], depth_range[0])  # 反转y轴方向
        ax.set_xlim(-60, 100)
        ax.set_ylabel('深度 (m)')
        ax.set_xticks([])
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('地层编号')
        cbar.set_ticks(np.linspace(0, 1, len(self.layers)))
        cbar.set_ticklabels([f'层{i+1}' for i in range(len(self.layers))])

    def validate(self) -> bool:
        """验证模型参数的有效性
        
        检查所有地层参数是否在合理的物理范围内，包括：
        1. 地层列表不能为空
        2. 最后一层必须是半无限空间
        3. 所有物理参数必须为正值
        4. 所有有限层的厚度必须为正值
        
        Returns:
            bool: 如果所有参数都有效则返回True，否则返回False
        """
        if not self.layers:
            return False
        if self.layers[-1].thickness is not None:
            return False
        
        for layer in self.layers:
            if layer.resistivity <= 0:
                return False
            if layer.relative_permittivity <= 0:
                return False
            if layer.relative_permeability <= 0:
                return False
            if layer.thickness is not None and layer.thickness <= 0:
                return False
        
        return True