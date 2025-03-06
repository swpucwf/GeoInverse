import numpy as np
from typing import List, Tuple, Optional

class ForwardSolver:
    """一维电磁波正演模拟类
    
    实现基于解析解的一维分层介质中电磁波传播特性计算
    支持多频率、多偏移距的电磁场计算
    
    Attributes:
        frequencies (np.ndarray): 工作频率数组 (Hz)
        offsets (np.ndarray): 发射源与接收器之间的偏移距数组 (m)
    """
    
    def __init__(self, frequencies: List[float], offsets: List[float]):
        """初始化正演模拟器
        
        Args:
            frequencies: 工作频率列表 (Hz)
            offsets: 偏移距列表 (m)
        """
        self.frequencies = np.array(frequencies)
        self.offsets = np.array(offsets)
        
    def calculate_wavenumber(self, frequency: float, resistivity: float,
                            relative_permittivity: float = 1.0,
                            relative_permeability: float = 1.0) -> complex:
        """计算介质中的波数
        
        Args:
            frequency: 频率 (Hz)
            resistivity: 电阻率 (Ω·m)
            relative_permittivity: 相对介电常数
            relative_permeability: 相对磁导率
            
        Returns:
            complex: 复波数
        """
        omega = 2 * np.pi * frequency
        mu0 = 4 * np.pi * 1e-7  # 真空磁导率
        epsilon0 = 8.854e-12    # 真空介电常数
        
        mu = mu0 * relative_permeability
        epsilon = epsilon0 * relative_permittivity
        sigma = 1.0 / resistivity
        
        return np.sqrt(omega**2 * mu * epsilon - 1j * omega * mu * sigma)
    
    def calculate_reflection_coefficient(self, k1: complex, k2: complex) -> complex:
        """计算两层介质界面的反射系数
        
        Args:
            k1: 上层介质波数
            k2: 下层介质波数
            
        Returns:
            complex: 反射系数
        """
        return (k1 - k2) / (k1 + k2)
    
    def forward(self, layers: List[Tuple[float, float]]) -> np.ndarray:
        """计算分层介质中的电磁场响应
        
        Args:
            layers: 地层模型，每个元素为(厚度, 电阻率)的元组
                   最后一层厚度可以为None，表示半无限空间
        
        Returns:
            np.ndarray: 形状为(频率数, 偏移距数)的复数数组，表示电磁场响应
        """
        nf = len(self.frequencies)
        nr = len(self.offsets)
        response = np.zeros((nf, nr), dtype=complex)
        
        for i, freq in enumerate(self.frequencies):
            # 计算每一层的波数
            k_list = [self.calculate_wavenumber(freq, resistivity)
                     for _, resistivity in layers]
            
            for j, offset in enumerate(self.offsets):
                # 计算初始场
                primary_field = np.exp(-k_list[0] * offset) / offset
                
                # 计算反射场
                total_field = primary_field
                for layer_idx in range(len(layers)-1):
                    r = self.calculate_reflection_coefficient(
                        k_list[layer_idx], k_list[layer_idx+1])
                    thickness = layers[layer_idx][0]
                    if thickness is not None:
                        path_length = 2 * thickness + offset
                        reflected_field = r * np.exp(-k_list[0] * path_length) / path_length
                        total_field += reflected_field
                
                response[i, j] = total_field
        
        return response
    
    def compute_response(self, model):
        """计算地层模型的电磁场响应
        
        Args:
            model: LayeredEarthModel对象，表示分层地球模型
            
        Returns:
            np.ndarray: 形状为(频率数, 偏移距数)的复数数组，表示电磁场响应
        """
        # 将LayeredEarthModel转换为forward方法需要的格式
        layers = []
        for layer in model.layers:
            layers.append((layer.thickness, layer.resistivity))
            
        # 调用forward方法计算响应
        return self.forward(layers)
    
    def plot_response(self, response: np.ndarray,
                      plot_type: str = 'amplitude',
                      ax: Optional[np.ndarray] = None) -> None:
        """绘制电磁场响应
        
        Args:
            response: 电磁场响应数组
            plot_type: 绘制类型，'amplitude'表示幅值，'phase'表示相位
            ax: matplotlib轴对象，如果为None则创建新的图形
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots()
            
        if plot_type == 'amplitude':
            for i, freq in enumerate(self.frequencies):
                ax.semilogx(self.offsets, np.abs(response[i, :]),
                           label=f'{freq} Hz')
            ax.set_ylabel('幅值 (V/m)')
        else:
            for i, freq in enumerate(self.frequencies):
                ax.semilogx(self.offsets, np.angle(response[i, :], deg=True),
                           label=f'{freq} Hz')
            ax.set_ylabel('相位 (度)')
            
        ax.set_xlabel('偏移距 (m)')
        ax.grid(True)
        ax.legend()