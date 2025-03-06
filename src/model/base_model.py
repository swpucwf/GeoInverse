from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

class BaseModel(ABC):
    """地球物理模型基类
    
    这是所有地球物理模型的抽象基类，定义了通用的接口和属性。所有具体的地球物理模型类
    都应该继承这个基类并实现其抽象方法。
    
    Attributes:
        name (str): 模型名称，用于标识和区分不同的模型实例
        
    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self, name="MyModel"):
        ...         super().__init__(name)
        ...     
        ...     def get_parameters(self):
        ...         return (np.array([1.0]),)
        ...     
        ...     def plot(self, ax=None):
        ...         pass
        ...     
        ...     def validate(self):
        ...         return True
    """
    
    def __init__(self, name: str = "BaseModel"):
        """初始化基类
        
        Args:
            name (str): 模型名称，默认为"BaseModel"
        """
        self.name = name
    
    @abstractmethod
    def get_parameters(self) -> Tuple[np.ndarray, ...]:
        """获取模型参数
        
        这个抽象方法需要被子类实现，用于返回模型的所有参数。返回的参数应该是numpy数组的元组，
        每个数组代表一类参数（如厚度、电阻率等）。
        
        Returns:
            Tuple[np.ndarray, ...]: 模型参数元组，每个元素是一个numpy数组
            
        Raises:
            NotImplementedError: 当子类没有实现这个方法时
        """
        pass
    
    @abstractmethod
    def plot(self, ax=None, **kwargs) -> None:
        """绘制模型
        
        这个抽象方法需要被子类实现，用于可视化模型。子类应该实现适合自己特点的可视化方式。
        
        Args:
            ax (matplotlib.axes.Axes, optional): matplotlib轴对象，如果为None则创建新的图形
            **kwargs: 其他绘图参数，具体参数由子类定义
            
        Raises:
            NotImplementedError: 当子类没有实现这个方法时
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """验证模型参数的有效性
        
        这个抽象方法需要被子类实现，用于检查模型参数是否有效（如参数是否在合理范围内）。
        
        Returns:
            bool: 如果所有参数都有效则返回True，否则返回False
            
        Raises:
            NotImplementedError: 当子类没有实现这个方法时
        """
        pass
    
    def __str__(self) -> str:
        """返回模型的字符串表示
        
        Returns:
            str: 模型的简要描述，包含模型名称
        """
        return f"{self.name} Model"