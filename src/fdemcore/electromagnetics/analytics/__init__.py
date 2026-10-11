"""FDEM 解析解模块。

提供全空间偶极子与层状介质 Hz 解析解，用于正演验证与 benchmark。
当前仅包含 FDEM 相关函数（TDEM/DC/NSEM 不在本项目范围）。
"""
from .FDEM import hzAnalyticDipoleF, MagneticDipoleWholeSpace, ElectricDipoleWholeSpace

__all__ = [
    "hzAnalyticDipoleF",
    "MagneticDipoleWholeSpace",
    "ElectricDipoleWholeSpace",
]
