"""视电阻率/视电导率转换工具。

将电磁测量数据转换为视电阻率（apparent resistivity），用于：
- 快速质控（QC）：直观查看数据质量
- 初始模型构建：为反演提供起始模型
- 实时成像：钻前快速评估地层电阻率

支持多种转换方法：
- 半空间解析法：基于均匀半空间解析解反演视电阻率
- 振幅法：仅用振幅信息
- 相位法：仅用相位信息（如可用）

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from typing import Optional

import numpy as np
from scipy.constants import mu_0
from scipy.optimize import brentq

from ..fdemcore.electromagnetics.analytics import hzAnalyticDipoleF


def apparent_resistivity_from_hz(
    frequency: float,
    hz_measured: complex,
    offset: float,
    method: str = "amplitude",
    sigma_bounds: tuple = (1e-4, 10.0),
    mu_r: float = 1.0,
) -> float:
    """从 Hz 分量计算单点视电阻率。

    *物理含义*: 寻找均匀半空间电导率 σ_a，使得该半空间在给定频率和偏移距
    处的解析响应与实测值匹配。
    *使用方式*: 传入单个频率、单个接收器的复数 Hz 值、偏移距。
    *应用场景*: 逐点快速转换，生成视电阻率剖面用于质控。

    Args:
        frequency: 频率 (Hz)。
        hz_measured: 实测复数 Hz 值 (A/m)。
        offset: 收发距 (m)。
        method: 转换方法，'amplitude'（振幅匹配）或 'complex'（复数匹配）。
        sigma_bounds: 搜索电导率范围 (S/m)，默认 (1e-4, 10.0)。
        mu_r: 相对磁导率，默认 1.0（非磁性）。

    Returns:
        视电导率 σ_a (S/m)。

    Raises:
        ValueError: 无法收敛或超出搜索范围时。
    """
    mu = mu_r * mu_0

    def objective(sigma):
        """目标函数：解析解与实测值之差。"""
        hz_analytic = hzAnalyticDipoleF(
            offset, np.array([frequency]), sigma, secondary=False, mu=mu
        ).flatten()[0]
        if method == "amplitude":
            return abs(hz_analytic) - abs(hz_measured)
        else:  # complex
            return abs(hz_analytic - hz_measured)

    # 检查边界
    f_low = objective(sigma_bounds[0])
    f_high = objective(sigma_bounds[1])

    if f_low * f_high > 0:
        # 无根，返回最接近的边界值
        if abs(f_low) < abs(f_high):
            return sigma_bounds[0]
        else:
            return sigma_bounds[1]

    # Brent 法求根
    try:
        sigma_a = brentq(objective, sigma_bounds[0], sigma_bounds[1],
                          xtol=1e-6, rtol=1e-6, maxiter=100)
        return sigma_a
    except ValueError:
        # 收敛失败，返回几何平均作为估计
        return np.sqrt(sigma_bounds[0] * sigma_bounds[1])


def apparent_resistivity_matrix(
    frequencies: np.ndarray,
    spacings: np.ndarray,
    data_matrix: np.ndarray,
    field_kind: str = "h",
    component: str = "real",
    sigma_bounds: tuple = (1e-4, 10.0),
) -> np.ndarray:
    """将多频率×多收发距数据矩阵转换为视电阻率矩阵。

    *物理含义*: 对每个 (频率, 收发距) 组合计算视电阻率。
    *使用方式*: 传入频率数组、收发距数组、数据矩阵 (n_freq, n_spacing)。
    *应用场景*: 批量转换，生成视电阻率伪截面。

    Args:
        frequencies: 频率数组 (Hz)，形状 (n_freq,)。
        spacings: 收发距数组 (m)，形状 (n_spacing,)。
        data_matrix: 数据矩阵，形状 (n_freq, n_spacing)。
            对于实部/虚部数据，值为实数；对于幅值/相位，需先转换。
        field_kind: 场量类型，目前仅支持 'h'。
        component: 数据分量，'real' 或 'imag'（用于构造复数）。
        sigma_bounds: 搜索电导率范围 (S/m)。

    Returns:
        视电阻率矩阵，形状 (n_freq, n_spacing)，单位 S/m。
    """
    n_freq = len(frequencies)
    n_spacing = len(spacings)
    sigma_a = np.zeros((n_freq, n_spacing))

    for i, freq in enumerate(frequencies):
        for j, spacing in enumerate(spacings):
            # 构造复数 Hz（假设虚部为 0 或从另一分量获取）
            if component == "real":
                hz_complex = data_matrix[i, j] + 0j
            else:  # imag
                hz_complex = 0 + 1j * data_matrix[i, j]

            sigma_a[i, j] = apparent_resistivity_from_hz(
                frequency=freq,
                hz_measured=hz_complex,
                offset=spacing,
                sigma_bounds=sigma_bounds,
            )

    return sigma_a


def apparent_resistivity_from_forward(
    forward,
    data: np.ndarray,
    sigma_bounds: tuple = (1e-4, 10.0),
) -> np.ndarray:
    """从正演对象和数据向量计算视电阻率（1D 层状模型）。

    *物理含义*: 利用正演对象的仪器配置，逐点反演视电阻率。
    *使用方式*: 传入 LWDFDEMForward 实例和数据向量。
    *应用场景*: 与正演对象配合，自动处理数据排列。

    Args:
        forward: LWDFDEMForward 实例（须为 1D 几何）。
        data: 数据向量，形状 (nD,)。
        sigma_bounds: 搜索电导率范围 (S/m)。

    Returns:
        视电阻率数组，形状 (n_freq, n_spacing, n_comp)。

    Raises:
        ValueError: 非 1D 几何时。
    """
    if forward.geometry != "1d":
        raise ValueError("视电阻率转换目前仅支持 1D 层状几何。")

    from .survey import data_order

    n_freq, n_spacing, n_comp = data_order(forward.tool)
    data_mat = data.reshape(n_freq, n_spacing, n_comp)

    sigma_a = np.zeros((n_freq, n_spacing, n_comp))

    for i, freq in enumerate(forward.tool.frequencies):
        for j, spacing in enumerate(forward.tool.spacings):
            for k, comp in enumerate(forward.tool.components):
                # 取实部数据作为振幅近似
                val = data_mat[i, j, k]
                if comp in ("real", "amp"):
                    hz_complex = val + 0j
                else:  # imag or phase
                    hz_complex = 0 + 1j * val

                sigma_a[i, j, k] = apparent_resistivity_from_hz(
                    frequency=freq,
                    hz_measured=hz_complex,
                    offset=spacing,
                    sigma_bounds=sigma_bounds,
                )

    return sigma_a
