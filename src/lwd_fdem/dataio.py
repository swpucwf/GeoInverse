"""LWD 测井数据 I/O 与不确定度工具。

负责观测数据数组与 fdemcore `data.Data` 之间的转换（决策清单 H5），
以及合成数据加噪、不确定度构建等辅助功能。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from typing import Dict, Tuple

import numpy as np

from ..fdemcore.data import Data
from ..fdemcore.electromagnetics.frequency_domain import Survey


def add_synthetic_noise(d_clean: np.ndarray, noise_level: float,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """对清洁响应施加相对高斯噪声，模拟测量数据。

    Args:
        d_clean: 正演清洁数据 (nD,)。
        noise_level: 相对噪声水平（如 0.05 表示 5%）。
        seed: 随机种子，保证可复现。

    Returns:
        (dobs, std)：观测数据与逐点标准差，形状均为 (nD,)。
    """
    if noise_level <= 0:
        raise ValueError("noise_level 必须为正。")
    rng = np.random.default_rng(seed)
    d_clean = np.asarray(d_clean, dtype=float)
    noise = noise_level * np.abs(d_clean) * rng.standard_normal(d_clean.size)
    dobs = d_clean + noise
    std = noise_level * np.abs(dobs)
    return dobs, std


def make_data_object(survey: Survey, dobs: np.ndarray,
                     relative_error: float = 0.05,
                     noise_floor: float = 0.0,
                     absolute_std: np.ndarray | None = None) -> Data:
    """构建带不确定度的观测数据对象。

    不确定度默认取 `relative_error * |dobs| + noise_floor`；
    若给出 absolute_std（形状 (nD,)）则直接作为逐点标准差。

    Args:
        survey: 对应的测量对象。
        dobs: 观测数据向量 (nD,)。
        relative_error: 相对误差水平。
        noise_floor: 绝对噪声底（与数据同单位）。
        absolute_std: 可选的逐点标准差。

    Returns:
        fdemcore data.Data 对象。
    """
    dobs = np.asarray(dobs, dtype=float)
    if absolute_std is not None:
        std = np.asarray(absolute_std, dtype=float)
    else:
        std = relative_error * np.abs(dobs) + noise_floor
    if np.any(std <= 0):
        raise ValueError("存在非正的数据标准差，请检查 relative_error/noise_floor。")
    data_obj = Data(survey, dobs=dobs)
    data_obj.standard_deviation = std
    return data_obj


def load_lwd_csv(path: str) -> Dict[str, np.ndarray]:
    """读取长表格式的 LWD 测量文件。

    CSV 需含表头：`frequency,spacing,component,value`，其中
    frequency 单位 Hz、spacing 单位 m、component 为 real/imag、value 为读数。

    Args:
        path: 文件路径。

    Returns:
        字典，键 'frequency' / 'spacing' / 'component' / 'value'
        对应各列数组（component 列为字符串数组）。
    """
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {"frequency", "spacing", "component", "value"}
    if raw.dtype.names is None or not required.issubset(set(raw.dtype.names)):
        raise ValueError(f"CSV 需包含表头列：{sorted(required)}，Got {raw.dtype.names}")
    return {
        "frequency": np.asarray(raw["frequency"], dtype=float),
        "spacing": np.asarray(raw["spacing"], dtype=float),
        "component": np.asarray(raw["component"], dtype=str),
        "value": np.asarray(raw["value"], dtype=float),
    }
