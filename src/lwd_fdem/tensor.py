"""多分量（张量）接收与多分量源阵列封装。

支持三轴接收器（Hx/Hy/Hz 或 Bx/By/Bz）和三轴发射源阵列，
用于张量电磁测量与方位敏感反演。底层复用 fdemcore 点接收器/
磁偶极子源，通过不同 orientation 组合实现多分量观测。

数据排列顺序：频率 → 源取向 → 收发距 → 接收取向 → 分量。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from typing import List, Sequence, Tuple

import numpy as np

from ..fdemcore.electromagnetics.frequency_domain import receivers as _rx
from ..fdemcore.electromagnetics.frequency_domain import sources as _src
from ..fdemcore.electromagnetics.frequency_domain import Survey
from ..fdemcore.utils import validate_string
from .survey import _RX_CLASSES, _SUPPORTED, _receiver_location

# 默认三轴取向
_TRIAD = ("x", "y", "z")


def build_tensor_receivers(
    spacings: np.ndarray,
    field_kind: str = "h",
    secondary: bool = False,
    components: Sequence[str] = ("real", "imag"),
    data_type: str = "field",
    receiver_geometry: str = "axial",
    rx_orientations: Sequence[str] = _TRIAD,
    form: str = "h",
) -> List:
    """构建多取向接收器列表（张量接收）。

    *物理含义*: 在同一收发距处放置多个不同轴向的接收器，
    同时记录场的多个分量（如 Hx, Hy, Hz），实现张量测量。
    *使用方式*: 指定接收取向列表，函数为每个 (收发距, 取向, 分量) 组合
    创建一个接收器对象。
    *应用场景*: 三轴电磁仪器、方位敏感测量、各向异性探测。

    Args:
        spacings: 收发距数组 (m)。
        field_kind: 场量类型，'h'/'b'/'e'/'j'。
        secondary: 是否二次场。
        components: 数据分量，'real'/'imag'/'amp'/'phase'。
        data_type: 'field' 或 'ppm'。
        receiver_geometry: 'axial' 或 'radial'。
        rx_orientations: 接收器取向列表，默认 ('x','y','z') 三轴。
        form: 离散形式，用于兼容性校验。

    Returns:
        接收器列表，排序：收发距 → 接收取向 → 分量。

    Raises:
        ValueError: 场量/取向/离散形式不兼容时。
    """
    # 兼容性校验（用第一个取向检查即可）
    key = (field_kind, secondary)
    if key not in _SUPPORTED[form]:
        raise ValueError(
            f"接收器 (field={field_kind!r}, secondary={secondary!r}) "
            f"与离散形式 {form!r} 不兼容。"
        )
    for ori in rx_orientations:
        validate_string("rx_orientations", ori, string_list=("x", "y", "z"))

    rx_cls = _RX_CLASSES[key]
    receiver_list = []
    for spacing in np.asarray(spacings, dtype=float):
        loc = _receiver_location_simple(receiver_geometry, float(spacing))
        for rx_ori in rx_orientations:
            for comp in components:
                receiver_list.append(
                    rx_cls(loc, orientation=rx_ori, component=comp,
                           data_type=data_type)
                )
    return receiver_list


def build_tensor_survey(
    frequencies: Sequence[float],
    spacings: np.ndarray,
    field_kind: str = "h",
    secondary: bool = False,
    components: Sequence[str] = ("real", "imag"),
    data_type: str = "field",
    receiver_geometry: str = "axial",
    rx_orientations: Sequence[str] = _TRIAD,
    src_orientations: Sequence[str] = ("z",),
    moment: float = 1.0,
    source_location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    form: str = "h",
) -> Survey:
    """构建多分量源 × 多分量接收的张量 Survey。

    *物理含义*: 每个频率下，为每个源取向创建一个独立源，所有源共享
    同一组多取向接收器。实现三轴发射 × 三轴接收的全张量测量。
    *使用方式*: 指定源取向列表和接收取向列表。
    *应用场景*: 三轴阵列仪器、交叉耦合测量、各向异性地层探测。

    数据排列顺序：频率 → 源取向 → 收发距 → 接收取向 → 分量。

    Args:
        frequencies: 工作频率列表 (Hz)。
        spacings: 收发距数组 (m)。
        field_kind: 场量类型，'h'/'b'/'e'/'j'。
        secondary: 是否二次场。
        components: 数据分量。
        data_type: 'field' 或 'ppm'。
        receiver_geometry: 'axial' 或 'radial'。
        rx_orientations: 接收器取向列表，默认 ('x','y','z')。
        src_orientations: 源取向列表，默认 ('z',) 单轴。
        moment: 磁偶极矩 (A·m²)。
        source_location: 源位置 (m)。
        form: 离散形式，用于兼容性校验。

    Returns:
        fdem.Survey 对象。

    Raises:
        ValueError: 参数不合法时。
    """
    for ori in src_orientations:
        validate_string("src_orientations", ori, string_list=("x", "y", "z"))

    receiver_list = build_tensor_receivers(
        spacings=spacings,
        field_kind=field_kind,
        secondary=secondary,
        components=components,
        data_type=data_type,
        receiver_geometry=receiver_geometry,
        rx_orientations=rx_orientations,
        form=form,
    )

    src_loc = np.asarray(source_location, dtype=float)
    source_list = []
    for freq in frequencies:
        for src_ori in src_orientations:
            source_list.append(
                _src.MagDipole(
                    receiver_list,
                    frequency=float(freq),
                    location=src_loc,
                    orientation=src_ori,
                    moment=moment,
                )
            )
    return Survey(source_list)


def tensor_data_order(
    frequencies: Sequence,
    src_orientations: Sequence[str] = ("z",),
    spacings: np.ndarray = None,
    rx_orientations: Sequence[str] = _TRIAD,
    components: Sequence[str] = ("real", "imag"),
) -> dict:
    """返回张量数据的各维度大小。

    *物理含义*: 描述张量数据向量的多维结构。
    *使用方式*: 传入与 build_tensor_survey 相同的参数。
    *应用场景*: 将一维数据向量 reshape 为多维数组进行分析。

    Args:
        frequencies: 频率列表。
        src_orientations: 源取向列表。
        spacings: 收发距数组。
        rx_orientations: 接收取向列表。
        components: 分量列表。

    Returns:
        字典，含 'n_freq', 'n_src_ori', 'n_spacing', 'n_rx_ori',
        'n_comp', 'total' 及各维度名称列表。
    """
    n_freq = len(frequencies)
    n_src_ori = len(src_orientations)
    n_spacing = len(spacings) if spacings is not None else 0
    n_rx_ori = len(rx_orientations)
    n_comp = len(components)
    return {
        "n_freq": n_freq,
        "n_src_ori": n_src_ori,
        "n_spacing": n_spacing,
        "n_rx_ori": n_rx_ori,
        "n_comp": n_comp,
        "total": n_freq * n_src_ori * n_spacing * n_rx_ori * n_comp,
        "shape": (n_freq, n_src_ori, n_spacing, n_rx_ori, n_comp),
        "axes": ["frequency", "src_orientation", "spacing",
                 "rx_orientation", "component"],
    }


def _receiver_location_simple(geometry: str, spacing: float) -> np.ndarray:
    """按 geometry 计算接收器坐标（源在原点）。"""
    if geometry == "axial":
        return np.array([0.0, 0.0, spacing])
    return np.array([spacing, 0.0, 0.0])
