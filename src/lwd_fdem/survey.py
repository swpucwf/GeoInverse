"""LWD 线圈观测系统（Survey）构建器。

将仪器参数（多频率、多收发距、场分量、数据类型）映射为 fdemcore 的
Survey 对象，并校验接收器与离散形式（e/b/h/j-form）的兼容性
（对应决策清单 C/D，类名按 SimPEG 0.25 API 校验）。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ..fdemcore.electromagnetics import frequency_domain as fdem
from ..fdemcore.electromagnetics.frequency_domain import receivers as _rx
from ..fdemcore.electromagnetics.frequency_domain import sources as _src
from ..fdemcore.utils import validate_string

# 各离散形式 Fields 支持的投影场（源：frequency_domain/fields.py 的 _GLoc 与 1D 解析）
# key: 形式, value: 允许的 (field, secondary) 组合
_SUPPORTED: dict = {
    "1d": {("e", False), ("e", True), ("b", False), ("b", True),
           ("h", False), ("h", True), ("j", False), ("j", True)},
    "e": {("e", False), ("b", False), ("b", True), ("h", False), ("j", False)},
    "b": {("e", False), ("b", False), ("b", True), ("h", False), ("j", False)},
    "h": {("h", False), ("h", True), ("b", False), ("j", False), ("e", False)},
    "j": {("h", False), ("h", True), ("b", False), ("j", False), ("e", False)},
}

_RX_CLASSES = {
    ("e", False): _rx.PointElectricField,
    ("b", False): _rx.PointMagneticFluxDensity,
    ("b", True): _rx.PointMagneticFluxDensitySecondary,
    ("h", False): _rx.PointMagneticField,
    ("h", True): _rx.PointMagneticFieldSecondary,
    ("j", False): _rx.PointCurrentDensity,
}


@dataclass
class LWDToolConfig:
    """LWD 仪器与观测系统配置。

    Attributes:
        frequencies: 工作频率列表 (Hz)，每个频率对应一个发射源。
        spacings: 接收器相对发射器的偏移距数组 (m)，全部为正。
        components: 记录的分量，取自 ('real', 'imag', 'amp', 'phase')。
        field_kind: 记录的场量，'h'（磁场）/'b'（磁通密度）/'e'（电场）/'j'（电流密度）。
        secondary: 是否记录二次场（'h' 时仅 h-form/j-form 支持二次场接收器）。
        data_type: 'field'（绝对值）或 'ppm'（相对一次场归一化）。
        orientation: 偶极子/接收线圈轴向，'x'/'y'/'z'；'z' 为沿井轴。
        receiver_geometry: 'axial'（接收器沿井轴 z 偏移，适用 2D/3D）
            或 'radial'（接收器径向 x 偏移；1D 层状必须使用此项）。
        moment: 磁偶极矩 (A·m^2)。
        source_type: 'mag_dipole' 或 'circular_loop'。
        loop_radius: 'circular_loop' 时的线圈半径 (m)。
        source_location: 发射器位置 (x, y, z) (m)。
    """

    frequencies: Tuple[float, ...]
    spacings: np.ndarray
    components: Tuple[str, ...] = ("real", "imag")
    field_kind: str = "h"
    secondary: bool = True
    data_type: str = "field"
    orientation: str = "z"
    receiver_geometry: str = "axial"
    moment: float = 1.0
    source_type: str = "mag_dipole"
    loop_radius: float = 0.05
    source_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """校验配置合法性。"""
        self.frequencies = tuple(float(f) for f in self.frequencies)
        if not self.frequencies or any(f <= 0 for f in self.frequencies):
            raise ValueError("frequencies 必须为非空的正频率列表 (Hz)。")
        self.spacings = np.asarray(self.spacings, dtype=float)
        if self.spacings.ndim != 1 or np.any(self.spacings <= 0):
            raise ValueError("spacings 必须为正的 1D 数组 (m)。")
        for comp in self.components:
            validate_string("components", comp,
                            string_list=("real", "imag", "amp", "phase"))
        validate_string("field_kind", self.field_kind,
                        string_list=("h", "b", "e", "j"))
        validate_string("data_type", self.data_type, string_list=("field", "ppm"))
        validate_string("orientation", self.orientation,
                        string_list=("x", "y", "z"))
        validate_string("receiver_geometry", self.receiver_geometry,
                        string_list=("axial", "radial"))
        validate_string("source_type", self.source_type,
                        string_list=("mag_dipole", "circular_loop"))
        if self.field_kind == "e" and self.secondary:
            raise ValueError("电场接收器仅支持总场（secondary=False）。")
        if self.receiver_geometry == "radial" and self.orientation == "z":
            # 径向偏移 + z 取向偶极子在 1D/轴对称中均合法，无需拦截
            pass


def _receiver_location(cfg: LWDToolConfig, spacing: float) -> np.ndarray:
    """按 receiver_geometry 计算单个接收器坐标（源在原点的相对位置）。"""
    if cfg.receiver_geometry == "axial":
        return np.array([0.0, 0.0, spacing])
    return np.array([spacing, 0.0, 0.0])


def build_receivers(cfg: LWDToolConfig, form: str) -> List:
    """按配置与离散形式构建接收器列表。

    Args:
        cfg: 仪器配置。
        form: 离散形式，'1d'/'e'/'b'/'h'/'j'。

    Returns:
        接收器对象列表（同一源下：先偏移距后分量排序）。

    Raises:
        ValueError: 接收器与离散形式不兼容时。
    """
    key = (cfg.field_kind, cfg.secondary)
    if key not in _SUPPORTED[form]:
        raise ValueError(
            f"接收器 (field={cfg.field_kind!r}, secondary={cfg.secondary!r}) "
            f"与离散形式 {form!r} 不兼容；请先用 dataio/receivers 兼容性表检查。"
        )
    rx_cls = _RX_CLASSES[key]
    receiver_list = []
    for spacing in cfg.spacings:
        loc = _receiver_location(cfg, float(spacing))
        for comp in cfg.components:
            receiver_list.append(
                rx_cls(loc, orientation=cfg.orientation, component=comp,
                       data_type=cfg.data_type)
            )
    return receiver_list


def build_survey(cfg: LWDToolConfig, form: str) -> fdem.Survey:
    """构建多频率 LWD 测量（每个频率一个发射源，共享接收器列表）。

    Args:
        cfg: 仪器配置。
        form: 离散形式，'1d'/'e'/'b'/'h'/'j'。

    Returns:
        fdem.Survey 对象，数据排列顺序为 频率 -> 收发距 -> 分量。
    """
    receiver_list = build_receivers(cfg, form)
    source_list = []
    for freq in cfg.frequencies:
        if cfg.source_type == "mag_dipole":
            source_list.append(
                _src.MagDipole(receiver_list, frequency=float(freq),
                               location=np.asarray(cfg.source_location, dtype=float),
                               orientation=cfg.orientation, moment=cfg.moment)
            )
        else:
            source_list.append(
                _src.CircularLoop(receiver_list, frequency=float(freq),
                                  location=np.asarray(cfg.source_location, dtype=float),
                                  orientation=cfg.orientation,
                                  radius=cfg.loop_radius)
            )
    return fdem.Survey(source_list)


def data_order(cfg: LWDToolConfig) -> Tuple[int, int, int]:
    """返回数据向量的 (n_freq, n_spacing, n_comp) 结构，供 reshape 使用。"""
    return (len(cfg.frequencies), len(cfg.spacings), len(cfg.components))
