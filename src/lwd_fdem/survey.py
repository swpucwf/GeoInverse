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
from ..fdemcore.utils import validate_string, validate_float

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


def rotation_matrix(inclination: float, azimuth: float, tool_face: float = 0.0) -> np.ndarray:
    """构建仪器坐标系到地层坐标系的旋转矩阵。

    旋转顺序为 R = Ry(inclination) @ Rz(azimuth) @ Rz(tool_face)，
    其中 Ry 绕 y 轴（井斜），Rz 绕 z 轴（方位/工具面）。

    Args:
        inclination: 井斜角（度），0 表示垂直向下，90 表示水平。
        azimuth: 方位角（度），0 表示正北，90 表示正东。
        tool_face: 工具面角（度），绕井轴旋转，默认 0。

    Returns:
        3x3 正交旋转矩阵，列向量为仪器坐标系各轴在地层坐标系中的方向余弦。
    """
    inc = np.radians(inclination)
    azi = np.radians(azimuth)
    tf = np.radians(tool_face)

    # Ry(inclination): 井斜旋转
    Ry = np.array([
        [np.cos(inc), 0.0, np.sin(inc)],
        [0.0, 1.0, 0.0],
        [-np.sin(inc), 0.0, np.cos(inc)]
    ])

    # Rz(azimuth + tool_face): 方位与工具面旋转合并
    angle = azi + tf
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])

    return Ry @ Rz


@dataclass
class LWDToolConfig:
    """LWD 仪器与观测系统配置。

    本数据类描述了一台随钻电磁波仪器的全部观测系统参数，包括发射源特性、
    接收器布局、记录场量类型与数据格式。它是构建 ``fdem.Survey`` 的唯一输入，
    贯穿正演 / 反演 / 可视化全流程。

    参数按物理意义分为四组：

    **① 发射源参数**
        source_type, moment, loop_radius, source_location, orientation

    **② 接收器布局参数**
        spacings, receiver_geometry

    **③ 观测频率与分量参数**
        frequencies, field_kind, secondary, components, data_type

    下面逐参数说明。

    Args:
        frequencies: 工作频率列表 (Hz)。
            *物理含义*: 仪器发射的电磁波频率，每个频率对应一个独立发射源。
            *使用方式*: 正数元组，如 ``(2e3, 2e4, 2e5)`` 表示 2 kHz / 20 kHz / 200 kHz。
            *应用场景*: 多频测量用于探测不同深度（趋肤深度 δ ∝ 1/√f）；
            高频灵敏度高但探测浅，低频反之。典型 LWD 仪器使用 2–8 个频率。

        spacings: 收发距数组 (m)。
            *物理含义*: 接收线圈中心相对于发射线圈中心的偏移距离，全部取正值。
            *使用方式*: 1D numpy 数组，如 ``np.array([0.5, 1.0, 2.0])``。
            实际坐标由 ``receiver_geometry`` 决定偏移方向。
            *应用场景*: 多收发距提供径向不同探测深度的信息；短收发距反映
            井眼附近，长收发距反映原状地层。典型 LWD 有 2–4 个收发距。

        components: 记录的数据分量。
            *物理含义*: 从复数场值中提取哪些可观测物理量。
            *使用方式*: 元组，取值范围 ``'real'``（实部）/ ``'imag'``（虚部）/
            ``'amp'``（幅值）/ ``'phase'``（相位，弧度）。
            *应用场景*: 多数仪器记录实部+虚部；部分仪器记录幅值+相位。
            反演时 components 须与实际观测数据一致。

        field_kind: 记录的场量类型。
            *物理含义*: 接收器测量的电磁场分量。
            *使用方式*: 单选字符串，``'h'``（磁场 H, A/m）/ ``'b'``（磁通密度 B, T）/
            ``'e'``（电场 E, V/m）/ ``'j'``（电流密度 J, A/m²）。
            *应用场景*: 感应式线圈接收器测 H（或 B）；电极接收器测 E。
            1D 层状介质中 H 与 B 等价（B = μ₀H），选择 'h' 或 'b' 结果一致。

        secondary: 是否仅记录二次场。
            *物理含义*: True 时接收器只记录散射/异常场（去掉背景一次场）；
            False 时记录总场。
            *使用方式*: 布尔值。仅 ``field_kind='h'`` 或 ``'b'`` 时支持
            （电场接收器只能测总场）。
            *应用场景*: 二次场接收器对异常体更灵敏（一次场被减掉），
            适用于金属矿勘探；总场接收器适用于地层电阻率测量。

        data_type: 数据归一化方式。
            *物理含义*: ``'field'`` 为绝对场值；``'ppm'`` 为相对于一次场
            的百万分比归一化（parts per million）。
            *使用方式*: 目前仅在反演数据预处理中生效；正演输出始终为绝对场值。
            *应用场景*: PPM 数据在航空电磁中常见，可消除一次场几何因子影响。

        orientation: 偶极子 / 线圈轴向。
            *物理含义*: 发射磁矩方向，``'x'`` / ``'y'`` 为横向，``'z'`` 为沿井轴。
            *使用方式*: 单选字符串。
            *应用场景*: 垂直井中通常用 ``'z'``（轴向偶极子），产生以井轴为
            对称轴的场；水平井或斜井可能需要 ``'x'`` 或 ``'y'``。

        receiver_geometry: 接收器相对发射器的偏移方向。
            *物理含义*: ``'axial'`` 表示接收器沿井轴 (z) 偏移；
            ``'radial'`` 表示沿径向 (x) 偏移。
            *使用方式*: 单选字符串。**1D 层状几何必须使用 ``'radial'``**
            （Hankel 变换以径向距离为自变量）。
            *应用场景*: 垂直井径向仪器 → ``'radial'``；
            2D/3D 轴向仪器 → ``'axial'``。

        moment: 磁偶极矩 (A·m²)。
            *物理含义*: 发射线圈的电流 × 面积 × 匝数 (I·A·N)。
            *使用方式*: 正浮点数，默认 1.0。
            *应用场景*: 决定发射强度；实际仪器典型值 1–100 A·m²。
            正演结果与 moment 线性成正比。

        source_type: 发射源类型。
            *物理含义*: ``'mag_dipole'`` 为理想磁偶极子（点源）；
            ``'circular_loop'`` 为有限尺寸圆环线源。
            *使用方式*: 单选字符串。
            *应用场景*: 偶极子适用于收发距远大于线圈尺寸的情况；
            圆环线源在收发距与线圈尺寸可比时更精确（如井眼校正场景）。

        loop_radius: 圆环线源半径 (m)。
            *物理含义*: 仅 ``source_type='circular_loop'`` 时生效。
            *使用方式*: 正浮点数，默认 0.05 m（5 cm）。
            *应用场景*: 有限尺寸线圈建模；偶极子源时忽略此参数。

        source_location: 发射器位置 (m)。
            *物理含义*: 发射偶极子 / 线圈中心的全局坐标 (x, y, z)。
            *使用方式*: 三元组，默认 ``(0, 0, 0)``。
            *应用场景*: 多发射器或需要全局定位时使用；单源时一般取原点。

    Attributes:
        frequencies: 工作频率列表 (Hz)。
        spacings: 收发距数组 (m)。
        components: 记录分量。
        field_kind: 场量类型。
        secondary: 是否二次场。
        data_type: 数据归一化方式。
        orientation: 偶极子轴向。
        receiver_geometry: 接收器偏移方向。
        moment: 磁偶极矩 (A·m²)。
        source_type: 发射源类型。
        loop_radius: 线圈半径 (m)。
        source_location: 发射器位置 (m)。
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
    inclination: float = 0.0
    azimuth: float = 0.0
    tool_face: float = 0.0

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
        validate_float("inclination", self.inclination, min_val=0.0, max_val=180.0)
        validate_float("azimuth", self.azimuth, min_val=0.0, max_val=360.0)
        validate_float("tool_face", self.tool_face, min_val=0.0, max_val=360.0)
        if self.field_kind == "e" and self.secondary:
            raise ValueError("电场接收器仅支持总场（secondary=False）。")
        if self.receiver_geometry == "radial" and self.orientation == "z":
            # 径向偏移 + z 取向偶极子在 1D/轴对称中均合法，无需拦截
            pass

    @property
    def orientation_vector(self) -> np.ndarray:
        """工具轴方向向量（单位向量，地层坐标系）。

        姿态角全零时退化为 orientation 字符串对应的单位向量；
        否则从旋转矩阵提取工具 z 轴方向。
        """
        if (self.inclination == 0.0 and self.azimuth == 0.0
                and self.tool_face == 0.0):
            mapping = {"x": np.r_[1.0, 0.0, 0.0],
                       "y": np.r_[0.0, 1.0, 0.0],
                       "z": np.r_[0.0, 0.0, 1.0]}
            return mapping[self.orientation]
        R = self.rotation_matrix
        return R[:, 2]  # 工具 z 轴在地层坐标系中的方向

    @property
    def rotation_matrix(self) -> np.ndarray:
        """完整 3x3 旋转矩阵 R = Ry(inc) @ Rz(azi + tf)。"""
        return rotation_matrix(self.inclination, self.azimuth, self.tool_face)


def _receiver_location(cfg: LWDToolConfig, spacing: float) -> np.ndarray:
    """按 receiver_geometry 计算单个接收器坐标（源在原点的相对位置）。"""
    if cfg.receiver_geometry == "axial":
        return np.array([0.0, 0.0, spacing])
    return np.array([spacing, 0.0, 0.0])


def _receiver_location_tilted(cfg: LWDToolConfig, spacing: float) -> np.ndarray:
    """计算倾斜仪器的接收器位置（考虑工具姿态角）。

    姿态角全零时退化为 _receiver_location 的行为；
    姿态角非零时沿倾斜后的工具轴或垂直方向偏移。

    Args:
        cfg: 仪器配置（含 inclination/azimuth/tool_face）。
        spacing: 收发距 (m)。

    Returns:
        接收器全局坐标 (3,)。
    """
    src_loc = np.array(cfg.source_location, dtype=float)
    ori = cfg.orientation_vector
    if cfg.receiver_geometry == "axial":
        return src_loc + spacing * ori
    else:  # radial
        if abs(ori[0]) < 0.9:
            perp = np.cross(ori, np.r_[1.0, 0.0, 0.0])
        else:
            perp = np.cross(ori, np.r_[0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        return src_loc + spacing * perp


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


# ------------------------------------------------------------------ #
# 倾斜仪器独立路径（新接口，不影响 build_survey / build_receivers）
# ------------------------------------------------------------------ #


def build_receivers_tilted(cfg: LWDToolConfig, form: str) -> List:
    """构建倾斜仪器的接收器列表（使用 orientation_vector 与位置偏移）。

    与 build_receivers 完全独立，使用 cfg.orientation_vector 代替字符串
    orientation，并用 _receiver_location_tilted 计算接收器位置。

    Args:
        cfg: 仪器配置（含 inclination/azimuth/tool_face）。
        form: 离散形式，'e'/'b'/'h'/'j'。

    Returns:
        接收器对象列表。

    Raises:
        ValueError: 接收器与离散形式不兼容时。
    """
    key = (cfg.field_kind, cfg.secondary)
    if key not in _SUPPORTED[form]:
        raise ValueError(
            f"接收器 (field={cfg.field_kind!r}, secondary={cfg.secondary!r}) "
            f"与离散形式 {form!r} 不兼容。"
        )
    rx_cls = _RX_CLASSES[key]
    receiver_list = []
    for spacing in cfg.spacings:
        loc = _receiver_location_tilted(cfg, float(spacing))
        for comp in cfg.components:
            receiver_list.append(
                rx_cls(loc, orientation=cfg.orientation_vector,
                       component=comp, data_type=cfg.data_type)
            )
    return receiver_list


def build_survey_tilted(cfg: LWDToolConfig, form: str) -> fdem.Survey:
    """构建倾斜仪器的多频率 Survey（独立于 build_survey）。

    使用 cfg.orientation_vector 作为源和接收器的方向向量，
    fdemcore 的 validate_direction 原生支持 ndarray 输入。

    Args:
        cfg: 仪器配置（含 inclination/azimuth/tool_face）。
        form: 离散形式，'e'/'b'/'h'/'j'。

    Returns:
        fdem.Survey 对象，数据排列顺序为 频率 -> 收发距 -> 分量。
    """
    receiver_list = build_receivers_tilted(cfg, form)
    source_list = []
    ori_vec = cfg.orientation_vector
    src_loc = np.asarray(cfg.source_location, dtype=float)
    for freq in cfg.frequencies:
        if cfg.source_type == "mag_dipole":
            source_list.append(
                _src.MagDipole(receiver_list, frequency=float(freq),
                               location=src_loc,
                               orientation=ori_vec,
                               moment=cfg.moment)
            )
        else:
            source_list.append(
                _src.CircularLoop(receiver_list, frequency=float(freq),
                                  location=src_loc,
                                  orientation=ori_vec,
                                  radius=cfg.loop_radius)
            )
    return fdem.Survey(source_list)
