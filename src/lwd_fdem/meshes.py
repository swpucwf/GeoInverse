"""LWD 频域电磁正反演的网格生成工具。

提供 1D 层状、2D 轴对称（圆柱）与 3D 直角张量网格的构建函数，
以及基于趋肤深度的网格分辨率自动校核（对应决策清单 A2/A3/A4）。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from discretize import CylindricalMesh, TensorMesh

# 真空电磁参数
_MU0 = 4.0e-7 * np.pi          # 真空磁导率 (H/m)
_EPS0 = 8.8541878128e-12       # 真空介电常数 (F/m)


def skin_depth(frequency: float, conductivity: float,
               relative_permittivity: float = 1.0) -> float:
    """计算介质中的趋肤深度（全频域 Maxwell，含位移电流）。

    Args:
        frequency: 频率 (Hz)。
        conductivity: 电导率 (S/m)。
        relative_permittivity: 相对介电常数。

    Returns:
        趋肤深度 (m)，即衰减常数的倒数； conductivity<=0 时返回 inf。
    """
    if conductivity <= 0.0 or frequency <= 0.0:
        return np.inf
    omega = 2.0 * np.pi * frequency
    mu = _MU0
    epsilon = _EPS0 * relative_permittivity
    # 复波数 k = sqrt(w^2*mu*eps - i*w*mu*sigma)，衰减系数 alpha = |Im(k)|
    k = np.sqrt(omega ** 2 * mu * epsilon - 1j * omega * mu * conductivity)
    alpha = abs(k.imag)
    if alpha < 1e-30:
        return np.inf
    return 1.0 / alpha


def recommend_cell_size(frequencies: Sequence[float], conductivities: Sequence[float],
                        relative_permittivity: float = 1.0,
                        divisor: float = 3.0) -> float:
    """按最不利（最深）趋肤深度推荐最大网格单元尺寸。

    判据（决策清单 A3）：单元尺寸 <= delta_min / divisor。

    Args:
        frequencies: 工作频率列表 (Hz)。
        conductivities: 介质电导率范围 (S/m)。
        relative_permittivity: 相对介电常数。
        divisor: 趋肤深度离散份数，默认 3。

    Returns:
        推荐的最大单元尺寸 (m)。
    """
    deltas = [
        skin_depth(f, s, relative_permittivity)
        for f in frequencies for s in conductivities
    ]
    delta_min = min(d for d in deltas if np.isfinite(d)) if deltas else np.inf
    if not np.isfinite(delta_min):
        raise ValueError("无法从给定的频率/电导率计算趋肤深度，请检查输入。")
    return delta_min / divisor


def _add_padding(h_core: np.ndarray, n_pad: int, pad_factor: float) -> np.ndarray:
    """为核心区单元宽度数组两侧添加等比递增 padding。

    Args:
        h_core: 核心区单元宽度 1D 数组 (m)。
        n_pad: 每侧 padding 层数。
        pad_factor: padding 等比放大因子 (>1)。

    Returns:
        含 padding 的完整单元宽度 1D 数组。
    """
    dh = h_core[0]
    pad_left = dh * pad_factor ** np.arange(n_pad, 0, -1)
    pad_right = dh * pad_factor ** np.arange(1, n_pad + 1)
    return np.concatenate([pad_left, h_core, pad_right])


def build_1d_mesh(layer_thicknesses: Sequence[float]) -> TensorMesh:
    """为 1D 层状模型构建用于正则化的张量网格。

    层数为 len(layer_thicknesses)+1（最后一层为无限厚半空间，
    以与厚度层等宽的单元近似）。

    Args:
        layer_thicknesses: 各层厚度 (m)，自上而下排列。

    Returns:
        1D TensorMesh，单元数 = len(layer_thicknesses)+1。
    """
    thicknesses = np.asarray(layer_thicknesses, dtype=float)
    if thicknesses.ndim != 1 or np.any(thicknesses <= 0):
        raise ValueError("layer_thicknesses 必须为正的 1D 数组。")
    h = np.r_[thicknesses, thicknesses[-1]]
    return TensorMesh([h], "0")


def build_2d_cyl_mesh(
    r_core: float,
    z_core: float,
    cell_size: Optional[float] = None,
    frequencies: Optional[Sequence[float]] = None,
    conductivities: Optional[Sequence[float]] = None,
    n_pad: int = 8,
    pad_factor: float = 1.3,
) -> CylindricalMesh:
    """构建 2D 轴对称（圆柱）网格：core + 等比外扩 padding。

    井轴位于 r=0 旋转轴上（源必须在轴上，接收器可径向/轴向偏移）。
    未显式给定 cell_size 时，按 frequencies/conductivities 自动校核（A3）。

    Args:
        r_core: 径向核心区域长度 (m)，覆盖井周/侵入带。
        z_core: 轴向核心区域长度 (m)，覆盖工具及附近地层。
        cell_size: 核心单元尺寸 (m)；为 None 时自动计算。
        frequencies: 自动计算单元尺寸时使用的工作频率 (Hz)。
        conductivities: 自动计算单元尺寸时使用的电导率范围 (S/m)。
        n_pad: 每个方向 padding 层数（决策清单 A4）。
        pad_factor: padding 单元等比放大系数。

    Returns:
        CylindricalMesh（phi 方向单段，m=0 模态轴对称）。
    """
    if cell_size is None:
        if not frequencies or not conductivities:
            raise ValueError("cell_size 为 None 时必须提供 frequencies 与 conductivities。")
        cell_size = recommend_cell_size(frequencies, conductivities)
    if r_core <= 0 or z_core <= 0 or cell_size <= 0:
        raise ValueError("r_core、z_core 与 cell_size 必须均为正。")

    nr = max(int(np.ceil(r_core / cell_size)), 1)
    nz = max(int(np.ceil(z_core / cell_size)), 1)
    h = [(cell_size, nr), (cell_size, n_pad, pad_factor)]
    v = [(cell_size, n_pad, -pad_factor), (cell_size, nz), (cell_size, n_pad, pad_factor)]
    mesh = CylindricalMesh([h, 1, v], x0="00C")
    # 记录核心区单元宽度，供 mesh_quality_report 校核（r/phi/z 三向）
    mesh._lwd_core_widths = (np.full(nr, cell_size), None, np.full(nz, cell_size))
    return mesh


def build_3d_tensor_mesh(
    core_extent: Tuple[float, float, float],
    cell_size: Optional[float] = None,
    frequencies: Optional[Sequence[float]] = None,
    conductivities: Optional[Sequence[float]] = None,
    n_pad: int = 6,
    pad_factor: float = 1.3,
) -> TensorMesh:
    """构建 3D 直角张量网格（core + padding），用于全空间精细模拟。

    Args:
        core_extent: 核心区域尺寸 (dx, dy, dz) (m)。
        cell_size: 核心单元尺寸 (m)；为 None 时自动计算。
        frequencies: 自动计算单元尺寸时使用的工作频率 (Hz)。
        conductivities: 自动计算单元尺寸时使用的电导率范围 (S/m)。
        n_pad: 各方向 padding 层数。
        pad_factor: padding 放大系数。

    Returns:
        TensorMesh，原点在中心 (CCC)。
    """
    if cell_size is None:
        if not frequencies or not conductivities:
            raise ValueError("cell_size 为 None 时必须提供 frequencies 与 conductivities。")
        cell_size = recommend_cell_size(frequencies, conductivities)
    if any(e <= 0 for e in core_extent) or cell_size <= 0:
        raise ValueError("core_extent 与 cell_size 必须均为正。")

    nx = max(int(np.ceil(core_extent[0] / cell_size)), 1)
    ny = max(int(np.ceil(core_extent[1] / cell_size)), 1)
    nz = max(int(np.ceil(core_extent[2] / cell_size)), 1)
    hx = [(cell_size, n_pad, -pad_factor), (cell_size, nx),
          (cell_size, n_pad, pad_factor)]
    hy = [(cell_size, n_pad, -pad_factor), (cell_size, ny),
          (cell_size, n_pad, pad_factor)]
    hz = [(cell_size, n_pad, -pad_factor), (cell_size, nz),
          (cell_size, n_pad, pad_factor)]
    mesh = TensorMesh([hx, hy, hz], origin="CCC")
    # 记录核心区单元宽度，供 mesh_quality_report 校核
    mesh._lwd_core_widths = (np.full(nx, cell_size), np.full(ny, cell_size),
                             np.full(nz, cell_size))
    return mesh


def build_2d_tensor_mesh(
    core_extent_xz: Tuple[float, float],
    cell_size: Optional[float] = None,
    frequencies: Optional[Sequence[float]] = None,
    conductivities: Optional[Sequence[float]] = None,
    relative_permittivity: float = 1.0,
    n_pad: int = 6,
    pad_factor: float = 1.3,
    ny: int = 3,
) -> TensorMesh:
    """构建 2.5D 像素级反演用的薄 TensorMesh。

    x-z 平面为核心反演区域，y 方向仅 ny 个 cell（薄切片）。
    正演为完整 3D FVM 求解，但模型参数仅在 x-z 平面独立，
    y 方向通过映射复制，实现"3D 正演 + 2D 反演"的 2.5D 范式。

    Args:
        core_extent_xz: (dx, dz) 核心区 x 和 z 方向总尺寸 (m)。
        cell_size: 核心区单元尺寸 (m)。若为 None 则自动推荐。
        frequencies: 工作频率列表 (Hz)，用于自动推荐 cell_size。
        conductivities: 电导率范围 (S/m)，用于自动推荐 cell_size。
        relative_permittivity: 相对介电常数，默认 1.0。
        n_pad: 每侧 padding 层数。
        pad_factor: padding 等比放大因子。
        ny: y 方向 cell 数量，默认 3（薄切片）。

    Returns:
        TensorMesh 实例，origin="CCC"（源在网格几何中心）。
    """
    dx, dz = core_extent_xz
    if cell_size is None:
        if frequencies is None or conductivities is None:
            raise ValueError(
                "cell_size 为 None 时必须提供 frequencies 和 conductivities"
            )
        cell_size = recommend_cell_size(
            frequencies, conductivities, relative_permittivity
        )

    nx_core = max(int(np.ceil(dx / cell_size)), 4)
    nz_core = max(int(np.ceil(dz / cell_size)), 4)

    # x 和 z 方向：核心区 + padding
    hx_core = np.ones(nx_core) * cell_size
    hz_core = np.ones(nz_core) * cell_size
    hx = _add_padding(hx_core, n_pad, pad_factor)
    hz = _add_padding(hz_core, n_pad, pad_factor)

    # y 方向：与核心区相同宽度但仅 ny 个 cell（薄切片）
    hy = np.ones(ny) * cell_size

    mesh = TensorMesh([hx, hy, hz], origin="CCC")
    # 记录核心区单元宽度，供 mesh_quality_report 校核
    mesh._lwd_core_widths = (
        np.full(nx_core, cell_size),
        np.full(ny, cell_size),
        np.full(nz_core, cell_size),
    )
    return mesh


def mesh_quality_report(mesh, frequencies: Sequence[float],
                        conductivities: Sequence[float],
                        relative_permittivity: float = 1.0) -> Dict[str, float]:
    """给出网格分辨率与最小趋肤深度的比值报告。

    分辨率判定优先使用本模块构建函数记录的核心区单元宽度
    （`mesh._lwd_core_widths`，padding 区单元本就允许逐步放大，不参与
    校核）；外部构建的网格退化为按全网格最大单元判定。

    Args:
        mesh: discretize 网格对象（build_2d_cyl_mesh/build_3d_tensor_mesh
            返回的网格带核心区记录）。
        frequencies: 工作频率 (Hz)。
        conductivities: 电导率范围 (S/m)。
        relative_permittivity: 相对介电常数。

    Returns:
        含 'max_cell_size'、'min_skin_depth'、'cells_per_skin_depth' 与
        'ok'（1 为单元尺寸 <= delta/3）的字典。
    """
    core_widths = getattr(mesh, "_lwd_core_widths", None)
    if core_widths is not None:
        arrays = [np.asarray(w, dtype=float) for w in core_widths if w is not None]
        cell_max = float(max(np.max(w) for w in arrays if w.size))
    else:
        cell_max = float(max(np.max(np.asarray(h)) for h in mesh.h))
    delta = min(skin_depth(f, s, relative_permittivity)
                for f in frequencies for s in conductivities)
    ratio = delta / cell_max if cell_max > 0 else np.inf
    return {
        "max_cell_size": cell_max,
        "min_skin_depth": float(delta),
        "cells_per_skin_depth": float(ratio),
        "ok": 1.0 if ratio >= 3.0 else 0.0,
    }
