"""灵敏度矩阵、并行频率计算与 LAS 数据导出封装。

E8: 灵敏度矩阵封装 — 计算并返回完整 J 矩阵或 JᵀJ 对角元。
E7: 并行频率计算 — 各频率独立正演，用 concurrent.futures 并行。
E6: LAS 格式导出 — 将反演结果导出为 LAS 2.0 文件。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# E8: 灵敏度矩阵封装
# ============================================================

def compute_sensitivity_matrix(
    forward,
    sigma: np.ndarray,
) -> np.ndarray:
    """计算完整灵敏度矩阵 J (nD × nP)。

    *物理含义*: J[i,j] = ∂d_i/∂σ_j，数据对模型的灵敏度。
    *使用方式*: 传入正演对象和当前模型。
    *应用场景*: 分辨率分析、不确定度评估、实验设计。

    注意：对大规模问题（nD·nP > 1e7），内存可能不足。
    建议改用 Jvec/Jtvec 算子。

    Args:
        forward: LWDFDEMForward 实例。
        sigma: 电导率模型 (S/m)，形状 (nP,)。

    Returns:
        灵敏度矩阵 J，形状 (nD, nP)。
    """
    sigma = np.asarray(sigma, dtype=float)
    nP = sigma.size
    d0 = forward.dpred(sigma)
    nD = d0.size
    J = np.zeros((nD, nP))

    eps = 1e-6  # 相对扰动
    for j in range(nP):
        v = np.zeros(nP)
        v[j] = 1.0
        J[:, j] = forward.jvec(sigma, v)

    return J


def compute_sensitivity_diagonal(
    forward,
    sigma: np.ndarray,
) -> np.ndarray:
    """计算 JᵀJ 的对角元（灵敏度加权向量）。

    *物理含义*: diag(JᵀJ)_j = Σ_i (∂d_i/∂σ_j)²，
    反映第 j 个模型参数的总灵敏度。
    *使用方式*: 传入正演对象和当前模型。
    *应用场景*: 灵敏度加权反演（补偿深部衰减）、实验设计优化。

    内存友好：不需要存储完整 J 矩阵。

    Args:
        forward: LWDFDEMForward 实例。
        sigma: 电导率模型 (S/m)，形状 (nP,)。

    Returns:
        JᵀJ 对角元，形状 (nP,)。
    """
    sigma = np.asarray(sigma, dtype=float)
    nP = sigma.size
    diag_JtJ = np.zeros(nP)

    for j in range(nP):
        v = np.zeros(nP)
        v[j] = 1.0
        Jv = forward.jvec(sigma, v)
        diag_JtJ[j] = np.dot(Jv, Jv)

    return diag_JtJ


# ============================================================
# E7: 并行频率计算
# ============================================================

def _forward_single_frequency(args):
    """单频率正演（供并行调用）。"""
    from .forward import LWDFDEMForward
    from .survey import LWDToolConfig

    freq, geometry, tool_dict, layer_thicknesses, sigma = args

    # 重建单频率 tool
    tool = LWDToolConfig(
        frequencies=(freq,),
        spacings=tool_dict["spacings"],
        components=tool_dict["components"],
        field_kind=tool_dict["field_kind"],
        secondary=tool_dict["secondary"],
        data_type=tool_dict["data_type"],
        orientation=tool_dict["orientation"],
        receiver_geometry=tool_dict["receiver_geometry"],
        moment=tool_dict["moment"],
        source_type=tool_dict["source_type"],
        loop_radius=tool_dict["loop_radius"],
        source_location=tool_dict["source_location"],
    )

    if geometry == "1d":
        fwd = LWDFDEMForward(geometry, tool,
                             layer_thicknesses=layer_thicknesses)
    else:
        raise NotImplementedError("并行正演目前仅支持 1D 几何。")

    return fwd.dpred(sigma)


def forward_parallel_frequencies(
    forward,
    sigma: np.ndarray,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """多频率并行正演（仅 1D 几何）。

    *物理含义*: 各频率的 Maxwell 方程相互独立，可并行求解。
    *使用方式*: 传入已配置的多频率正演对象。
    *应用场景*: 多频率（>8 个）时加速正演计算。

    注意：1D 正演已极快（毫秒级），并行加速有限。
    主要用于大量频率（>50）或批量正演场景。

    Args:
        forward: LWDFDEMForward 实例（须为 1D 几何）。
        sigma: 电导率模型 (S/m)。
        max_workers: 最大并行进程数，默认 CPU 核心数。

    Returns:
        合并后的数据向量，形状同单进程结果。

    Raises:
        ValueError: 非 1D 几何时。
    """
    if forward.geometry != "1d":
        raise ValueError("并行正演目前仅支持 1D 层状几何。")

    frequencies = forward.tool.frequencies
    if len(frequencies) <= 1:
        # 单频率无需并行
        return forward.dpred(sigma)

    # 序列化 tool 配置
    tool_dict = {
        "spacings": forward.tool.spacings,
        "components": forward.tool.components,
        "field_kind": forward.tool.field_kind,
        "secondary": forward.tool.secondary,
        "data_type": forward.tool.data_type,
        "orientation": forward.tool.orientation,
        "receiver_geometry": forward.tool.receiver_geometry,
        "moment": forward.tool.moment,
        "source_type": forward.tool.source_type,
        "loop_radius": forward.tool.loop_radius,
        "source_location": forward.tool.source_location,
    }

    args_list = [
        (freq, forward.geometry, tool_dict,
         forward.layer_thicknesses, sigma)
        for freq in frequencies
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_forward_single_frequency, args): i
                   for i, args in enumerate(args_list)}
        for future in as_completed(futures):
            idx = futures[future]
            results.append((idx, future.result()))

    # 按频率顺序合并
    results.sort(key=lambda x: x[0])
    return np.concatenate([r for _, r in results])


# ============================================================
# E6: LAS 格式导出
# ============================================================

def export_las2(
    depths: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    filepath: str,
    well_name: str = "LWD-EM-Inv",
    null_value: float = -999.25,
) -> str:
    """将深度-数据对导出为 LAS 2.0 格式文件。

    *物理含义*: LAS（Log ASCII Standard）是测井数据交换标准格式。
    *使用方式*: 传入深度数组和 {曲线名: 数据数组} 字典。
    *应用场景*: 与商业测井软件交换数据、存档反演结果。

    Args:
        depths: 深度数组 (m)，形状 (n,)。
        data_dict: 字典，键为曲线名（如 'RHOO', 'COND'），
            值为对应数据数组，形状均为 (n,)。
        filepath: 输出文件路径。
        well_name: 井名，写入 LAS 头。
        null_value: 空值标记，默认 -999.25。

    Returns:
        输出文件路径。
    """
    n = len(depths)
    curve_names = list(data_dict.keys())
    n_curves = len(curve_names)

    lines = []

    # ~VERSION
    lines.append("~VERSION INFORMATION")
    lines.append(" VERS.                  2.0 : LAS version")
    lines.append(" WRAP.                    NO : One line per depth")

    # ~WELL
    lines.append("~WELL INFORMATION")
    lines.append(f" WELL.    {well_name} : Well name")
    lines.append(f" NULL.    {null_value} : Null value")
    lines.append(f" NCON.    {n} : Number of records")
    lines.append(f" NMCN.    {n_curves + 1} : Number of curves (including DEPT)")

    # ~CURVES
    lines.append("~CURVE INFORMATION")
    lines.append(" DEPT.M                    M  : Depth")
    for name in curve_names:
        lines.append(f" {name:8s}.M                    : {name}")

    # ~DATA
    lines.append("~ASCII DATA")

    for i in range(n):
        row = [f"{depths[i]:10.4f}"]
        for name in curve_names:
            val = data_dict[name][i]
            if np.isnan(val) or np.isinf(val):
                row.append(f"{null_value:10.4f}")
            else:
                row.append(f"{val:10.6f}")
        lines.append("  ".join(row))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("LAS 2.0 文件已导出: %s (%d 深度 × %d 曲线)",
                filepath, n, n_curves)
    return filepath
