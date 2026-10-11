"""探测深度（DOI）与分辨率分析封装。

基于灵敏度矩阵评估反演结果的可信深度范围：
- 累积灵敏度法：1D 层状模型逐层累积灵敏度
- 阈值法：灵敏度降至峰值一定比例时的深度
- 趋肤深度估计：准静态近似下的理论探测深度

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from typing import Optional, Tuple

import numpy as np

from .advanced import compute_sensitivity_diagonal


def doi_1d_sensitivity(
    forward,
    sigma: np.ndarray,
    threshold_ratio: float = 0.1,
) -> dict:
    """1D 层状模型的探测深度分析（基于灵敏度）。

    *物理含义*: 计算每个地层参数的灵敏度，确定灵敏度降至峰值
    threshold_ratio 倍时的深度作为探测深度上限。
    *使用方式*: 传入 1D 正演对象和当前模型。
    *应用场景*: 评估反演结果的可信深度范围，指导解释。

    Args:
        forward: LWDFDEMForward 实例（须为 1D 几何）。
        sigma: 电导率模型 (S/m)，形状 (n_layers+1,)。
        threshold_ratio: 灵敏度阈值比例（相对峰值），默认 0.1。

    Returns:
        字典，含：
        - 'sensitivity': 各层灵敏度 (nP,)
        - 'cumulative_depths': 各层底界深度 (m)
        - 'doi_depth': 探测深度 (m)
        - 'reliable_mask': 可靠层布尔掩码

    Raises:
        ValueError: 非 1D 几何时。
    """
    if forward.geometry != "1d":
        raise ValueError("DOI 分析目前仅支持 1D 层状几何。")

    # 计算灵敏度对角元
    sens = compute_sensitivity_diagonal(forward, sigma)

    # 计算各层底界深度
    thicknesses = forward.layer_thicknesses
    n_layers = len(thicknesses) + 1  # 含半空间
    cumulative_depths = np.zeros(n_layers)
    cumulative_depths[:n_layers - 1] = np.cumsum(thicknesses)
    cumulative_depths[n_layers - 1] = cumulative_depths[n_layers - 2] + thicknesses[-1]

    # 归一化灵敏度
    sens_max = np.max(sens)
    if sens_max > 0:
        sens_norm = sens / sens_max
    else:
        sens_norm = sens

    # 确定探测深度：灵敏度首次降至阈值以下时的深度
    below_threshold = sens_norm < threshold_ratio
    if np.all(below_threshold):
        doi_depth = 0.0
    elif np.all(~below_threshold):
        doi_depth = cumulative_depths[-1]
    else:
        # 首次降至阈值以下的层
        first_below = np.argmax(below_threshold)
        if first_below == 0:
            doi_depth = 0.0
        else:
            doi_depth = cumulative_depths[first_below - 1]

    # 可靠层掩码
    reliable_mask = sens_norm >= threshold_ratio

    return {
        "sensitivity": sens,
        "sensitivity_normalized": sens_norm,
        "cumulative_depths": cumulative_depths,
        "doi_depth": doi_depth,
        "reliable_mask": reliable_mask,
        "threshold_ratio": threshold_ratio,
    }


def doi_skin_depth_estimate(
    frequencies: np.ndarray,
    conductivities: np.ndarray,
    threshold_ratio: float = 1.0,
) -> float:
    """基于趋肤深度估计理论探测深度。

    *物理含义*: 趋肤深度 δ = 503/√(f·σ) 是电磁波衰减到 1/e 的深度。
    最低频率和最低电导率给出最大探测深度。
    *使用方式*: 传入频率范围和电导率范围。
    *应用场景*: 快速估计仪器探测能力，指导频率选择。

    Args:
        frequencies: 工作频率列表 (Hz)。
        conductivities: 地层电导率范围 (S/m)。
        threshold_ratio: 探测深度 = threshold_ratio × δ_max，默认 1.0。

    Returns:
        理论探测深度 (m)。
    """
    from .meshes import skin_depth

    # 最低频率 + 最低电导率 → 最大趋肤深度
    f_min = np.min(frequencies)
    sigma_min = np.min(conductivities)
    delta_max = skin_depth(f_min, sigma_min)

    return threshold_ratio * delta_max


def doi_resolution_matrix(
    forward,
    sigma: np.ndarray,
    beta: float = 1.0,
    alpha: float = 1e-3,
) -> np.ndarray:
    """计算分辨率矩阵 R = (JᵀJ + αI)⁻¹ JᵀJ。

    *物理含义*: R 的对角元反映各模型参数的分辨能力。
    R[j,j] → 1 表示该参数完全分辨；→ 0 表示不可分辨。
    *使用方式*: 传入正演对象、模型、正则化参数。
    *应用场景*: 定量评估反演分辨率，指导正则化参数选择。

    注意：需要计算完整 J 矩阵，对大规模问题内存可能不足。

    Args:
        forward: LWDFDEMForward 实例。
        sigma: 电导率模型 (S/m)，形状 (nP,)。
        beta: 数据权重（默认 1.0）。
        alpha: 正则化参数（默认 1e-3）。

    Returns:
        分辨率矩阵 R，形状 (nP, nP)。
    """
    from .advanced import compute_sensitivity_matrix

    J = compute_sensitivity_matrix(forward, sigma)
    nP = sigma.size

    # JᵀJ + αI
    JtJ = J.T @ J
    A = JtJ + alpha * np.eye(nP)

    # R = A⁻¹ JᵀJ
    try:
        R = np.linalg.solve(A, JtJ)
    except np.linalg.LinAlgError:
        # 奇异矩阵，用伪逆
        R = np.linalg.lstsq(A, JtJ, rcond=None)[0]

    return R


def doi_summary(
    forward,
    sigma: np.ndarray,
    threshold_ratio: float = 0.1,
) -> str:
    """生成 DOI 分析摘要报告。

    *物理含义*: 汇总灵敏度分布和探测深度信息。
    *使用方式*: 传入正演对象和模型。
    *应用场景*: 快速查看反演结果的可信范围。

    Args:
        forward: LWDFDEMForward 实例。
        sigma: 电导率模型 (S/m)。
        threshold_ratio: 灵敏度阈值比例。

    Returns:
        摘要文本。
    """
    if forward.geometry == "1d":
        result = doi_1d_sensitivity(forward, sigma, threshold_ratio)

        lines = [
            "=" * 50,
            "1D DOI 分析报告",
            "=" * 50,
            f"层数: {len(sigma)}",
            f"灵敏度阈值: {threshold_ratio:.1%}",
            f"探测深度: {result['doi_depth']:.2f} m",
            "",
            "各层灵敏度:",
        ]

        for i, (s, d, r) in enumerate(zip(
            result["sensitivity_normalized"],
            result["cumulative_depths"],
            result["reliable_mask"],
        )):
            status = "可靠" if r else "不可靠"
            layer_info = f"  层 {i+1}: 灵敏度={s:.4f}, 深度≤{d:.1f}m [{status}]"
            lines.append(layer_info)

        lines.append("=" * 50)
        return "\n".join(lines)
    else:
        return "DOI 分析目前仅支持 1D 层状几何。"
