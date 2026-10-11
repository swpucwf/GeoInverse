"""滑动窗口反演封装（深度扫描）。

LWD 随钻实时处理核心功能：沿井深逐窗口反演，生成连续电阻率剖面。

工作流程：
1. 输入：深度-数据序列（多深度点 × 多频率 × 多收发距）
2. 滑动窗口：在每个深度点取前后若干数据点
3. 逐点反演：对窗口内数据执行 1D 反演
4. 输出：深度-电阻率剖面

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def sliding_window_inversion_1d(
    depths: np.ndarray,
    data_at_depths: np.ndarray,
    forward_builder: Callable,
    inversion_runner: Callable,
    window_size: int = 5,
    step: int = 1,
    initial_model: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """1D 滑动窗口反演（深度扫描）。

    *物理含义*: 在每个深度点，取前后 window_size 个数据点构成局部数据集，
    执行 1D 反演得到该深度处的地层电阻率。沿井深滑动，生成连续剖面。
    *使用方式*: 传入深度数组、各深度处的数据、正演构建函数、反演执行函数。
    *应用场景*: LWD 实时处理、深度域电阻率成像。

    Args:
        depths: 深度数组 (m)，形状 (n_depths,)。
        data_at_depths: 各深度处的数据，形状 (n_depths, nD)。
            nD 为单个深度点的数据向量长度（多频率×多收发距×多分量）。
        forward_builder: 正演对象构建函数，签名 (data_window) → forward。
            接收窗口内数据的平均值，返回配置好的 LWDFDEMForward 实例。
        inversion_runner: 反演执行函数，签名 (forward, data_target) → sigma。
            接收正演对象和目标数据，返回反演电阻率模型。
        window_size: 窗口大小（数据点数），默认 5。
        step: 滑动步长（深度点数），默认 1。
        initial_model: 初始模型，形状 (nP,)。None 时由 inversion_runner 处理。
        verbose: 是否打印进度。

    Returns:
        字典，含：
        - 'depths': 反演深度数组（与输入相同或按 step 抽取）
        - 'models': 各深度反演模型，形状 (n_inv, nP)
        - 'misfits': 各深度拟合差 φ_d
        - 'converged': 各深度是否收敛

    Example:
        >>> def build_fwd(data_avg):
        ...     # 根据平均数据配置正演对象
        ...     return LWDFDEMForward(...)
        >>> def run_inv(forward, data_target):
        ...     # 执行反演
        ...     inv = LWDFDEMInversion(forward, data_obj, ...)
        ...     sigma, hist = inv.run()
        ...     return sigma
        >>> result = sliding_window_inversion_1d(
        ...     depths, data_at_depths, build_fwd, run_inv
        ... )
    """
    n_depths = len(depths)
    if data_at_depths.shape[0] != n_depths:
        raise ValueError(
            f"data_at_depths 第一维 ({data_at_depths.shape[0]}) "
            f"须与 depths 长度 ({n_depths}) 一致。"
        )

    half_win = window_size // 2
    inv_depths = []
    inv_models = []
    inv_misfits = []
    inv_converged = []

    # 滑动窗口
    for i in range(0, n_depths, step):
        # 窗口范围
        i_start = max(0, i - half_win)
        i_end = min(n_depths, i + half_win + 1)

        # 窗口内数据
        window_data = data_at_depths[i_start:i_end]

        # 构建正演（用窗口平均数据）
        data_avg = np.mean(window_data, axis=0)
        try:
            forward = forward_builder(data_avg)
        except Exception as e:
            logger.warning("深度 %.1f m: 正演构建失败 - %s", depths[i], e)
            continue

        # 执行反演（目标为当前深度数据）
        data_target = data_at_depths[i]
        try:
            sigma = inversion_runner(forward, data_target)
            inv_depths.append(depths[i])
            inv_models.append(sigma)
            # 拟合差需要重新计算
            d_pred = forward.dpred(sigma)
            misfit = np.linalg.norm(d_pred - data_target) ** 2
            inv_misfits.append(misfit)
            inv_converged.append(True)

            if verbose:
                logger.info("深度 %.1f m: φ_d = %.4f", depths[i], misfit)

        except Exception as e:
            logger.warning("深度 %.1f m: 反演失败 - %s", depths[i], e)
            inv_depths.append(depths[i])
            inv_models.append(np.full_like(inv_models[-1] if inv_models else
                                           np.zeros(1), np.nan))
            inv_misfits.append(np.nan)
            inv_converged.append(False)

    return {
        "depths": np.array(inv_depths),
        "models": np.array(inv_models) if inv_models else np.array([]),
        "misfits": np.array(inv_misfits),
        "converged": np.array(inv_converged),
    }


def simple_window_builder(
    tool_config,
    layer_thicknesses: np.ndarray,
    n_layers: int,
    sigma_bounds: Tuple[float, float] = (1e-3, 10.0),
) -> Callable:
    """生成简单窗口正演构建器（用于滑动窗口反演）。

    *物理含义*: 根据平均数据估计初始电导率，构建 1D 正演对象。
    *使用方式*: 传入仪器配置、层厚度、层数。
    *应用场景*: 配合 sliding_window_inversion_1d 使用。

    Args:
        tool_config: LWDToolConfig 实例。
        layer_thicknesses: 层厚度数组 (m)。
        n_layers: 反演层数。
        sigma_bounds: 电导率范围 (S/m)。

    Returns:
        正演构建函数 (data_avg) → LWDFDEMForward。
    """
    from .apparent import apparent_resistivity_from_hz
    from .forward import LWDFDEMForward

    def builder(data_avg: np.ndarray) -> LWDFDEMForward:
        # 用第一个频率、第一个收发距的实部数据估计视电阻率
        n_freq = len(tool_config.frequencies)
        n_spacing = len(tool_config.spacings)
        n_comp = len(tool_config.components)

        if data_avg.size >= n_freq * n_spacing * n_comp:
            # 取第一个频率第一个收发距的实部
            val = data_avg[0]  # 假设第一个是实部
            freq = tool_config.frequencies[0]
            spacing = tool_config.spacings[0]
            sigma_est = apparent_resistivity_from_hz(
                freq, val + 0j, spacing, sigma_bounds=sigma_bounds
            )
        else:
            sigma_est = 0.1  # 默认

        # 构建均匀初始模型
        sigma_init = np.full(n_layers, sigma_est)

        fwd = LWDFDEMForward(
            "1d", tool_config,
            layer_thicknesses=layer_thicknesses,
        )
        return fwd

    return builder


def simple_inversion_runner(
    max_iter: int = 20,
    beta_cooling: float = 5.0,
    chifact: float = 1.0,
) -> Callable:
    """生成简单反演执行器（用于滑动窗口反演）。

    *物理含义*: 配置 Occam 反演参数并执行。
    *使用方式*: 传入反演控制参数。
    *应用场景*: 配合 sliding_window_inversion_1d 使用。

    Args:
        max_iter: 最大迭代次数。
        beta_cooling: β 冷却因子。
        chifact: 目标拟合差系数。

    Returns:
        反演执行函数 (forward, data_target) → sigma。
    """
    from .dataio import make_data_object
    from .inversion import LWDFDEMInversion

    def runner(forward, data_target: np.ndarray) -> np.ndarray:
        # 构建数据对象（假设 5% 相对误差）
        data_obj = make_data_object(
            forward.survey, data_target,
            relative_error=0.05,
        )

        # 初始模型（均匀 0.1 S/m）
        n_params = forward.n_params
        m0 = np.log(0.1 * np.ones(n_params))

        # 反演
        inv = LWDFDEMInversion(
            forward, data_obj,
            m0=m0,
            max_iter=max_iter,
            beta_cooling=beta_cooling,
            chifact=chifact,
        )
        sigma, history = inv.run()
        return sigma

    return runner
