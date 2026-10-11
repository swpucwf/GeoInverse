# -*- coding: utf-8 -*-
"""论文复现共享的物理模型定义与工具（纯 numpy，不依赖 lwd_fdem / simpeg）。

本模块被 paper1/paper2 的两个版本（lwd_fdem 与 SimPEG）共同导入，
以确保两版本使用**完全一致**的地层参数、网格规格与观测系统定义，
从而使 compare_results.py 的数值对比是严格的同物理量比较。

包含内容：
  * 论文1（Multi-task U-net look-ahead LWD）的 5 层各向异性 VTI 模型；
  * 论文2（Knowledge-guided DL look-ahead LWD）的 2 层敏感性模型；
  * 网格规格（quick / full 两档）；
  * 张量数据整形、衰减比 Att、UHRA/UHRP 等后处理工具。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ------------------------------------------------------------------ #
# 网格规格
# ------------------------------------------------------------------ #


@dataclass
class MeshSpec:
    """3D 直角张量网格规格（core + 等比 padding，origin=CCC）。

    Attributes:
        cell_size: 核心区单元尺寸 (m)。
        core: 核心区尺寸 (dx, dy, dz) (m)。
        n_pad: 每侧 padding 层数。
        pad_factor: padding 单元等比放大因子。
    """

    cell_size: float
    core: Tuple[float, float, float]
    n_pad: int
    pad_factor: float = 1.3

    def nxyz(self) -> Tuple[int, int, int]:
        """核心区各方向单元数 (nx, ny, nz)。"""
        return (
            max(int(np.ceil(self.core[0] / self.cell_size)), 1),
            max(int(np.ceil(self.core[1] / self.cell_size)), 1),
            max(int(np.ceil(self.core[2] / self.cell_size)), 1),
        )

    def h_arrays(self) -> Tuple[list, list, list]:
        """返回 discretize.TensorMesh 所需的 (hx, hy, hz) 单元宽度列表。

        与 lwd_fdem.build_3d_tensor_mesh 的构造严格一致，
        保证两版本网格逐单元相同。
        """
        nx, ny, nz = self.nxyz()
        cs, npad, pf = self.cell_size, self.n_pad, self.pad_factor
        hx = [(cs, npad, -pf), (cs, nx), (cs, npad, pf)]
        hy = [(cs, npad, -pf), (cs, ny), (cs, npad, pf)]
        hz = [(cs, npad, -pf), (cs, nz), (cs, npad, pf)]
        return hx, hy, hz


# quick 档：粗网格、少频点，用于秒级—分钟级演示与对比验证
MESH_QUICK = MeshSpec(cell_size=6.0, core=(42.0, 42.0, 54.0), n_pad=2)
# full 档：接近论文精度的细网格（计算量大，需数十分钟至数小时）
MESH_FULL = MeshSpec(cell_size=2.0, core=(80.0, 80.0, 96.0), n_pad=5)

# 论文2 专用网格：接收器最远 R3=30 m，需更大的 z 向覆盖
MESH_QUICK_P2 = MeshSpec(cell_size=8.0, core=(48.0, 48.0, 72.0), n_pad=2)
MESH_FULL_P2 = MeshSpec(cell_size=2.5, core=(90.0, 90.0, 120.0), n_pad=5)


# ------------------------------------------------------------------ #
# 论文1：5 层各向异性 VTI 模型
# ------------------------------------------------------------------ #

# 各层水平/垂直电导率 (S/m)：层1(背景)、层2、层3(目标)、层4、层5(基底)
PAPER1_SIGMA_H = np.array([0.1, 0.5, 0.01, 0.3, 0.1])
PAPER1_SIGMA_V = np.array([0.05, 0.25, 0.005, 0.15, 0.05])
# 层界面深度 (m)，相对仪器（发射器）位置，自上而下
PAPER1_BOUNDARIES = np.array([5.0, 10.0, 15.0, 20.0])

# 仪器：单发射 + 双接收，收发距 TR1=10 m, TR2=14 m
PAPER1_SPACINGS = np.array([10.0, 14.0])
# 工作频率 (Hz)：10, 20, 30, 50 kHz
PAPER1_FREQS_FULL = (10e3, 20e3, 30e3, 50e3)
PAPER1_FREQS_QUICK = (10e3, 30e3)


def paper1_layer_index(z_centers: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """按单元中心 z 坐标判定所属层号（0..4）。

    层界面随 ``shift`` 沿 -z 平移，用于模拟仪器相对地层的前视推进
    （仪器固定于原点，界面向仪器靠近等价于仪器向下钻进）。

    Args:
        z_centers: 单元中心 z 坐标数组 (m)。
        shift: 界面平移量 (m)，>=0 表示界面向 -z（仪器后方）移动。

    Returns:
        层索引数组，范围 [0, 4]。
    """
    bnds = PAPER1_BOUNDARIES - shift
    idx = np.searchsorted(bnds, z_centers, side="right")
    return np.clip(idx, 0, len(PAPER1_SIGMA_H) - 1)


def paper1_sigma_tensor(z_centers: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """构建 5 层 VTI 各向异性电导率张量 (n_cells, 3) = [σxx, σyy, σzz]。

    VTI（垂直横向各向同性）：水平方向 σxx=σyy=σh，垂直方向 σzz=σv。

    Args:
        z_centers: 单元中心 z 坐标数组 (m)。
        shift: 界面平移量 (m)。

    Returns:
        shape (n_cells, 3) 的电导率数组 (S/m)。
    """
    layer = paper1_layer_index(z_centers, shift)
    sig = np.zeros((z_centers.size, 3))
    sig[:, 0] = PAPER1_SIGMA_H[layer]
    sig[:, 1] = PAPER1_SIGMA_H[layer]
    sig[:, 2] = PAPER1_SIGMA_V[layer]
    return sig


# ------------------------------------------------------------------ #
# 论文2：2 层敏感性模型
# ------------------------------------------------------------------ #

# 上层固定电阻率 (Ohm·m)，下层扫描
PAPER2_RHO1 = 100.0
# 收发距 R1/R2/R3 (m)
PAPER2_SPACINGS_FULL = np.array([10.0, 20.0, 30.0])
PAPER2_SPACINGS_QUICK = np.array([10.0, 20.0, 30.0])
# 距离地层界面深度 DTB (m)
PAPER2_DTB_FULL = (5.0, 15.0, 30.0)
PAPER2_DTB_QUICK = (15.0,)
# 下层电阻率扫描序列 (Ohm·m)
PAPER2_RHO2_FULL = np.array([1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
                             1000.0, 2000.0, 5000.0, 10000.0])
PAPER2_RHO2_QUICK = np.array([1.0, 10.0, 100.0, 1000.0])
# 代表频率 (Hz)：完整为 Table 1 各子阵并集；quick 取代表值
PAPER2_FREQS_FULL = (2e3, 6e3, 12e3, 24e3, 48e3, 96e3)
PAPER2_FREQS_QUICK = (6e3, 24e3)


def paper2_sigma(z_centers: np.ndarray, dtb: float, rho2: float) -> np.ndarray:
    """构建 2 层各向同性电导率 (n_cells,)。

    上层（z < DTB）电阻率固定 RHO1；下层（z >= DTB）电阻率为 rho2。

    Args:
        z_centers: 单元中心 z 坐标数组 (m)。
        dtb: 界面深度 (m)，相对仪器位置。
        rho2: 下层电阻率 (Ohm·m)。

    Returns:
        shape (n_cells,) 的电导率数组 (S/m)。
    """
    sigma = np.full(z_centers.size, 1.0 / PAPER2_RHO1)
    sigma[z_centers >= dtb] = 1.0 / rho2
    return sigma


# ------------------------------------------------------------------ #
# 张量数据后处理工具
# ------------------------------------------------------------------ #

# 张量数据排列：频率 -> 源取向(x,y,z) -> 收发距 -> 接收取向(x,y,z) -> 分量(real,imag)
TENSOR_SHAPE = "n_freq, n_src(3), n_spacing, n_rx(3), n_comp(2)"


def to_complex_tensor(d: np.ndarray, n_freq: int, n_spacing: int) -> np.ndarray:
    """将实虚部交替的数据向量整形为复数张量。

    Args:
        d: 一维数据向量，排列顺序见 TENSOR_SHAPE。
        n_freq: 频率个数。
        n_spacing: 收发距个数。

    Returns:
        复数数组，shape (n_freq, 3, n_spacing, 3)，末两维为 (源取向, 接收取向)。
    """
    arr = np.asarray(d, dtype=float).reshape(n_freq, 3, n_spacing, 3, 2)
    return arr[..., 0] + 1j * arr[..., 1]


def attenuation_ratio(h_complex: np.ndarray) -> np.ndarray:
    """计算衰减比 Att_ij = -20*log10(|H_R1_ij| / |H_R2_ij|)。

    Args:
        h_complex: 复数张量 (n_freq, 3, n_spacing, 3)，n_spacing>=2，
            索引 0 为近接收器 R1，索引 1 为远接收器 R2。

    Returns:
        Att 数组，shape (n_freq, 3, 3)，维度为 (频率, 源取向, 接收取向)。
    """
    amp_r1 = np.abs(h_complex[:, :, 0, :])
    amp_r2 = np.abs(h_complex[:, :, 1, :])
    tiny = np.finfo(float).tiny
    return -20.0 * np.log10((amp_r1 + tiny) / (amp_r2 + tiny))


def uhra_uhrp(h_complex: np.ndarray, spacing_idx: int = 0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """计算 UHRA / UHRP 视电阻率相关量。

    UHRA = 20*log10| 2*Vzz / (Vxx + Vyy) |；
    UHRP = angle[ 2*Vzz / (Vxx + Vyy) ]（弧度）。
    其中 Vxx/Vyy/Vzz 取张量对角分量（源取向=接收取向）。

    Args:
        h_complex: 复数张量 (n_freq, 3, n_spacing, 3)。
        spacing_idx: 使用第几个收发距的接收器。

    Returns:
        (UHRA, UHRP)，均为 shape (n_freq,) 数组。
    """
    hxx = h_complex[:, 0, spacing_idx, 0]
    hyy = h_complex[:, 1, spacing_idx, 1]
    hzz = h_complex[:, 2, spacing_idx, 2]
    tiny = np.finfo(float).tiny
    ratio = 2.0 * hzz / (hxx + hyy + tiny)
    uhra = 20.0 * np.log10(np.abs(ratio) + tiny)
    uhrp = np.angle(ratio)
    return uhra, uhrp


def diagonal_components(h_complex: np.ndarray, spacing_idx: int = 0) -> Dict[str, np.ndarray]:
    """提取主对角分量 Hxx/Hyy/Hzz（复数），shape (n_freq,)。"""
    return {
        "Hxx": h_complex[:, 0, spacing_idx, 0],
        "Hyy": h_complex[:, 1, spacing_idx, 1],
        "Hzz": h_complex[:, 2, spacing_idx, 2],
    }


def normalized_sensitivity(values: np.ndarray) -> float:
    """归一化敏感度指标：沿扫描序列的极差除以均方根幅值。

    用于定量比较各分量（实部/虚部）对下层电阻率变化的敏感程度：
        S = (max - min) / rms(|values|)
    值越大表示该分量对电阻率变化越敏感。

    Args:
        values: 一维数组，随电阻率扫描的某分量取值（实部或虚部）。

    Returns:
        归一化敏感度标量；幅值退化时返回 0。
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    rng = float(np.nanmax(v) - np.nanmin(v))
    rms = float(np.sqrt(np.nanmean(v ** 2)))
    return rng / rms if rms > 0 else 0.0


def paper1_meta(quick: bool) -> Dict:
    """返回论文1的运行配置元信息。"""
    if quick:
        return {
            "mesh": MESH_QUICK,
            "freqs": PAPER1_FREQS_QUICK,
            "shifts": (0.0, 5.0),
        }
    return {
        "mesh": MESH_FULL,
        "freqs": PAPER1_FREQS_FULL,
        "shifts": (0.0, 3.0, 6.0, 9.0, 12.0),
    }


def paper2_meta(quick: bool) -> Dict:
    """返回论文2的运行配置元信息。"""
    if quick:
        return {
            "mesh": MESH_QUICK_P2,
            "freqs": PAPER2_FREQS_QUICK,
            "dtbs": PAPER2_DTB_QUICK,
            "rho2_scan": PAPER2_RHO2_QUICK,
            "spacings": PAPER2_SPACINGS_QUICK,
        }
    return {
        "mesh": MESH_FULL_P2,
        "freqs": PAPER2_FREQS_FULL,
        "dtbs": PAPER2_DTB_FULL,
        "rho2_scan": PAPER2_RHO2_FULL,
        "spacings": PAPER2_SPACINGS_FULL,
    }
