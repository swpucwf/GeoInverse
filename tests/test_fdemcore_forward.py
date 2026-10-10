"""fdemcore FDEM 正演验证测试（移植自 fdem_forward_validation 三套判据）。

判据（对应实施计划阶段 4 / 决策清单 E4）：
  1) 1D 层状正演 vs Ward & Hohmann (1988) Eq 4.56 解析解，相对误差 <= 1e-3；
  2) 2D 轴对称 vs 3D 直角交叉验证 <= 15%；sigma->0 一致性 <= 2%；
  3) Simulation1DLayered 与小尺度 2D FVM 的 check_derivative 梯度校验。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import numpy as np
import pytest
from scipy.constants import mu_0
from discretize.tests import check_derivative

from src.fdemcore import maps
from src.fdemcore.electromagnetics import frequency_domain as fdem
from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig,
    build_2d_cyl_mesh, build_3d_tensor_mesh,
)

TOL_1D = 1e-3        # 1D vs 解析解（0.1%）
TOL_XCHECK = 0.15    # 2D-3D 交叉验证
TOL_SIGMA0 = 0.02    # sigma -> 0 一致性


def hz_quasi_static_dipole(r, freq, sigma, secondary=True):
    """垂直磁偶极子（moment=1 A·m^2）在均匀全空间赤道面上的 Hz。

    Ward & Hohmann (1988) Eq 4.56 的准静态形式（k^2 = -i·w·mu·sigma，
    忽略位移电流）；secondary=True 时减去自由空间一次场
    Hp = -1/(4*pi*r^3)（纯实数，故二次场虚部等于总场虚部）。
    """
    r = np.abs(np.asarray(r, dtype=float))
    w = 2.0 * np.pi * np.asarray(freq, dtype=float)
    k = np.sqrt(-1j * w * mu_0 * sigma)
    front = 1.0 / (2.0 * np.pi * k**2 * r**5)
    back = 9.0 - (
        9.0 + 9j * k * r - 4.0 * (k * r) ** 2 - 1j * (k * r) ** 3
    ) * np.exp(-1j * k * r)
    hz = front * back
    if secondary:
        hz = hz - (-1.0 / (4.0 * np.pi * r**3))
    return hz


# -------------------------------------------------------------------- #
# 1) 1D 层状正演 vs 解析解
# -------------------------------------------------------------------- #
def test_1d_forward_vs_analytic_wholespace():
    """均匀全空间（nC=1、无层厚）Hz 二次场 vs 准静态解析解，误差 <= 0.1%。"""
    sigma = 0.1
    freqs = np.logspace(2, 5, 13)
    cfg = LWDToolConfig(
        frequencies=tuple(float(f) for f in freqs),
        spacings=np.array([1.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
    )
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=np.array([]))
    d = fwd.dpred(np.array([sigma]))
    d_real = fwd.to_matrix(d)[:, 0, 0]
    d_imag = fwd.to_matrix(d)[:, 0, 1]

    hz = hz_quasi_static_dipole(1.0, freqs, sigma, secondary=True)
    err_real = np.max(np.abs(d_real - hz.real) / np.abs(hz.real))
    err_imag = np.max(np.abs(d_imag - hz.imag) / np.abs(hz.imag))
    assert err_real <= TOL_1D, f"实部最大相对误差 {err_real:.3e}"
    assert err_imag <= TOL_1D, f"虚部最大相对误差 {err_imag:.3e}"


def test_1d_forward_vs_analytic_halfspace_layers():
    """层状模型最底层为半空间时，浅层厚度取大值应退化为半空间解析解。"""
    sigma = 0.1
    freqs = np.logspace(2, 4, 7)
    cfg = LWDToolConfig(
        frequencies=tuple(float(f) for f in freqs),
        spacings=np.array([1.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
    )
    # 两层均等于半空间电导率，厚度远大于收发距 -> 数值上等价于全空间
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=np.array([1e4, 1e4]))
    d = fwd.to_matrix(fwd.dpred(sigma * np.ones(3)))
    d_real, d_imag = d[:, 0, 0], d[:, 0, 1]

    hz = hz_quasi_static_dipole(1.0, freqs, sigma, secondary=True)
    assert np.max(np.abs(d_real - hz.real) / np.abs(hz.real)) <= 5e-3
    assert np.max(np.abs(d_imag - hz.imag) / np.abs(hz.imag)) <= 5e-3


# -------------------------------------------------------------------- #
# 2) 2D 轴对称 vs 3D 交叉验证、sigma -> 0 一致性
# -------------------------------------------------------------------- #
def _cyl_vs_cart_sim(mesh, freq=1e4, d_rx=20.0):
    """轴上 z 磁偶极子 + 径向偏移接收器（总场 Hz 实部）的 b-form 仿真。"""
    rx = fdem.receivers.PointMagneticField(
        np.array([d_rx, 0.0, 0.0]), "z", "real", data_type="field"
    )
    src = fdem.sources.MagDipole(
        [rx], frequency=freq, location=np.array([0.0, 0.0, 0.0]),
        orientation="z", moment=1.0,
    )
    return fdem.Simulation3DMagneticFluxDensity(
        mesh, survey=fdem.Survey([src]),
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
    )


@pytest.mark.slow
def test_2d_cyl_vs_3d_crosscheck():
    """同一物理问题（全空间 sigma=0.1, f=1e4 Hz）2D 圆柱 vs 3D 直角 <= 15%。"""
    sigma = 0.1
    mesh_cyl = build_2d_cyl_mesh(r_core=60.0, z_core=60.0, cell_size=10.0,
                                 n_pad=6, pad_factor=1.3)
    mesh_3d = build_3d_tensor_mesh(core_extent=(60.0, 60.0, 60.0),
                                   cell_size=10.0, n_pad=6, pad_factor=1.3)
    d_cyl = _cyl_vs_cart_sim(mesh_cyl).dpred(sigma * np.ones(mesh_cyl.n_cells))[0]
    d_3d = _cyl_vs_cart_sim(mesh_3d).dpred(sigma * np.ones(mesh_3d.n_cells))[0]
    rel = abs(d_cyl - d_3d) / abs(d_3d)
    assert rel <= TOL_XCHECK, f"2D-3D 交叉偏差 {rel:.3%} (2D={d_cyl:.4e}, 3D={d_3d:.4e})"


@pytest.mark.slow
def test_sigma_vanishing_consistency():
    """sigma -> 0 时总场趋于自由空间一次场，1e-9 与 1e-12 S/m 差异 <= 2%。"""
    mesh_cyl = build_2d_cyl_mesh(r_core=60.0, z_core=60.0, cell_size=10.0,
                                 n_pad=6, pad_factor=1.3)
    sim = _cyl_vs_cart_sim(mesh_cyl)
    t9 = sim.dpred(1e-9 * np.ones(mesh_cyl.n_cells))[0]
    t12 = sim.dpred(1e-12 * np.ones(mesh_cyl.n_cells))[0]
    rel = abs(t9 - t12) / abs(t12)
    assert rel <= TOL_SIGMA0, f"sigma->0 一致性偏差 {rel:.3%}"


# -------------------------------------------------------------------- #
# 3) check_derivative 梯度校验（决策清单 E4）
# -------------------------------------------------------------------- #
def test_check_derivative_1d():
    """Simulation1DLayered 的 Jvec 与方向导数一致性（discretize 梯度校验）。"""
    cfg = LWDToolConfig(
        frequencies=(1e5, 5e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
    )
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=np.array([2.0]))
    sim = fwd.build_simulation(maps.IdentityMap(nP=2))
    m_true = np.array([0.1, 0.5])

    def fctn(m):
        f = sim.fields(m)
        return sim.dpred(m, f=f), lambda v: sim.Jvec(m, v, f=f)

    assert check_derivative(fctn, m_true, num=4, plotIt=False,
                            expectedOrder=2, random_seed=7)


@pytest.mark.slow
def test_check_derivative_2d_cyl():
    """小尺度 2D 轴对称 b-form 的 Jvec 梯度校验。"""
    cfg = LWDToolConfig(
        frequencies=(1e4,),
        spacings=np.array([1.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    mesh = build_2d_cyl_mesh(r_core=6.0, z_core=16.0, cell_size=1.0,
                             n_pad=4, pad_factor=1.35)
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)
    sim = fwd.build_simulation(maps.IdentityMap(nP=mesh.n_cells))
    m_true = 0.1 * np.ones(mesh.n_cells)

    def fctn(m):
        f = sim.fields(m)
        return sim.dpred(m, f=f), lambda v: sim.Jvec(m, v, f=f)

    assert check_derivative(fctn, m_true, num=3, plotIt=False,
                            expectedOrder=2, random_seed=11)
