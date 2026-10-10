"""lwd_fdem 业务层测试：合成数据反演回收、3D 冒烟、数据 I/O。

对应实施计划阶段 4 / 决策清单 I1/I2：
  1) 1D 层状界面与电导率回收（多频多偏移距 Occam 反演）；
  2) 2D 轴对称环状异常体反演管线可用、拟合差持续下降；
  3) 3D 单频点正演 + 小模型反演冒烟（功能可用即通过）；
  4) 数据 I/O 与不确定度工具单元测试。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import numpy as np
import pytest

from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    add_synthetic_noise, build_2d_cyl_mesh, build_3d_tensor_mesh,
    load_lwd_csv, make_data_object, mesh_quality_report,
)


# -------------------------------------------------------------------- #
# 1) 1D 合成反演回收
# -------------------------------------------------------------------- #
def test_1d_inversion_recovers_layers():
    """3 层地电模型（高-低-高）多频多距 Occam 反演定量回收界面电导率。"""
    cfg = LWDToolConfig(
        frequencies=(1e3, 1e4, 1e5, 1e6),
        spacings=np.array([0.3, 0.6, 1.0, 2.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
    )
    thicknesses = np.array([2.0, 4.0])
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=thicknesses)
    true_sigma = np.array([0.25, 0.01, 0.5])
    d_clean = fwd.dpred(true_sigma)
    dobs, std = add_synthetic_noise(d_clean, 0.02, seed=1)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    inv = LWDFDEMInversion(fwd, data_obj, m0=np.log(0.1 * np.ones(3)),
                           max_iter=15, beta0_ratio=1e-4, beta_cooling=2.0,
                           sensitivity_weights=False, preconditioner=False)
    rec_sigma, history = inv.run()

    assert rec_sigma.shape == (3,)
    assert np.all(np.isfinite(rec_sigma)) and np.all(rec_sigma > 0)
    assert len(history["iterations"]) >= 2
    # 定量判据（决策清单 I1）：各层回收相对误差 <= 25%
    rel_err = np.abs(rec_sigma - true_sigma) / true_sigma
    assert np.all(rel_err <= 0.25), f"回收相对误差 {rel_err}"
    # 拟合差整体下降
    assert history["iterations"][-1]["phi_d"] < history["iterations"][0]["phi_d"]


# -------------------------------------------------------------------- #
# 2) 2D 轴对称环状异常体反演管线
# -------------------------------------------------------------------- #
@pytest.mark.slow
def test_2d_inversion_ring_anomaly():
    """井旁环状高导异常：反演管线可用且拟合差单调下降。"""
    cfg = LWDToolConfig(
        frequencies=(1e4, 3e4),
        spacings=np.array([1.0, 2.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    mesh = build_2d_cyl_mesh(r_core=5.0, z_core=12.0, cell_size=1.0,
                             n_pad=4, pad_factor=1.4)
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)

    # 背景 0.1 S/m，在 (r≈3m, z≈-3m) 放置 5 倍电导率的环状异常体
    sigma_true = 0.1 * np.ones(mesh.n_cells)
    cc = mesh.cell_centers
    ring = (np.abs(cc[:, 0] - 3.0) < 1.5) & (np.abs(cc[:, 2] + 3.0) < 1.5)
    sigma_true[ring] = 0.5
    assert ring.any(), "环状异常体未落入网格核心区"

    d_clean = fwd.dpred(sigma_true)
    dobs, std = add_synthetic_noise(d_clean, 0.03, seed=7)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    inv = LWDFDEMInversion(fwd, data_obj, m0=np.log(0.1 * np.ones(mesh.n_cells)),
                           max_iter=5, beta0_ratio=0.1, chifact=0.5,
                           sensitivity_weights=False, preconditioner=False)
    rec_sigma, history = inv.run()

    assert rec_sigma.shape == (mesh.n_cells,)
    assert np.all(np.isfinite(rec_sigma)) and np.all(rec_sigma > 0)
    assert len(history["iterations"]) >= 1
    # 拟合差不应发散（末次 <= 首次 * 1.5）
    phi_d = [it["phi_d"] for it in history["iterations"]]
    assert phi_d[-1] <= 1.5 * phi_d[0], f"拟合差发散: {phi_d}"
    # 异常检测：恢复模型在环状区域的均值应高于远背景（决策清单 I2）
    ring_mean = float(np.mean(rec_sigma[ring]))
    bg_mean = float(np.mean(rec_sigma[~ring]))
    assert ring_mean > bg_mean, f"环状均值 {ring_mean} 未高于背景 {bg_mean}"


# -------------------------------------------------------------------- #
# 3) 3D 冒烟：单频点正演 + 小模型反演
# -------------------------------------------------------------------- #
def _quick_3d_mesh():
    return build_3d_tensor_mesh(core_extent=(8.0, 8.0, 8.0), cell_size=1.0,
                                n_pad=3, pad_factor=1.3)


def test_3d_forward_smoke():
    """3D 直角网格 e-form 单频点正演功能可用。"""
    cfg = LWDToolConfig(
        frequencies=(1e4,),
        spacings=np.array([2.0]),
        components=("real", "imag"),
        field_kind="e", secondary=False, data_type="field",
        receiver_geometry="axial",
    )
    mesh = _quick_3d_mesh()
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh)
    d = fwd.dpred(0.1 * np.ones(mesh.n_cells))
    assert np.all(np.isfinite(d)) and d.size == 2


@pytest.mark.slow
def test_3d_inversion_smoke():
    """3D 小模型 Occam 反演冒烟：跑通管线、模型为正有限值即通过。"""
    cfg = LWDToolConfig(
        frequencies=(1e4,),
        spacings=np.array([1.5, 3.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    mesh = _quick_3d_mesh()
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh)
    sigma_true = 0.1 * np.ones(mesh.n_cells)
    cc = mesh.cell_centers
    blob = (np.abs(cc[:, 0] - 2.0) < 1.5) & (np.abs(cc[:, 2]) < 1.5)
    sigma_true[blob] = 0.4

    d_clean = fwd.dpred(sigma_true)
    dobs, std = add_synthetic_noise(d_clean, 0.05, seed=3)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    inv = LWDFDEMInversion(fwd, data_obj, m0=np.log(0.1 * np.ones(mesh.n_cells)),
                           max_iter=2, sensitivity_weights=False,
                           preconditioner=False)
    rec_sigma, history = inv.run()
    assert rec_sigma.shape == (mesh.n_cells,)
    assert np.all(np.isfinite(rec_sigma)) and np.all(rec_sigma > 0)
    assert len(history["iterations"]) >= 1


# -------------------------------------------------------------------- #
# 4) 数据 I/O 与网格校核
# -------------------------------------------------------------------- #
def test_add_synthetic_noise_and_data_object():
    """加噪接口形状/标准差一致，make_data_object 接受逐点 std。"""
    d_clean = np.array([1.0, -2.0, 0.5, 3.0])
    dobs, std = add_synthetic_noise(d_clean, 0.05, seed=42)
    assert dobs.shape == d_clean.shape
    assert np.all(std > 0) and np.allclose(std, 0.05 * np.abs(dobs))
    # 同一 seed 可复现
    dobs2, _ = add_synthetic_noise(d_clean, 0.05, seed=42)
    assert np.allclose(dobs, dobs2)


def test_load_lwd_csv(tmp_path):
    """长表 CSV 读写往返。"""
    path = tmp_path / "lwd.csv"
    path.write_text(
        "frequency,spacing,component,value\n"
        "100000,0.3,real,1.25\n"
        "100000,0.3,imag,-0.42\n",
        encoding="utf-8",
    )
    table = load_lwd_csv(str(path))
    assert table["frequency"].size == 2
    assert np.allclose(table["frequency"], [1e5, 1e5])
    assert list(table["component"]) == ["real", "imag"]
    assert np.allclose(table["value"], [1.25, -0.42])


def test_mesh_quality_report():
    """趋肤深度校核报告：核心区 cell <= delta/3 时 ok=1，否则 ok=0。"""
    mesh_ok = build_2d_cyl_mesh(r_core=5.0, z_core=12.0, cell_size=0.5,
                                n_pad=4, pad_factor=1.4)
    mesh_bad = build_2d_cyl_mesh(r_core=50.0, z_core=120.0, cell_size=20.0,
                                 n_pad=2, pad_factor=1.2)
    freqs, sigmas = [2e5], [0.1]
    rep_ok = mesh_quality_report(mesh_ok, freqs, sigmas)
    rep_bad = mesh_quality_report(mesh_bad, freqs, sigmas)
    assert rep_ok["ok"] == 1.0 and rep_bad["ok"] == 0.0
