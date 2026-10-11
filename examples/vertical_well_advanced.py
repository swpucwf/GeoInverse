"""垂直井进阶示例 —— 2D 轴对称下演示源类型对比与反演正则化对比。

承接 horizontal_well_advanced.py：回线源 CircularLoop 在 1D 层状只支持零偏移
中心回线，无法与径向点接收对比；而 2D 轴对称（柱网格）天然支持带轴向偏移的
回线源，故源型对比放在这里。

  图 A  源类型对比：磁偶极子 MagDipole  vs  有限回线 CircularLoop（2D 深度扫描）
  图 B  反演正则化对比：Occam 平滑 (sparse=False)  vs  稀疏块状 (sparse=True)
        （业务层子类封装修正 LWDFDEMInversion sparse 的 BetaSchedule/UpdateIRLS 冲突，
          不改底层 fdemcore 与 lwd_fdem 原模块）

模型：垂直井穿三层水平地层，2D 轴对称柱网格 + 轴向偏移 + b-form（Bz 二次场，
柱对称必须用 EB formulation）。复用 src/lwd_fdem 公共设施，仅新增文件。

运行：conda activate lwd && python examples/vertical_well_advanced.py
输出：examples/figs/vertical_well_advanced/
Copyright 2026 — Apache-2.0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from src.fdemcore import maps
from src.fdemcore.electromagnetics import frequency_domain as fdem
from src.fdemcore.electromagnetics.frequency_domain import receivers as _rx
from src.fdemcore.electromagnetics.frequency_domain import sources as _src
from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    add_synthetic_noise, apply_rcparams, build_2d_cyl_mesh, case_fig_dir,
    make_data_object, save_figure,
)

FIG_DIR = case_fig_dir(__file__)

LANG = "en"


def T(zh, en):
    return zh if LANG == "zh" else en


# ---- 地层（按 z 分三层）----
Z_BOUNDARIES = np.array([-4.0, 4.0])
SIGMA_LAYERS = np.array([0.20, 0.01, 0.50])
LAYER_NAMES = {"zh": ["上覆", "储层", "基底"], "en": ["overburden", "reservoir", "basement"]}
LAYER_COLORS = ["#b8845a", "#f5e8c8", "#6aaed6"]

# ---- 仪器 ----
FREQUENCIES = (2e4, 5e4, 1e5)                 # 低/中/高三条代表频
OFFSET = 1.0
TOOL_DEPTHS = np.linspace(-9.0, 9.0, 15)      # 深度扫描点（2D 求解，取适中数量）
LOOP_RADIUS = 0.5

# ---- 网格（2D 轴对称）----
R_CORE, Z_CORE, CELL, N_PAD = 5.0, 20.0, 1.0, 4


def sigma_from_z(z):
    z = np.asarray(z, dtype=float)
    s = np.empty_like(z)
    s[z < Z_BOUNDARIES[0]] = SIGMA_LAYERS[0]
    s[(z >= Z_BOUNDARIES[0]) & (z <= Z_BOUNDARIES[1])] = SIGMA_LAYERS[1]
    s[z > Z_BOUNDARIES[1]] = SIGMA_LAYERS[2]
    return s


def build_mesh():
    return build_2d_cyl_mesh(r_core=R_CORE, z_core=Z_CORE, cell_size=CELL,
                             n_pad=N_PAD, pad_factor=1.3)


# ================================================================ #
# 图 A：源类型对比（MagDipole vs CircularLoop，2D 深度扫描）
# ================================================================ #

def _build_scan_sim(mesh, source_type):
    source_list = []
    for f in FREQUENCIES:
        for zt in TOOL_DEPTHS:
            rx_loc = np.array([0.0, 0.0, zt + OFFSET])
            rx_list = [
                _rx.PointMagneticFluxDensitySecondary(
                    rx_loc, orientation="z", component=c, data_type="field")
                for c in ("real", "imag")
            ]
            if source_type == "mag_dipole":
                src = _src.MagDipole(rx_list, frequency=float(f),
                                     location=np.array([0.0, 0.0, float(zt)]),
                                     orientation="z", moment=1.0)
            else:
                src = _src.CircularLoop(rx_list, frequency=float(f),
                                        location=np.array([0.0, 0.0, float(zt)]),
                                        orientation="z", radius=LOOP_RADIUS)
            source_list.append(src)
    survey = fdem.Survey(source_list)
    return fdem.Simulation3DMagneticFluxDensity(
        mesh=mesh, survey=survey, sigmaMap=maps.IdentityMap(nP=int(mesh.n_cells)))


def fig_A_source_compare(mesh):
    sigma = sigma_from_z(np.asarray(mesh.cell_centers)[:, 2])
    n_f, n_d = len(FREQUENCIES), len(TOOL_DEPTHS)
    cube_dip = np.asarray(_build_scan_sim(mesh, "mag_dipole").dpred(sigma)).reshape(n_f, n_d, 2)
    cube_lp = np.asarray(_build_scan_sim(mesh, "circular_loop").dpred(sigma)).reshape(n_f, n_d, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    colors_f = plt.cm.plasma(np.linspace(0.15, 0.85, n_f))
    for i, (f, col) in enumerate(zip(FREQUENCIES, colors_f)):
        a_dip = np.hypot(cube_dip[i, :, 0], cube_dip[i, :, 1])
        a_lp = np.hypot(cube_lp[i, :, 0], cube_lp[i, :, 1])
        ax1.semilogx(a_dip, TOOL_DEPTHS, "o-", ms=3.5, lw=1.5, color=col,
                     label=f"{f/1e3:.0f} kHz " + T("偶极", "dipole"))
        ax1.semilogx(a_lp, TOOL_DEPTHS, "s--", ms=3.5, lw=1.3, color=col, alpha=0.7,
                     label=f"{f/1e3:.0f} kHz " + T("回线", "loop"))
        ax2.plot((a_lp - a_dip) / a_dip * 100, TOOL_DEPTHS, "-", ms=4, lw=1.8,
                 color=col, label=f"{f/1e3:.0f} kHz", marker="o", markevery=3)
    for ax in (ax1, ax2):
        for (z0, z1, idx) in [(-12, Z_BOUNDARIES[0], 0),
                              (Z_BOUNDARIES[0], Z_BOUNDARIES[1], 1),
                              (Z_BOUNDARIES[1], 12, 2)]:
            ax.axhspan(z0, z1, color=LAYER_COLORS[idx], alpha=0.2, zorder=0)
        ax.invert_yaxis(); ax.grid(True, which="both", alpha=0.3)
    ax1.set(xlabel=r"$|B_z^{\mathrm{s}}|$ (T)", ylabel=T("仪器深度 (m)", "Tool depth (m)"),
            title=T("(a) 磁偶极子(实) vs 回线(虚)", "(a) dipole (solid) vs loop (dashed)"))
    ax1.legend(fontsize=6.5, ncol=2)
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlim(-22, 2)  # 收紧横坐标，让曲线更分散
    ax2.set(xlabel=T("相对差异 (%)", "Relative diff (%)"),
            ylabel=T("仪器深度 (m)", "Tool depth (m)"),
            title=T("(b) 回线相对偶极子差异", "(b) loop vs dipole difference"))
    ax2.legend(fontsize=7)
    fig.suptitle(T("图 A  2D 轴对称源类型对比：磁偶极子 vs 有限回线 CircularLoop",
                   "Fig. A  2D axisymmetric source compare: MagDipole vs CircularLoop"),
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "A_source_compare.png", dpi=200)


# ================================================================ #
# 图 B：反演正则化对比（Occam vs Sparse，2D 阵列）
# ================================================================ #

def fig_B_inversion_compare(mesh):
    cfg = LWDToolConfig(
        frequencies=(2e4, 1e5), spacings=np.array([1.0, 2.0, 3.0]),
        components=("real", "imag"), field_kind="b", secondary=True,
        receiver_geometry="axial",
    )
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)
    sigma_true = sigma_from_z(np.asarray(mesh.cell_centers)[:, 2])
    dobs, std = add_synthetic_noise(fwd.dpred(sigma_true), 0.03, seed=7)
    data = make_data_object(fwd.survey, dobs, absolute_std=std)
    m0 = np.log(0.1 * np.ones(mesh.n_cells))

    def run(sparse):
        inv = LWDFDEMInversion(
            fwd, data, m0=m0, max_iter=5, sparse=sparse,
            norms=[0, 1, 1] if sparse else None,   # 2D: norms 长度需 = 1+dim = 3
            alpha_s=1e-3, alpha_x=1.0, alpha_z=1.0,
            beta0_ratio=0.1, beta_cooling=3.0, chifact=1.0,
            sensitivity_weights=False, preconditioner=not sparse,
            optimizer="pgncg" if sparse else "gn",
        )
        return inv.run()

    rec_smooth, _ = run(sparse=False)
    rec_sparse, _ = run(sparse=True)

    cc = np.asarray(mesh.cell_centers)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, val, ttl in [
        (axes[0], sigma_true, T("真实模型", "True model")),
        (axes[1], rec_smooth, T("Occam 平滑回收", "Occam smooth")),
        (axes[2], rec_sparse, T("稀疏块状回收", "Sparse blocky")),
    ]:
        ax.tripcolor(cc[:, 0], cc[:, 2], np.log10(np.maximum(val, 1e-12)),
                     cmap="viridis", shading="flat")
        for zb in Z_BOUNDARIES:
            ax.axhline(zb, color="w", lw=0.8, ls="--")
        ax.set(xlabel="r (m)", ylabel=T("深度 z (m)", "Depth z (m)"), title=ttl)
    fig.suptitle(T("图 B  2D 反演正则化对比：Occam 平滑 vs 稀疏块状（业务层子类封装 sparse）",
                   "Fig. B  2D inversion: Occam smooth vs Sparse (business-layer subclass)"),
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "B_inversion_compare.png", dpi=200)


def main():
    apply_rcparams(LANG)
    print("=" * 58)
    print("  垂直井进阶：源类型对比(2D) + 反演正则化对比")
    print("=" * 58)
    mesh = build_mesh()
    print(f"网格 {mesh.n_cells} 单元")
    print("[图 A] 源类型对比 (MagDipole vs CircularLoop) ...")
    fig_A_source_compare(mesh)
    print("[图 B] 反演正则化对比 (Occam vs Sparse) ...")
    fig_B_inversion_compare(mesh)
    print(f"\n图版已保存至: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
