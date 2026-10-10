"""垂直井三层地层 LWD-FDEM 正演 —— 仪器沿深度连续穿透扫描（2D 轴对称）。

物理场景（感应测井纵向扫描）：垂直磁偶极子源 + 单个固定轴向收发距 L 的点
磁探头接收器；整个 Tx–Rx 阵列沿井轴深度 z 从上部地层连续向下穿透，逐点记录
二次场，得到"响应 vs 仪器深度"曲线，从而反映穿越各地层界面时的边界异常。

  - 上覆（overburden）：z < -4 m，σ = 0.20 S/m
  - 储层（reservoir） ：-4 ≤ z ≤ +4 m，σ = 0.01 S/m
  - 基底（basement）  ：z > +4 m，σ = 0.50 S/m

关键设置：
  * 源：竖直磁偶极子 MagDipole(orientation="z")（绕垂直工具的小线圈）
  * 接收器：点磁探头 PointMagneticFluxDensitySecondary（点接收，小线圈近似）
  * 求解：2D 轴对称柱网格 + b-form（柱对称必须用 EB formulation）
  * 频率：20 / 100 kHz（去掉 500 kHz——其在导电基底趋肤深度≈1 m，网格难满足）
  * 网格：cell=0.5 m，使 cells/δ ≥ 3

本脚本仅新增/自维护，复用 src/lwd_fdem 公共设施（apply_rcparams / save_figure /
case_fig_dir / build_2d_cyl_mesh 等），不修改任何原有项目代码。

运行（仓库根目录）：
    conda activate lwd
    python examples/vertical_well_3layer.py

输出：examples/figs/vertical_well_3layer/
Copyright 2026 — Apache-2.0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.fdemcore import maps
from src.fdemcore.electromagnetics import frequency_domain as fdem
from src.fdemcore.electromagnetics.frequency_domain import receivers as _rx
from src.fdemcore.electromagnetics.frequency_domain import sources as _src
from src.lwd_fdem import (
    apply_rcparams,
    build_2d_cyl_mesh,
    case_fig_dir,
    mesh_quality_report,
    save_figure,
    skin_depth,
)

FIG_DIR = case_fig_dir(__file__)   # 输出到 figs/vertical_well_3layer/


# ================================================================ #
# 步骤 0：全局参数
# ================================================================ #

LANG = "en"                       # "en"=英文 SCI 图版；"zh"=中文


def T(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


# ---- 地层模型（水平三层，按 z 分层）----
Z_BOUNDARIES = np.array([-4.0, 4.0])          # 储层 -4~+4 m
SIGMA_LAYERS = np.array([0.20, 0.01, 0.50])   # 上覆 / 储层 / 基底 (S/m)
LAYER_NAMES = {
    "zh": ["上覆泥岩", "砂岩储层", "水侵基底"],
    "en": ["Overburden", "Reservoir", "Basement"],
}
LAYER_COLORS = ["#b8845a", "#f5e8c8", "#6aaed6"]

# ---- 仪器（单收发距深度扫描）----
FREQUENCIES = (2e4, 1e5)          # 20 / 100 kHz
OFFSET = 1.0                      # 固定轴向收发距 L (m)
TOOL_DEPTHS = np.linspace(-11.0, 11.0, 35)    # 仪器(源)深度扫描范围 (m)

# ---- 网格（加密到 0.5 m 保证 cells/δ≥3）----
R_CORE, Z_CORE, CELL = 5.0, 28.0, 0.5
N_PAD, PAD_FACTOR = 6, 1.3


def sigma_from_z(z):
    z = np.asarray(z, dtype=float)
    s = np.empty_like(z)
    s[z < Z_BOUNDARIES[0]] = SIGMA_LAYERS[0]
    s[(z >= Z_BOUNDARIES[0]) & (z <= Z_BOUNDARIES[1])] = SIGMA_LAYERS[1]
    s[z > Z_BOUNDARIES[1]] = SIGMA_LAYERS[2]
    return s


def build_mesh():
    return build_2d_cyl_mesh(r_core=R_CORE, z_core=Z_CORE, cell_size=CELL,
                             n_pad=N_PAD, pad_factor=PAD_FACTOR)


def build_scan_simulation(mesh, sigma_map=None):
    """构建"深度扫描"正演：每个仪器深度一个磁偶极子源，配一个 +L 处的点接收器。

    数据排列：频率 -> 仪器深度 -> (real, imag)，形状 (n_f, n_depth, 2)。
    """
    source_list = []
    for f in FREQUENCIES:
        for zt in TOOL_DEPTHS:
            rx_loc = np.array([0.0, 0.0, zt + OFFSET])
            rx_list = [
                _rx.PointMagneticFluxDensitySecondary(
                    rx_loc, orientation="z", component=c, data_type="field")
                for c in ("real", "imag")
            ]
            src = _src.MagDipole(
                rx_list, frequency=float(f),
                location=np.array([0.0, 0.0, float(zt)]),
                orientation="z", moment=1.0)
            source_list.append(src)
    survey = fdem.Survey(source_list)
    sim = fdem.Simulation3DMagneticFluxDensity(
        mesh=mesh, survey=survey,
        sigmaMap=sigma_map or maps.IdentityMap(nP=int(mesh.n_cells)))
    return sim, survey


def scan_dpred(sim, sigma):
    """正演并整形为 (n_f, n_depth, 2)。"""
    d = np.asarray(sim.dpred(np.asarray(sigma, dtype=float)))
    return d.reshape(len(FREQUENCIES), len(TOOL_DEPTHS), 2)


# ================================================================ #
# 图 01：地质概念模型 —— 仪器沿深度连续穿透
# ================================================================ #

def fig_01_concept_model():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    z_top, z_bot = -14.0, 14.0
    bands = [(z_top, Z_BOUNDARIES[0], 0), (Z_BOUNDARIES[0], Z_BOUNDARIES[1], 1),
             (Z_BOUNDARIES[1], z_bot, 2)]
    for z0, z1, idx in bands:
        ax.add_patch(Rectangle((0, z0), 12, z1 - z0, facecolor=LAYER_COLORS[idx],
                               edgecolor="k", lw=1.0, alpha=0.90, zorder=1))
        ax.text(8.5, (z0 + z1) / 2,
                f"{LAYER_NAMES[LANG][idx]}\n$\\sigma={SIGMA_LAYERS[idx]:.2f}$ S/m",
                ha="center", va="center", fontsize=10, zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.82))
    for zb in Z_BOUNDARIES:
        ax.axhline(zb, color="k", lw=1.5, zorder=2)

    # 井轴（r=1.5 竖线）+ 扫描方向箭头
    ax.plot([1.5, 1.5], [z_top, z_bot], color="0.4", lw=1.2, ls=":", zorder=3)
    ax.annotate("", xy=(0.7, z_bot - 1), xytext=(0.7, z_top + 1),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="darkred"))
    ax.text(0.4, 0, T("扫描\n深度", "scan\ndepth"), fontsize=8, color="darkred",
            ha="center", va="center")

    # 仪器在三个代表性深度（上覆/储层/基底）：Tx(红)+Rx(蓝)，Rx 在 Tx 下方 L
    for zt, alpha in [(-8.0, 0.5), (0.0, 1.0), (8.0, 0.5)]:
        ax.add_patch(Rectangle((1.35, zt - 0.35), 0.3, 0.7, facecolor="crimson",
                               edgecolor="k", alpha=alpha, zorder=6))
        ax.add_patch(Rectangle((1.35, zt + OFFSET - 0.3), 0.3, 0.6,
                               facecolor="tab:blue", edgecolor="k",
                               alpha=alpha, zorder=6))
    ax.text(1.9, -8.0, "Tx", fontsize=7, color="crimson", va="center")
    ax.text(1.9, -8.0 + OFFSET, "Rx", fontsize=7, color="tab:blue", va="center")
    ax.text(1.9, 0.0, T("Tx/Rx 阵列\n(收发距 L=%.0f m)" % OFFSET,
                        "Tx/Rx array\n(offset L=%.0f m)" % OFFSET),
            fontsize=7.5, color="k", va="center")

    ax.set_xlim(-0.5, 12.5); ax.set_ylim(z_bot, z_top)
    ax.set_xticks([])
    ax.set_ylabel(T("深度 z (m)", "Depth z (m)"), fontsize=11)
    ax.set_title(T("图 01  垂直井连续穿透三层地层的深度扫描示意",
                   "Fig. 01  Vertical well - continuous penetration depth scan"),
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v01_concept_model.png", dpi=200)


# ================================================================ #
# 图 02：计算网格与趋肤深度校核
# ================================================================ #

def fig_02_mesh(mesh):
    cc = np.asarray(mesh.cell_centers)
    sig = sigma_from_z(cc[:, 2])
    ls = np.log10(np.maximum(sig, 1e-12))

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.1]},
                                   figsize=(12, 5.5))
    ax1.tripcolor(cc[:, 0], cc[:, 2], ls, cmap="viridis", shading="flat")
    for zb in Z_BOUNDARIES:
        ax1.axhline(zb, color="w", lw=1.0, ls="--")
    ax1.set(xlabel="r (m)", ylabel=T("深度 z (m)", "Depth z (m)"),
            title=T("(a) 轴对称 r-z 网格（$\\log_{10}\\sigma$）",
                    "(a) Axisymmetric r-z mesh ($\\log_{10}\\sigma$)"))

    freqs_plot = np.logspace(np.log10(FREQUENCIES[0]) - 0.4,
                             np.log10(FREQUENCIES[-1]) + 0.4, 50)
    names = (["上覆", "储层", "基底"] if LANG == "zh"
             else ["overburden", "reservoir", "basement"])
    colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, 3))
    for s, col, nm in zip(SIGMA_LAYERS, colors_f, names):
        ax2.semilogx(freqs_plot, [skin_depth(f, s) for f in freqs_plot],
                     lw=2.0, color=col, label=f"{nm} {s:.2g} S/m")
    report = mesh_quality_report(mesh, FREQUENCIES, SIGMA_LAYERS)
    ok_str = "OK" if report["ok"] else "NG"
    crit = T("判据(>=3)", "criterion (>=3)")
    ax2.text(0.02, 0.97,
             f"cells = {mesh.n_cells}, cell = {CELL:.1f} m\n"
             f"min $\\delta$ = {report['min_skin_depth']:.2f} m\n"
             f"cells/$\\delta$ = {report['cells_per_skin_depth']:.1f}  ({crit}: {ok_str})",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85))
    ax2.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
            ylabel=T("趋肤深度 δ (m)", "Skin depth $\\delta$ (m)"),
            title=T("(b) 趋肤深度与网格分辨率", "(b) Skin depth vs resolution"))
    ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle(T("图 02  计算网格与趋肤深度校核（cell=0.5 m）",
                   "Fig. 02  Mesh & skin-depth check (cell=0.5 m)"),
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v02_mesh.png", dpi=200)


# ================================================================ #
# 图 03：扫描观测系统示意
# ================================================================ #

def fig_03_scan_geometry():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    z_top, z_bot = -14.0, 14.0
    for (z0, z1, idx) in [(z_top, Z_BOUNDARIES[0], 0),
                          (Z_BOUNDARIES[0], Z_BOUNDARIES[1], 1),
                          (Z_BOUNDARIES[1], z_bot, 2)]:
        ax.add_patch(Rectangle((0, z0), 6, z1 - z0, facecolor=LAYER_COLORS[idx],
                               edgecolor="k", lw=0.8, alpha=0.55, zorder=1))
        ax.text(5.0, (z0 + z1) / 2, LAYER_NAMES[LANG][idx], ha="center",
                va="center", fontsize=9, zorder=3)
    # 仪器在多个深度：源(红点) + 接收器(蓝点, +L)
    for zt in TOOL_DEPTHS:
        ax.plot(1.0, zt, "o", ms=4, color="crimson", zorder=5)
        ax.plot(1.6, zt + OFFSET, "s", ms=4, color="tab:blue", zorder=5)
        ax.plot([1.0, 1.6], [zt, zt + OFFSET], color="0.5", lw=0.5, zorder=4)
    ax.text(1.0, z_top + 0.6, T("Tx(源)", "Tx"), ha="center", fontsize=8,
            color="crimson")
    ax.text(1.6, z_top + 0.6, T("Rx(点探头)", "Rx"), ha="center", fontsize=8,
            color="tab:blue")
    ax.set_xlim(0, 6); ax.set_ylim(z_bot, z_top); ax.set_xticks([])
    ax.set_ylabel(T("仪器深度 z (m)", "Tool depth z (m)"), fontsize=10)
    ax.set_title(T(f"(a) 固定收发距 L={OFFSET:.0f} m、源沿深度扫描的观测系统",
                   f"(a) Fixed offset L={OFFSET:.0f} m, source scanned in depth"),
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v03_scan_geometry.png", dpi=200)


# ================================================================ #
# 图 04：正演响应 vs 仪器深度（连续穿透核心图）
# ================================================================ #

def fig_04_forward_scan(mesh):
    sim, _ = build_scan_simulation(mesh)
    sigma = sigma_from_z(np.asarray(mesh.cell_centers)[:, 2])
    cube = scan_dpred(sim, sigma)          # (n_f, n_depth, 2): [:,0]=Re, [:,1]=Im

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors_f = plt.cm.plasma(np.linspace(0.15, 0.85, len(FREQUENCIES)))

    def shade(ax):
        for (z0, z1, idx) in [(-14, Z_BOUNDARIES[0], 0),
                              (Z_BOUNDARIES[0], Z_BOUNDARIES[1], 1),
                              (Z_BOUNDARIES[1], 14, 2)]:
            ax.axhspan(z0, z1, color=LAYER_COLORS[idx], alpha=0.22, zorder=0)

    def plot_depth(ax, get, logx, marker):
        for i, (f, col) in enumerate(zip(FREQUENCIES, colors_f)):
            (ax.semilogx if logx else ax.plot)(
                get(i), TOOL_DEPTHS, marker + "-", ms=4, lw=1.6, color=col,
                label=f"{f/1e3:.0f} kHz")
        ax.invert_yaxis(); ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    # (a) 实部 Re(Bz_s)（带符号，线性）
    ax = axes[0, 0]; shade(ax)
    plot_depth(ax, lambda i: cube[i, :, 0], False, "o")
    ax.set(xlabel=r"$\mathrm{Re}(B_z^{\mathrm{s}})$ (T)",
           ylabel=T("仪器深度 (m)", "Tool depth (m)"),
           title=T("(a) 实部 Re — 连续穿透", "(a) In-phase (Re) vs depth"))

    # (b) 虚部 Im(Bz_s)（带符号，线性）
    ax = axes[0, 1]; shade(ax)
    plot_depth(ax, lambda i: cube[i, :, 1], False, "s")
    ax.set(xlabel=r"$\mathrm{Im}(B_z^{\mathrm{s}})$ (T)",
           ylabel=T("仪器深度 (m)", "Tool depth (m)"),
           title=T("(b) 虚部 Im — 连续穿透", "(b) Quadrature (Im) vs depth"))

    # (c) 幅值 |Bz_s|（对数）
    ax = axes[1, 0]; shade(ax)
    plot_depth(ax, lambda i: np.hypot(cube[i, :, 0], cube[i, :, 1]), True, "^")
    ax.set(xlabel=r"$|B_z^{\mathrm{s}}|$ (T)",
           ylabel=T("仪器深度 (m)", "Tool depth (m)"),
           title=T("(c) 幅值 |Bz| — 连续穿透", "(c) Amplitude vs depth"))

    # (d) 相位（对数）
    ax = axes[1, 1]; shade(ax)
    plot_depth(ax, lambda i: np.degrees(
        np.arctan2(cube[i, :, 1], cube[i, :, 0])), False, "D")
    ax.set(xlabel=T("相位 (°)", "Phase ($^\\circ$)"),
           ylabel=T("仪器深度 (m)", "Tool depth (m)"),
           title=T("(d) 相位 — 连续穿透", "(d) Phase vs depth"))

    fig.suptitle(T("图 04  垂直井深度扫描响应：实部/虚部/幅值/相位（磁偶极子·点接收）",
                   "Fig. 04  Vertical-well depth-scan response: Re / Im / |Bz| / phase "
                   "(mag. dipole, point receiver)"),
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v04_forward_scan.png", dpi=200)
    return cube


# ================================================================ #
# 图 05：视电导率 vs 深度（由虚部归一的测井式曲线）
# ================================================================ #

def fig_05_apparent(mesh):
    sim, _ = build_scan_simulation(mesh)
    sigma = sigma_from_z(np.asarray(mesh.cell_centers)[:, 2])
    cube = scan_dpred(sim, sigma)
    # 均匀参考：用各频率在处中仪器位于储层中心的 Im 作归一，得定性视电导率
    im = np.abs(cube[:, :, 1])                              # (n_f, n_depth)
    ref = im[:, np.argmin(np.abs(TOOL_DEPTHS))]            # (n_f,) 储层中心参考
    im_norm = im / ref[:, None] * SIGMA_LAYERS[1]          # 定性视电导率刻度

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for (z0, z1, idx) in [(-14, Z_BOUNDARIES[0], 0),
                          (Z_BOUNDARIES[0], Z_BOUNDARIES[1], 1),
                          (Z_BOUNDARIES[1], 14, 2)]:
        ax.axhspan(z0, z1, color=LAYER_COLORS[idx], alpha=0.25, zorder=0)
    colors_f = plt.cm.plasma(np.linspace(0.15, 0.85, len(FREQUENCIES)))
    for i, (f, col) in enumerate(zip(FREQUENCIES, colors_f)):
        ax.semilogx(im_norm[i, :], TOOL_DEPTHS, "o-", ms=4, lw=1.6, color=col,
                    label=f"{f/1e3:.0f} kHz")
    ax.axvline(SIGMA_LAYERS[0], color="0.5", ls=":", lw=1)
    ax.axvline(SIGMA_LAYERS[2], color="0.5", ls=":", lw=1)
    ax.set(xlabel=T("定性视电导率 (S/m)", "Qualitative apparent $\\sigma$ (S/m)"),
           ylabel=T("仪器深度 (m)", "Tool depth (m)"),
           title=T("图 05  视电导率纵向扫描曲线", "Fig. 05  Apparent-$\\sigma$ log"))
    ax.invert_yaxis(); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v05_apparent.png", dpi=200)


# ================================================================ #
# 主流程
# ================================================================ #

def main():
    apply_rcparams(LANG)
    print("=" * 60)
    print("  垂直井三层地层 LWD-FDEM 深度扫描正演（连续穿透）")
    print("=" * 60)
    print(f"层界 z={Z_BOUNDARIES.tolist()}, σ={SIGMA_LAYERS.tolist()} S/m")
    print(f"源=磁偶极子(点), 接收=点磁探头, 收发距 L={OFFSET} m, "
          f"扫描深度 {TOOL_DEPTHS[0]:.0f}~{TOOL_DEPTHS[-1]:.0f} m ({TOOL_DEPTHS.size} 点)")
    print(f"频率 {FREQUENCIES} Hz")

    mesh = build_mesh()
    report = mesh_quality_report(mesh, FREQUENCIES, SIGMA_LAYERS)
    print(f"网格 {mesh.n_cells} 单元, cells/δ={report['cells_per_skin_depth']:.1f}, "
          f"ok={report['ok']:.0f}")

    print("\n[图 01] 概念模型 ...")
    fig_01_concept_model()
    print("[图 02] 网格与趋肤深度 ...")
    fig_02_mesh(mesh)
    print("[图 03] 扫描观测系统 ...")
    fig_03_scan_geometry()
    print("[图 04] 深度扫描正演响应 ...")
    fig_04_forward_scan(mesh)
    print("[图 05] 视电导率曲线 ...")
    fig_05_apparent(mesh)

    print(f"\n全部图版已保存至: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
