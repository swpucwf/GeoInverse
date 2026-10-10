"""垂直井三层地层 LWD-FDEM 正反演完整流程（2D 轴对称 · 轴向偏移）。

场景：垂直井穿过三层水平地层（经典感应测井几何），发射/接收线圈沿井轴
（z 方向）做轴向偏移，用轴对称柱坐标 FVM 求解 Hz 二次场。
  - 上覆（overburden）：z < -4 m，σ = 0.20 S/m
  - 储层（reservoir） ：-4 ≤ z ≤ +4 m，σ = 0.01 S/m（源位于中心 z=0）
  - 基底（basement）  ：z > +4 m，σ = 0.50 S/m

与水平井案例（horizontal_well_3layer.py）的区别：
  水平井 = 1D 层状 + 径向偏移（Hankel 解析）；
  垂直井 = 2D 轴对称柱网格 + 轴向偏移（FVM 数值）。

本脚本仅新增，复用 src/lwd_fdem 既有公共设施（apply_rcparams / save_figure /
LWDFDEMForward / LWDFDEMInversion 等），不修改任何原有代码。

运行（仓库根目录）：
    conda activate lwd
    python examples/vertical_well_3layer.py

输出目录：examples/figs/
Copyright 2026 — Apache-2.0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D

from src.lwd_fdem import (
    LWDFDEMForward,
    LWDFDEMInversion,
    LWDToolConfig,
    add_synthetic_noise,
    apply_rcparams,
    build_2d_cyl_mesh,
    make_data_object,
    mesh_quality_report,
    plot_inversion_figure,
    save_figure,
    skin_depth,
)

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


# ================================================================ #
# 步骤 0：全局参数定义（唯一修改入口）
# ================================================================ #

# ---- 图版语言 ----
# "en" = 英文 SCI 图版（默认）；"zh" = 中文图版。改此一行即切换全部图版语言。
LANG = "en"


def T(zh: str, en: str) -> str:
    """按全局 LANG 返回图面文字（中文或英文）。"""
    return zh if LANG == "zh" else en


# ---- 地层模型（按 z 分层的水平三层）----
Z_BOUNDARIES = np.array([-4.0, 4.0])         # 层界面深度 (m)：储层 -4~+4
SIGMA_LAYERS = np.array([0.20, 0.01, 0.50])  # 上覆 / 储层 / 基底 电导率 (S/m)
LAYER_NAMES = {
    "zh": ["上覆泥岩", "砂岩储层\n(目标)", "水侵基底"],
    "en": ["Overburden\n(shale)", "Reservoir\n(sandstone)", "Basement\n(aquifer)"],
}
LAYER_COLORS = ["#b8845a", "#f5e8c8", "#6aaed6"]

# ---- 仪器配置（轴向偏移，源在井轴 z=0）----
FREQUENCIES = (2e4, 1e5, 5e5)                # 3 个频率 (Hz)，控制 2D FVM 运行量
SPACINGS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 轴向收发距 (m)
COMPONENTS = ("real", "imag")

# ---- 网格参数 ----
R_CORE, Z_CORE, CELL = 6.0, 24.0, 1.0
N_PAD, PAD_FACTOR = 5, 1.3

# ---- 反演设置 ----
M0_SIGMA = 0.1
NOISE_LEVEL = 0.03
SEED = 42
MAX_ITER = 16          # 2D 轴对称参数多，需更多迭代才能拟合到目标


def sigma_from_z(z: np.ndarray) -> np.ndarray:
    """按单元深度 z 赋三层电导率（垂直井穿水平地层）。"""
    z = np.asarray(z, dtype=float)
    sigma = np.empty_like(z)
    sigma[z < Z_BOUNDARIES[0]] = SIGMA_LAYERS[0]          # 上覆
    mid = (z >= Z_BOUNDARIES[0]) & (z <= Z_BOUNDARIES[1])
    sigma[mid] = SIGMA_LAYERS[1]                           # 储层
    sigma[z > Z_BOUNDARIES[1]] = SIGMA_LAYERS[2]          # 基底
    return sigma


def build_forward():
    """构建垂直井正演计算器（2D 轴对称 + 轴向偏移 + Hz 二次场）。"""
    cfg = LWDToolConfig(
        frequencies=FREQUENCIES,
        spacings=SPACINGS,
        components=COMPONENTS,
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",     # 垂直井：接收器沿井轴 z 偏移
        moment=1.0, source_type="mag_dipole",
        orientation="z",
    )
    mesh = build_2d_cyl_mesh(r_core=R_CORE, z_core=Z_CORE, cell_size=CELL,
                             n_pad=N_PAD, pad_factor=PAD_FACTOR)
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)
    return fwd, cfg, mesh


def true_sigma(mesh) -> np.ndarray:
    """真实模型：按单元 z 深度赋三层电导率。"""
    cc = np.asarray(mesh.cell_centers)
    return sigma_from_z(cc[:, 2])


# ================================================================ #
# 图 01：地质概念模型（垂直井穿水平地层的侧视图）
# ================================================================ #

def fig_01_concept_model():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    z_top, z_bot = -13.0, 13.0
    bands = [
        (z_top, Z_BOUNDARIES[0], SIGMA_LAYERS[0], 0),   # 上覆
        (Z_BOUNDARIES[0], Z_BOUNDARIES[1], SIGMA_LAYERS[1], 1),  # 储层
        (Z_BOUNDARIES[1], z_bot, SIGMA_LAYERS[2], 2),   # 基底
    ]
    for z0, z1, s, idx in bands:
        ax.add_patch(Rectangle((0, z0), 12, z1 - z0,
                               facecolor=LAYER_COLORS[idx],
                               edgecolor="k", lw=1.0, alpha=0.90, zorder=1))
        ax.text(7.5, (z0 + z1) / 2,
                f"{LAYER_NAMES[LANG][idx]}\n"
                rf"$\sigma={s:.2f}$ S/m"
                f"  ($\\rho={1/s:.1f}$ Ω·m)",
                ha="center", va="center", fontsize=10, zorder=3,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.82,
                          ec="0.5", lw=0.8))
    # 层界面
    for zb in Z_BOUNDARIES:
        ax.axhline(zb, color="k", lw=1.5, zorder=2)

    # 垂直井眼轨迹（沿 r=0 的竖直线）
    ax.plot([1.5, 1.5], [z_top, z_bot], color="crimson", lw=2.5, zorder=4)
    ax.text(1.0, z_top + 0.6, T("垂直井", "Vertical well"),
            fontsize=9, color="crimson", fontstyle="italic")

    # Tx 在储层中心 z=0，Rx 沿井轴向下（轴向偏移）
    ax.add_patch(Ellipse((1.5, 0.0), 0.35, 0.7, facecolor="crimson",
                         edgecolor="k", lw=1.0, zorder=6))
    ax.text(1.9, 0.0, "Tx", fontsize=8, color="crimson", fontweight="bold",
            va="center")
    for j, sp in enumerate(SPACINGS):
        ax.add_patch(Ellipse((1.5, sp), 0.28, 0.55, facecolor="tab:blue",
                             edgecolor="k", lw=0.8, zorder=6))
        ax.text(1.9, sp, f"Rx{j+1}", fontsize=7, color="tab:blue", va="center")

    # 轴向收发距标注
    ax.annotate("", xy=(3.2, 0.0), xytext=(3.2, SPACINGS[-1]),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="gray"))
    ax.text(3.4, SPACINGS[-1] / 2,
            T(f"轴向收发距 1–{SPACINGS[-1]:.0f} m",
              f"Axial offset 1-{SPACINGS[-1]:.0f} m"),
            fontsize=8, color="gray", va="center")

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(z_bot, z_top)      # z 向下为正（深度习惯）
    ax.set_xticks([])
    ax.set_ylabel(T("深度 z (m)", "Depth z (m)"), fontsize=11)
    ax.set_title(T("图 01  地质概念模型 — 垂直井穿三层水平地层（2D 轴对称）",
                   "Fig. 01  Conceptual model - Vertical well through 3 horizontal "
                   "layers (2D axisymmetric)"),
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v01_concept_model.png", dpi=200)


# ================================================================ #
# 图 02：计算网格（r-z 轴对称截面）与趋肤深度校核
# ================================================================ #

def fig_02_mesh(fwd, mesh):
    cc = np.asarray(mesh.cell_centers)
    sig = true_sigma(mesh)
    ls = np.log10(np.maximum(sig, 1e-12))

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   gridspec_kw={"width_ratios": [1.1, 1.0]},
                                   figsize=(12, 5.5))

    # 左图：r-z 截面网格 + 电导率着色
    ax1.tripcolor(cc[:, 0], cc[:, 2], ls, cmap="viridis", shading="flat")
    ax1.set_aspect("auto")
    ax1.axhline(Z_BOUNDARIES[0], color="w", lw=1.0, ls="--")
    ax1.axhline(Z_BOUNDARIES[1], color="w", lw=1.0, ls="--")
    ax1.plot(0, 0, "o", ms=8, mfc="crimson", mec="k", zorder=6,
             label=T("源 Tx", "Tx"))
    for sp in SPACINGS:
        ax1.plot(0, sp, "s", ms=5, mfc="tab:blue", mec="k", zorder=6,
                 label=T("接收器 Rx", "Rx")) if sp == SPACINGS[0] else None
    ax1.set(xlabel="r (m)", ylabel=T("深度 z (m)", "Depth z (m)"),
            title=T("(a) 轴对称 r-z 计算网格（$\\log_{10}\\sigma$）",
                    "(a) Axisymmetric r-z mesh ($\\log_{10}\\sigma$)"))
    ax1.legend(loc="lower right", fontsize=8)

    # 右图：趋肤深度校核
    freqs_plot = np.logspace(np.log10(FREQUENCIES[0]) - 0.5,
                             np.log10(FREQUENCIES[-1]) + 0.5, 60)
    names = (["上覆", "储层", "基底"] if LANG == "zh"
             else ["overburden", "reservoir", "basement"])
    colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(SIGMA_LAYERS)))
    for s, col, name in zip(SIGMA_LAYERS, colors_f, names):
        deltas = [skin_depth(f, s) for f in freqs_plot]
        ax2.semilogx(freqs_plot, deltas, lw=2.0, color=col,
                     label=f"{name} {s:.2g} S/m")
    for f in FREQUENCIES:
        for s, col in zip(SIGMA_LAYERS, colors_f):
            ax2.plot(f, skin_depth(f, s), "o", ms=5, color=col, zorder=5)

    report = mesh_quality_report(mesh, FREQUENCIES, SIGMA_LAYERS)
    ok_str = "OK" if report["ok"] else "NG"
    crit = T("判据(>=3)", "criterion (>=3)")
    ax2.text(0.02, 0.97,
             f"cells = {mesh.n_cells}\n"
             f"max cell = {report['max_cell_size']:.2f} m\n"
             f"min $\\delta$ = {report['min_skin_depth']:.2f} m\n"
             f"cells/$\\delta$ = {report['cells_per_skin_depth']:.1f}\n"
             f"{crit}: {ok_str}",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85))
    ax2.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
            ylabel=T("趋肤深度 δ (m)", "Skin depth $\\delta$ (m)"),
            title=T("(b) 趋肤深度与网格分辨率校核",
                    "(b) Skin depth vs mesh resolution check"))
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle(T("图 02  轴对称计算网格与趋肤深度校核",
                   "Fig. 02  Axisymmetric mesh & skin-depth check"),
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v02_mesh.png", dpi=200)


# ================================================================ #
# 图 03：观测系统（轴向接收器阵列 + 数据强度热图）
# ================================================================ #

def fig_03_survey(fwd):
    n_f = len(FREQUENCIES)
    n_s = len(SPACINGS)
    n_c = len(COMPONENTS)

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   gridspec_kw={"width_ratios": [1.0, 1.2]},
                                   figsize=(12, 5))

    # 左图：井轴上的 Tx/Rx 布局（竖直）
    ax1.set_facecolor("#f0f8ff")
    ax1.plot([0, 0], [-1, SPACINGS[-1] + 1], color="0.4", lw=1.5, zorder=2)
    ax1.add_patch(Ellipse((0, 0), 0.5, 0.9, facecolor="crimson",
                          edgecolor="k", lw=1.0, zorder=5))
    ax1.text(0.4, 0, "Tx", fontsize=9, color="crimson", fontweight="bold",
             va="center")
    for j, sp in enumerate(SPACINGS):
        ax1.add_patch(Ellipse((0, sp), 0.42, 0.75, facecolor="tab:blue",
                              edgecolor="k", lw=0.9, zorder=5))
        ax1.text(0.4, sp, f"Rx{j+1} ({sp:.0f} m)", fontsize=7.5,
                 color="tab:blue", va="center")
    ax1.set_xlim(-1.2, 3.2)
    ax1.set_ylim(SPACINGS[-1] + 1.5, -1.5)
    ax1.set_xticks([])
    ax1.set_ylabel(T("沿井轴深度 (m)", "Along-axis depth (m)"), fontsize=10)
    ax1.set_title(T("(a) 垂直井轴向线圈布局", "(a) Axial coil layout"),
                  fontsize=11, fontweight="bold")

    # 右图：|Hz| 强度热图（行=频率，列=轴向收发距）
    d_demo = fwd.dpred(true_sigma(fwd.mesh))
    mat = fwd.to_matrix(d_demo)
    amp = np.abs(mat[:, :, 0])
    im = ax2.imshow(np.log10(amp), aspect="auto", cmap="YlOrRd", origin="upper")
    ax2.set_xticks(np.arange(n_s))
    ax2.set_xticklabels([f"{s:.0f}" for s in SPACINGS])
    ax2.set_yticks(np.arange(n_f))
    ax2.set_yticklabels([f"{f/1e3:.0f}k" for f in FREQUENCIES])
    ax2.set_xlabel(T("轴向收发距 (m)", "Axial spacing (m)"), fontsize=10)
    ax2.set_ylabel(T("频率 (Hz)", "Frequency (Hz)"), fontsize=10)
    ax2.set_title(T("(b) 实部 $\\log_{10}|B_z^{\\mathrm{real}}|$ 强度\n"
                    "（亮=高幅值，暗=低幅值）",
                    "(b) In-phase $\\log_{10}|B_z^{\\mathrm{real}}|$ intensity\n"
                    "(bright=high, dark=low)"),
                  fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, pad=0.04)
    cbar.set_label(r"$\log_{10}|B_z^{\mathrm{real}}|$ (T)", fontsize=9)

    fig.suptitle(T(f"图 03  观测系统配置（{n_f}频×{n_s}距×{n_c}分量 "
                   f"= {n_f*n_s*n_c} 道）",
                   f"Fig. 03  Survey configuration "
                   f"({n_f}f x {n_s}s x {n_c}comp = {n_f*n_s*n_c} channels)"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v03_survey.png", dpi=200)


# ================================================================ #
# 图 04：正演响应曲线（|Hz| 随轴向收发距，多频率）
# ================================================================ #

def fig_04_forward_response(fwd):
    sig = true_sigma(fwd.mesh)
    d_true = fwd.dpred(sig)
    mat = fwd.to_matrix(d_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(FREQUENCIES)))

    # (a) 实部 / 虚部 vs 轴向收发距
    ax = axes[0]
    for i, (f, col) in enumerate(zip(FREQUENCIES, colors_f)):
        ax.semilogy(SPACINGS, np.abs(mat[i, :, 0]), "o-", ms=6, lw=1.8,
                    color=col, label=f"{f/1e3:.0f} kHz Re")
        ax.semilogy(SPACINGS, np.abs(mat[i, :, 1]), "s--", ms=5, lw=1.4,
                    color=col, alpha=0.7, label=f"{f/1e3:.0f} kHz Im")
    ax.axvline(Z_BOUNDARIES[1], color="gray", ls=":", lw=1.2)
    ax.text(Z_BOUNDARIES[1] + 0.05, ax.get_ylim()[0] * 1.3,
            T("储层/基底界面", "reservoir/basement"), fontsize=7.5, color="gray")
    ax.set(xlabel=T("轴向收发距 (m)", "Axial spacing (m)"),
           ylabel=r"$|B_z^{\mathrm{s}}|$ (T)",
           title=T("(a) 二次场幅值 — 轴向收发距",
                   "(a) $|B_z^s|$ vs axial spacing"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    # (b) 相位 vs 轴向收发距
    ax = axes[1]
    for i, (f, col) in enumerate(zip(FREQUENCIES, colors_f)):
        phase = np.degrees(np.unwrap(np.arctan2(mat[i, :, 1], mat[i, :, 0])))
        ax.plot(SPACINGS, phase, "o-", ms=6, lw=1.8, color=col,
                label=f"{f/1e3:.0f} kHz")
    ax.axvline(Z_BOUNDARIES[1], color="gray", ls=":", lw=1.2)
    ax.set(xlabel=T("轴向收发距 (m)", "Axial spacing (m)"),
           ylabel=T("相位 (°)", "Phase ($^\\circ$)"),
           title=T("(b) 相位 — 轴向收发距", "(b) Phase vs axial spacing"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(T("图 04  垂直井正演响应（三层模型，清洁数据）",
                   "Fig. 04  Vertical-well forward response (3-layer, clean)"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v04_forward_response.png", dpi=200)


# ================================================================ #
# 图 05：合成含噪观测数据
# ================================================================ #

def fig_05_synthetic_data(fwd):
    sig = true_sigma(fwd.mesh)
    d_clean = fwd.dpred(sig)
    dobs, std = add_synthetic_noise(d_clean, noise_level=NOISE_LEVEL, seed=SEED)
    mat_c = fwd.to_matrix(d_clean)
    mat_n = fwd.to_matrix(dobs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) 清洁 vs 含噪（取最低频实部）
    ax = axes[0]
    ax.semilogy(SPACINGS, np.abs(mat_c[0, :, 0]), "o-", ms=6, lw=2.0,
                color="steelblue", label=T("清洁数据", "clean"))
    ax.semilogy(SPACINGS, np.abs(mat_n[0, :, 0]), "s--", ms=6, lw=1.8,
                color="crimson", alpha=0.8,
                label=T(f"含 {NOISE_LEVEL*100:.0f}% 噪声",
                        f"observed +{NOISE_LEVEL*100:.0f}% noise"))
    ax.set(xlabel=T("轴向收发距 (m)", "Axial spacing (m)"),
           ylabel=r"$|\mathrm{Re}(B_z^s)|$ (T)",
           title=T(f"(a) 最低频（{FREQUENCIES[0]/1e3:.0f} kHz）清洁 vs 含噪",
                   f"(a) Lowest freq ({FREQUENCIES[0]/1e3:.0f} kHz) clean vs noisy"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # (b) 残差分布
    ax = axes[1]
    residual = (dobs - d_clean) / d_clean
    ax.hist(residual * 100, bins=20, color="steelblue", alpha=0.78, edgecolor="k")
    ax.axvline(-NOISE_LEVEL * 100, color="crimson", ls="--", lw=1.5,
               label=T(f"±{NOISE_LEVEL*100:.0f}% 噪声带",
                       f"±{NOISE_LEVEL*100:.0f}% band"))
    ax.axvline(NOISE_LEVEL * 100, color="crimson", ls="--", lw=1.5)
    ax.axvline(0, color="k", lw=1.2)
    ax.set(xlabel=T("相对残差 (%)", "Relative residual (%)"),
           ylabel=T("频数", "Count"),
           title=T(f"(b) 噪声分布（seed={SEED}, nD={dobs.size}）",
                   f"(b) Noise distribution (seed={SEED}, nD={dobs.size})"))
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(T(f"图 05  合成观测数据（噪声水平 {NOISE_LEVEL*100:.0f}%）",
                   f"Fig. 05  Synthetic data ({NOISE_LEVEL*100:.0f}% noise)"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v05_synthetic_data.png", dpi=200)
    return dobs, std


# ================================================================ #
# 图 06：Occam 反演回收（r-z 截面，复用内置图版）
# ================================================================ #

def fig_06_inversion(fwd, dobs, std):
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)
    m0 = np.log(M0_SIGMA * np.ones(fwd.mesh.n_cells))

    inv = LWDFDEMInversion(
        fwd, data_obj, m0=m0,
        max_iter=MAX_ITER,
        alpha_s=1e-3, alpha_x=1.0, alpha_z=1.0,
        beta0_ratio=1.0, beta_cooling=4.0, chifact=1.0,
        sensitivity_weights=True, preconditioner=True,
        optimizer="gn",
    )
    rec_sigma, history = inv.run()

    sig_true = true_sigma(fwd.mesh)
    print("\n=== 反演结果 ===")
    print(f"迭代次数: {len(history['iterations'])}")
    print(f"最终 phi_d/nD: {history['final_phi_d']/dobs.size:.3f}")
    for idx, name in enumerate(SIGMA_LAYERS):
        print(f"  层{idx+1} 真实 {name:.3f} S/m")

    # 复用内置反演图版（真实/回收 r-z 截面 + 收敛史），已支持双语
    plate = plot_inversion_figure(
        fwd, rec_sigma, history, sigma_true=sig_true,
        out_dir=FIG_DIR, prefix="v06_inversion", lang=LANG)
    print(f"[反演图版] {plate}")
    return rec_sigma, history


# ================================================================ #
# 图 07：收敛历史与 β 调度诊断
# ================================================================ #

def fig_07_convergence(history, dobs):
    it = history["iterations"]
    x = list(range(1, len(it) + 1))
    phi_d = [d["phi_d"] for d in it]
    phi_m = [d["phi_m"] for d in it]
    beta = [d["beta"] for d in it]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) φ_d 收敛
    ax = axes[0]
    ax.semilogy(x, phi_d, "o-", ms=6, lw=2.0, color="C0", label=r"$\phi_d$")
    target = len(dobs)
    ax.axhline(target, color="crimson", ls="--", lw=1.5,
               label=T(f"目标 = nD = {target}", f"target = nD = {target}"))
    ax.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\phi_d$",
           title=T("(a) 数据拟合差收敛", "(a) Data-misfit convergence"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)

    # (b) β 冷却 + φ_m
    ax = axes[1]
    ax.loglog(x, beta, "s-", ms=6, lw=2.0, color="darkorange",
              label=r"$\beta$")
    axb = ax.twinx()
    axb.semilogy(x, phi_m, "^-", ms=5, lw=1.8, color="seagreen",
                 label=r"$\phi_m$")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8, loc="center right")
    ax.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\beta$",
           title=T("(b) β 冷却与正则化演化", "(b) $\\beta$ cooling & regularization"))
    axb.set_ylabel(r"$\phi_m$")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(T("图 07  反演收敛历史与质量诊断",
                   "Fig. 07  Convergence history & diagnostics"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "v07_convergence.png", dpi=200)


# ================================================================ #
# 主流程
# ================================================================ #

def main():
    # 按图版语言设置字体（复用公共设施）
    apply_rcparams(LANG)

    print("=" * 60)
    print("  垂直井三层地层 LWD-FDEM 正反演（2D 轴对称 · 轴向偏移）")
    print("=" * 60)
    print(f"层界 z = {Z_BOUNDARIES.tolist()} m, "
          f"σ = {SIGMA_LAYERS.tolist()} S/m")
    print(f"观测: {len(FREQUENCIES)}频 × {len(SPACINGS)}距 × {len(COMPONENTS)}分量"
          f" = {len(FREQUENCIES)*len(SPACINGS)*len(COMPONENTS)} 道")

    fwd, cfg, mesh = build_forward()
    print(f"网格: {mesh.n_cells} 单元 (2D 轴对称 r-z 截面)")

    print("\n[图 01] 地质概念模型 ...")
    fig_01_concept_model()
    print("[图 02] 计算网格与趋肤深度 ...")
    fig_02_mesh(fwd, mesh)
    print("[图 03] 观测系统配置 ...")
    fig_03_survey(fwd)
    print("[图 04] 正演响应曲线 ...")
    fig_04_forward_response(fwd)
    print("[图 05] 合成含噪观测数据 ...")
    dobs, std = fig_05_synthetic_data(fwd)
    print("[图 06] Occam 反演 ...")
    rec_sigma, history = fig_06_inversion(fwd, dobs, std)
    print("[图 07] 收敛诊断 ...")
    fig_07_convergence(history, dobs)

    print(f"\n全部图版已保存至: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
