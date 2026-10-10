"""水平井三层地层 LWD-FDEM 正反演完整流程。

流程顺序（与图版编号对应）：
  图 01  地质概念模型        — 地层结构与井眼轨迹示意
  图 02  计算网格            — 1D 层状离散网格与趋肤深度校核
  图 03  观测系统配置        — 仪器参数与数据排列结构可视化
  图 04  正演响应            — 多频幅相曲线（清洁模型响应）
  图 05  合成数据与噪声      — 含噪观测 vs 真实响应对比
  图 06  反演回收剖面        — Occam 反演结果
  图 07  收敛历史与 β 调度   — 反演质量诊断

运行（仓库根目录）：
    conda activate lwd
    python examples/horizontal_well_3layer.py

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
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

from src.lwd_fdem import (
    LWDFDEMForward,
    LWDFDEMInversion,
    LWDToolConfig,
    add_synthetic_noise,
    build_1d_mesh,
    make_data_object,
    mesh_quality_report,
    plot_forward_figure,
    plot_inversion_figure,
    pub_style,
    apply_rcparams,
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


# ---- 地层模型 ----
THICKNESSES  = np.array([3.0, 6.0])          # 各层厚度 (m)，半空间另计
SIGMA_TRUE   = np.array([0.20, 0.01, 0.50])  # 各层电导率 (S/m)
LAYER_NAMES  = {
    "zh": ["上覆泥岩", "砂岩储层\n(目标)", "水侵基底"],
    "en": ["Overburden\n(shale)", "Reservoir\n(sandstone)", "Basement\n(aquifer)"],
}
LAYER_COLORS = ["#b8845a", "#f5e8c8", "#6aaed6"]

# ---- 仪器配置 ----
# 频率：对数均匀 10 点，上限限为 100 kHz（准静态区）。
# 经诊断：实部与虚部的过零频率不同，虚部（正交分量）过零更早（~200 kHz）。
# 频率≤100 kHz 时，所有收发距的实部与虚部均不过零、单调上升，
# 彻底消除 |Re| 与 |Im| 曲线因穿越零点产生的 V 形尖谷（突变）。
# 10–100 kHz 亦为真实 LWD 仪器（GeoSphere/Halo 等）的标准工作频段。
FREQUENCIES  = tuple(np.logspace(np.log10(1e4), np.log10(1e5), 10))  # 10 kHz – 100 kHz
SPACINGS     = np.array([0.5, 1.0, 1.5, 2.0, 3.0])  # 收发距 (m)
COMPONENTS   = ("real", "imag")

# ---- 反演设置 ----
M0_SIGMA     = 0.1       # 初始均匀模型电导率 (S/m)
NOISE_LEVEL  = 0.02      # 相对噪声水平 (2%)
SEED         = 42
MAX_ITER     = 20
BETA_COOLING = 3.0
CHIFACT      = 1.0

# 按图版语言设置字体（英文 serif/Times；中文 SimHei + stix 数学），复用公共工具
apply_rcparams(LANG)


def build_forward():
    """构建正演计算器（步骤 0 共用）。"""
    cfg = LWDToolConfig(
        frequencies=FREQUENCIES,
        spacings=SPACINGS,
        components=COMPONENTS,
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
        moment=1.0, source_type="mag_dipole",
    )
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=THICKNESSES)
    return fwd, cfg


# ================================================================ #
# 图 01：地质概念模型
# ================================================================ #

def fig_01_concept_model():
    """绘制水平井穿越三层地层的地质概念模型。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    depth = 0.0
    total_h = list(THICKNESSES) + [THICKNESSES[-1] * 1.5]
    max_depth = sum(total_h)

    # 地层矩形
    for i, s in enumerate(SIGMA_TRUE):
        h = total_h[i]
        ax.add_patch(Rectangle((0, depth), 12, h,
                               facecolor=LAYER_COLORS[i],
                               edgecolor="k", lw=1.0, alpha=0.90, zorder=1))
        ax.text(6.5, depth + h / 2,
                f"{LAYER_NAMES[LANG][i]}\n"
                rf"$\sigma_{i+1} = {s:.2f}$ S/m"
                f"   ($\\rho = {1/s:.1f}$ Ω·m)",
                ha="center", va="center", fontsize=10, zorder=3,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.80,
                          ec="0.5", lw=0.8))
        # 层厚标注
        if i < len(THICKNESSES):
            ax.annotate("", xy=(12.6, depth), xytext=(12.6, depth + h),
                        arrowprops=dict(arrowstyle="<->", lw=1.3, color="darkred"))
            ax.text(12.85, depth + h / 2, f"{h:.0f} m",
                    va="center", fontsize=9, color="darkred", fontweight="bold")
        depth += h

    # 层界面线
    depth = 0.0
    for h in THICKNESSES:
        depth += h
        ax.axhline(depth, color="k", lw=1.5, xmin=0, xmax=1, zorder=2)
    ax.axhline(0, color="steelblue", lw=2.0, zorder=2)
    ax.text(-0.3, 0, T("地表", "Surface"), ha="right", va="center",
            fontsize=9, color="steelblue")

    # 水平井眼轨迹（穿过目标层中部）
    well_z = THICKNESSES[0] + THICKNESSES[1] / 2.0
    ax.annotate("", xy=(11.5, well_z), xytext=(0.5, well_z),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="crimson",
                                connectionstyle="arc3,rad=0"))
    ax.text(1.0, well_z - 0.5, T("水平井", "Horizontal well"),
            fontsize=9, color="crimson", fontstyle="italic")

    # Tx / Rx 线圈符号
    tx_x = 5.0
    rx_offsets = [0.5, 1.0, 1.5, 2.0, 2.5]
    ax.add_patch(Ellipse((tx_x, well_z), 0.12, 0.55,
                         facecolor="crimson", edgecolor="k", lw=1.0, zorder=6))
    ax.text(tx_x, well_z + 0.55, "Tx", ha="center", va="bottom",
            fontsize=8, color="crimson", fontweight="bold")
    for off in rx_offsets:
        rx_x = tx_x + off
        ax.add_patch(Ellipse((rx_x, well_z), 0.12, 0.55,
                             facecolor="tab:blue", edgecolor="k", lw=0.8, zorder=6))
    # 收发距标注
    ax.annotate("", xy=(tx_x, well_z - 1.1), xytext=(tx_x + rx_offsets[-1], well_z - 1.1),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="gray"))
    ax.text(tx_x + rx_offsets[-1] / 2, well_z - 1.45,
            T(f"接收器阵列  0.5–{rx_offsets[-1]:.1f} m",
              f"Rx array  0.5-{rx_offsets[-1]:.1f} m"),
            ha="center", fontsize=8, color="gray")

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(max_depth + 0.5, -1.5)
    ax.set_xticks([])
    ax.set_ylabel(T("深度 (m)", "Depth (m)"), fontsize=11)
    ax.set_title(T("图 01  地质概念模型 — 水平井三层地层 LWD-FDEM 观测场景",
                   "Fig. 01  Conceptual model - Horizontal-well 3-layer LWD-FDEM survey"),
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "01_concept_model.png")


# ================================================================ #
# 图 02：计算网格与趋肤深度校核
# ================================================================ #

def fig_02_mesh(sigma_model=None):
    """绘制 1D 层状离散网格，并校核趋肤深度。"""
    if sigma_model is None:
        sigma_model = SIGMA_TRUE
    mesh = build_1d_mesh(THICKNESSES)
    n_cells = mesh.n_cells
    # 1D TensorMesh: h 返回 tuple of arrays; cell_centers 返回 (n_cells,)
    cell_widths = np.asarray(mesh.h[0])           # (n_cells,) 每层单元宽度
    cell_centers = np.asarray(mesh.cell_centers).ravel()  # (n_cells,) 中心深度
    zb = np.r_[0.0, np.cumsum(cell_widths)]       # 单元底界深度 (n_cells+1,)

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   gridspec_kw={"width_ratios": [1.1, 1.0]},
                                   figsize=(11, 5.5))

    # ---- 左图：网格单元 ----
    ls = np.log10(sigma_model)
    norm = (ls - ls.min()) / (np.ptp(ls) + 1e-12)
    for i in range(n_cells):
        h = cell_widths[i]
        ax1.add_patch(Rectangle((0, zb[i]), 4, h,
                                facecolor=plt.cm.viridis(norm[i]),
                                edgecolor="k", lw=1.2, alpha=0.90))
        label = (f"Layer {i+1}" if i < len(THICKNESSES)
                 else "Half-space\n(approx.)")
        ax1.text(2.0, zb[i] + h / 2,
                 f"Cell {i+1}\n{label}\n"
                 rf"$h_i={h:.1f}$ m" "\n"
                 rf"$\log_{{10}}\sigma={ls[i]:.2f}$",
                 ha="center", va="center", fontsize=8,
                 color="w" if norm[i] < 0.5 else "k")
    # 层界面
    for zi in zb:
        ax1.axhline(zi, color="k", lw=1.5, xmin=0, xmax=4/4.5, zorder=3)
    ax1.set_xlim(-0.5, 4.5); ax1.set_xticks([])
    ax1.set_ylim(zb[-1] + 1, -1)
    ax1.set_ylabel(T("深度 (m)", "Depth (m)"), fontsize=10)
    ax1.set_title(T("(a) 1D 张量网格离散", "(a) 1D tensor mesh discretization"),
                  fontsize=10, fontweight="bold")

    # ---- 右图：趋肤深度校核 ----
    freqs_plot = np.logspace(np.log10(FREQUENCIES[0]) - 0.5,
                             np.log10(FREQUENCIES[-1]) + 0.5, 80)
    colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(sigma_model)))
    names = (["上覆 s1", "储层 s2", "基底 s3"] if LANG == "zh"
             else ["overburden", "reservoir", "basement"])
    for s, col, name in zip(sigma_model, colors_f, names):
        deltas = [skin_depth(f, s) for f in freqs_plot]
        ax2.semilogx(freqs_plot, deltas, lw=2.0, color=col, label=f"{name}={s:.2g} S/m")

    # 标记各工作频率的趋肤深度
    for f in FREQUENCIES:
        for s, col in zip(sigma_model, colors_f):
            ax2.plot(f, skin_depth(f, s), "o", ms=5, color=col, zorder=5)

    # 画参考线：储层在最低频的趋肤深度
    delta_ref = skin_depth(FREQUENCIES[0], sigma_model[1])
    ax2.axhline(delta_ref, color="gray", ls="--", lw=1.2)
    ax2.text(freqs_plot[0], delta_ref * 1.15,
             rf"$\delta(\sigma_2, f_{{min}})={delta_ref:.1f}$ m",
             fontsize=8, color="gray")

    # 网格质量报告
    report = mesh_quality_report(mesh, FREQUENCIES, sigma_model)
    ok_str = "OK" if report["ok"] else "NG"
    crit = T("判据(>=3)", "criterion (>=3)")
    ax2.text(0.02, 0.97,
             f"cells/δ  = {report['cells_per_skin_depth']:.1f}\n"
             f"max cell = {report['max_cell_size']:.2f} m\n"
             f"min δ    = {report['min_skin_depth']:.2f} m\n"
             f"{crit}: {ok_str}",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85))

    ax2.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
            ylabel=T("趋肤深度 δ (m)", "Skin depth $\\delta$ (m)"),
            title=T("(b) 趋肤深度与网格分辨率校核",
                    "(b) Skin depth vs mesh resolution check"))
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle(T("图 02  计算网格构建与趋肤深度校核",
                   "Fig. 02  Mesh construction & skin-depth check"),
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "02_mesh.png")


# ================================================================ #
# 图 03：观测系统配置
# ================================================================ #

def fig_03_survey(cfg, fwd):
    """可视化观测系统：仪器布局 + 数据排列结构热图。"""
    n_f = len(FREQUENCIES)
    n_s = len(SPACINGS)
    n_c = len(COMPONENTS)

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   gridspec_kw={"width_ratios": [1.35, 1.0]},
                                   figsize=(12, 5))

    # ---- 左图：仪器布局（Tx + Rx 阵列沿径向，频率用图例）----
    ax1.set_facecolor("#f0f8ff")
    ax1.add_patch(Ellipse((0, 0), 0.16, 0.30, facecolor="crimson",
                          edgecolor="k", lw=1.0, zorder=5))
    ax1.text(0, 0.24, "Tx", ha="center", fontsize=9, color="crimson",
             fontweight="bold")
    for j, sp in enumerate(SPACINGS):
        ax1.add_patch(Ellipse((sp, 0), 0.13, 0.26, facecolor="tab:blue",
                              edgecolor="k", lw=0.9, zorder=5))
        ax1.text(sp, 0.24, f"Rx{j+1}", ha="center", fontsize=8, color="tab:blue")
        # 收发距双向箭头：交替上下错开，避免文字重叠
        y = -0.34 - 0.17 * (j % 2)
        ax1.annotate("", xy=(0, y), xytext=(sp, y),
                     arrowprops=dict(arrowstyle="<->", lw=0.8, color="gray"))
        ax1.text(sp / 2, y, f"{sp:.1f} m", ha="center", va="center",
                 fontsize=7.5, color="gray",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#f0f8ff", ec="none"))

    # 频率图例（彩色圆点，两列缩短），置于顶部中央，不遮挡 Rx5
    freq_colors = plt.cm.plasma(np.linspace(0.05, 0.95, n_f))
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=freq_colors[i],
                      mec=freq_colors[i],
                      label=f"f{i+1}={FREQUENCIES[i]/1e3:.1f}k")
               for i in range(n_f)]
    ax1.legend(handles=handles, loc="upper center", fontsize=7.5, ncol=2,
               columnspacing=0.9, handletextpad=0.3,
               title=T("工作频率 (Hz)", "Frequency (Hz)"),
               title_fontsize=8, framealpha=0.92)

    ax1.set_xlim(-0.4, SPACINGS[-1] + 0.4)
    ax1.set_ylim(-1.05, 1.75)
    ax1.set_xlabel(T("径向收发距 (m)", "Radial offset (m)"), fontsize=10)
    ax1.set_yticks([])
    ax1.set_title(T("(a) 仪器空间布局", "(a) Instrument layout"),
                  fontsize=11, fontweight="bold")

    # ---- 右图：数据排列结构热图（整数单元坐标，刻度与色块严格对齐）----
    d_demo = fwd.dpred(SIGMA_TRUE)
    mat = fwd.to_matrix(d_demo)
    amp_matrix = np.abs(mat[:, :, 0])   # (n_f, n_s)

    im = ax2.imshow(np.log10(amp_matrix), aspect="auto", cmap="YlOrRd",
                    origin="upper")
    ax2.set_xticks(np.arange(n_s))
    ax2.set_xticklabels([f"{s:.1f}" for s in SPACINGS])
    ax2.set_yticks(np.arange(n_f))
    ax2.set_yticklabels([f"{f/1e3:.0f}k" for f in FREQUENCIES])
    ax2.set_xlabel(T("收发距 (m)", "Spacing (m)"), fontsize=10)
    ax2.set_ylabel(T("频率 (Hz)", "Frequency (Hz)"), fontsize=10)
    ax2.set_title(T("(b) 实部响应 $\\log_{10}|H_z^{\\mathrm{real}}|$ 强度图\n（亮=高幅值，暗=低幅值）",
                    "(b) In-phase $\\log_{10}|H_z^{\\mathrm{real}}|$ intensity\n(bright=high, dark=low)"),
                  fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, pad=0.04)
    cbar.set_label(r"$\log_{10}|H_z^{\mathrm{real}}|$ (A/m)", fontsize=9)

    fig.suptitle(T(f"图 03  观测系统配置与数据排列结构（{n_f}频×{n_s}距×{n_c}分量 = {n_f*n_s*n_c} 道）",
                   f"Fig. 03  Survey configuration & data layout ({n_f}f x {n_s}s x {n_c}comp = {n_f*n_s*n_c} channels)"),
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "03_survey.png")


# ================================================================ #
# 图 04：正演响应曲线
# ================================================================ #

def fig_04_forward_response(fwd, cfg):
    """多频幅相曲线（清洁模型，无反演）。

    修复说明：
      - 图(a)(b) 改用 log-log 坐标，避免线性坐标下指数衰减表现为悬崖
      - 图(d) 相位用 np.unwrap 展开，消除 arctan2 在 -180° 处的卷绕跳变
    """
    d_true = fwd.dpred(SIGMA_TRUE)
    mat = fwd.to_matrix(d_true)
    freqs = np.array(FREQUENCIES, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    markers = ["o", "s", "^", "D", "v"]
    colors_sp = plt.cm.Set1(np.linspace(0, 0.9, len(SPACINGS)))

    # (a) 实部绝对值 vs 频率（log-log，消除悬崖假象）
    ax = axes[0, 0]
    for j, (sp, mk, col) in enumerate(zip(SPACINGS, markers, colors_sp)):
        ax.loglog(freqs, np.abs(mat[:, j, 0]) + 1e-20,
                  marker=mk, ms=5, lw=1.8, color=col, label=f"$s$={sp:.1f} m")
    ax.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"), ylabel=r"$|\mathrm{Re}(H_z^s)|$ (A/m)",
           title=T("(a) 实部幅值 — 对数坐标（10频点）",
                   "(a) In-phase amplitude - log-log (10 freq)"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)

    # (b) 虚部绝对值 vs 频率
    ax = axes[0, 1]
    for j, (sp, mk, col) in enumerate(zip(SPACINGS, markers, colors_sp)):
        ax.loglog(freqs, np.abs(mat[:, j, 1]) + 1e-20,
                  marker=mk, ms=5, lw=1.8, color=col, label=f"$s$={sp:.1f} m")
    ax.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"), ylabel=r"$|\mathrm{Im}(H_z^s)|$ (A/m)",
           title=T("(b) 虚部幅值 — 对数坐标",
                   "(b) Quadrature amplitude - log-log"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)

    # (c) 总幅值 vs 收发距
    ax = axes[1, 0]
    colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    for i, (f, col) in enumerate(zip(freqs, colors_f)):
        amp = np.hypot(mat[i, :, 0], mat[i, :, 1])
        ax.loglog(SPACINGS, amp, marker=markers[i % len(markers)],
                  ms=5, lw=1.6, color=col, alpha=0.85,
                  label=f"{f/1e3:.0f} kHz")
    ax.set(xlabel=T("收发距 (m)", "Spacing (m)"), ylabel=r"$|H_z^s|$ (A/m)",
           title=T("(c) 总幅值 — 收发距特性", "(c) Total amplitude vs spacing"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5, ncol=2)

    # (d) 相位 vs 频率（np.unwrap 消除卷绕）
    ax = axes[1, 1]
    for j, (sp, mk, col) in enumerate(zip(SPACINGS, markers, colors_sp)):
        phase_rad = np.unwrap(np.arctan2(mat[:, j, 1], mat[:, j, 0]))
        phase = np.degrees(phase_rad)
        ax.semilogx(freqs, phase, marker=mk, ms=5, lw=1.8, color=col,
                    label=f"$s$={sp:.1f} m")
    ax.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
           ylabel=T("相位 (°，展开)", "Phase ($^\\circ$, unwrapped)"),
           title=T("(d) 相位 — np.unwrap 消除卷绕",
                   "(d) Phase - np.unwrap (no wrapping)"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)

    fig.suptitle(T("图 04  正演响应曲线（三层模型，清洁数据）",
                   "Fig. 04  Forward response (3-layer model, clean data)"),
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "04_forward_response.png")


# ================================================================ #
# 图 05：合成观测数据与噪声
# ================================================================ #

def fig_05_synthetic_data(fwd, cfg):
    """生成含噪合成数据，展示噪声与观测误差。"""
    d_clean = fwd.dpred(SIGMA_TRUE)
    dobs, std = add_synthetic_noise(d_clean, noise_level=NOISE_LEVEL, seed=SEED)
    mat_c = fwd.to_matrix(d_clean)
    mat_n = fwd.to_matrix(dobs)
    freqs = np.array(FREQUENCIES)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # (a) 实部清洁 vs 含噪（取第3个收发距 = 1.5 m）
    ax = axes[0, 0]
    ax.semilogx(freqs, mat_c[:, 2, 0], "o-", ms=6, lw=2.0, color="steelblue",
                label=T("清洁数据", "clean"))
    ax.semilogx(freqs, mat_n[:, 2, 0], "s--", ms=6, lw=1.8, color="crimson",
                alpha=0.8, label=T(f"含 {NOISE_LEVEL*100:.0f}% 噪声观测",
                                   f"observed +{NOISE_LEVEL*100:.0f}% noise"))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"), ylabel=r"$\mathrm{Re}(H_z^s)$ (A/m)",
           title=T(f"(a) 实部：清洁 vs 含噪（$s$=1.5 m）",
                   f"(a) In-phase: clean vs noisy ($s$=1.5 m)"))
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    # (b) 噪声分布直方图
    ax = axes[0, 1]
    noise = dobs - d_clean
    ax.hist(noise / d_clean * 100, bins=20, color="steelblue", alpha=0.75,
            edgecolor="k", density=False)
    ax.axvline(-NOISE_LEVEL * 100, color="crimson", lw=1.5, ls="--",
               label=T(f"±{NOISE_LEVEL*100:.0f}% 噪声带",
                       f"±{NOISE_LEVEL*100:.0f}% noise band"))
    ax.axvline(NOISE_LEVEL * 100, color="crimson", lw=1.5, ls="--")
    ax.set(xlabel=T("相对噪声 (%)", "Relative noise (%)"), ylabel=T("频次", "Count"),
           title=T(f"(b) 噪声分布（seed={SEED}）", f"(b) Noise distribution (seed={SEED})"))
    ax.legend(); ax.grid(alpha=0.3)

    # (c) 绝对误差棒图（各收发距，实部）
    ax = axes[1, 0]
    abs_err = np.abs(noise).reshape(len(freqs), len(SPACINGS), 2)
    for i, (f, col) in enumerate(zip(freqs, plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs))))):
        ax.errorbar(SPACINGS, np.abs(mat_n[i, :, 0]),
                    yerr=abs_err[i, :, 0], fmt="o-", ms=5, lw=1.5,
                    capsize=3, color=col, alpha=0.8, label=f"{f/1e3:.0f} kHz")
    ax.set(xlabel=T("收发距 (m)", "Spacing (m)"), ylabel=r"$|\mathrm{Re}(H_z^s)|$ (A/m)",
           title=T("(c) 含噪观测与误差棒", "(c) Noisy data with error bars"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)

    # (d) 数据 vs 模型散点（全局）
    ax = axes[1, 1]
    ax.scatter(d_clean / std, dobs / std, s=20, alpha=0.6,
               c=np.log10(np.abs(d_clean) + 1e-20), cmap="coolwarm",
               edgecolors="k", lw=0.3)
    lim = [min(d_clean / std) * 0.9, max(d_clean / std) * 1.1]
    ax.plot(lim, lim, "k--", lw=1.5, label=T("理想 1:1 线", "ideal 1:1"))
    ax.set(xlabel=r"$d_{\rm true}/\sigma_n$",
           ylabel=r"$d_{\rm obs}/\sigma_n$",
           title=T(f"(d) 标准化观测散点图（nD={d_clean.size}）",
                   f"(d) Normalized scatter (nD={d_clean.size})"))
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(T(f"图 05  合成观测数据（噪声水平 {NOISE_LEVEL*100:.0f}%）",
                   f"Fig. 05  Synthetic data ({NOISE_LEVEL*100:.0f}% noise)"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_synthetic_data.png")
    return dobs, std


# ================================================================ #
# 图 06：反演回收剖面
# ================================================================ #

def fig_06_inversion(fwd, dobs, std):
    """Occam 反演，输出回收剖面与真实模型对比。"""
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)
    m0 = np.log(M0_SIGMA * np.ones(len(SIGMA_TRUE)))

    inv = LWDFDEMInversion(
        fwd, data_obj, m0=m0,
        max_iter=MAX_ITER,
        alpha_s=1e-4, alpha_x=1.0,
        beta0_ratio=1e-3, beta_cooling=BETA_COOLING,
        chifact=CHIFACT,
        sensitivity_weights=False, preconditioner=False,
        optimizer="gn",
    )
    rec_sigma, history = inv.run()

    # 打印
    print("\n=== 反演结果 ===")
    print(f"真实 σ (S/m):  {SIGMA_TRUE}")
    print(f"回收 σ (S/m):  {np.round(rec_sigma, 4)}")
    rel_err = np.abs(rec_sigma - SIGMA_TRUE) / SIGMA_TRUE * 100
    print(f"相对误差 (%):  {np.round(rel_err, 2)}")

    th = THICKNESSES
    n = len(SIGMA_TRUE)
    bounds = np.r_[0.0, np.cumsum(th)]
    centers = np.array([0.5 * (bounds[i] + bounds[i + 1]) if i < len(th)
                        else bounds[-1] + 0.5 * th[-1] for i in range(n)])

    fig, axes = plt.subplots(1, 3,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]},
                             figsize=(13, 5))

    # (a) 深度剖面（线性σ）
    ax = axes[0]
    ax.step(SIGMA_TRUE, centers, where="mid", color="k", ls="--", lw=2.2,
           label=T("真实模型", "true model"), zorder=3)
    ax.step(rec_sigma, centers, where="mid", color="C0", lw=2.2, marker="o",
           ms=6, label=T("回收模型", "recovered"), zorder=4)
    for i, (yt, yb) in enumerate(zip(bounds[:-1],
                                      np.r_[bounds[1:], bounds[-1] + th[-1] * 1.5])):
        ax.axhspan(yt, yb, alpha=0.07, color=LAYER_COLORS[i])
        ax.axhline(yt, color="gray", lw=0.8, ls=":")
    ax.set(xlabel=r"$\sigma$ (S/m)", ylabel=T("深度 (m)", "Depth (m)"),
           title=T("(a) 电导率剖面", "(a) Conductivity profile"))
    ax.invert_yaxis(); ax.legend(fontsize=9)

    # (b) 对数坐标剖面
    ax = axes[1]
    ax.step(SIGMA_TRUE, centers, where="mid", color="k", ls="--", lw=2.2,
            label=T("真实模型", "true model"))
    ax.step(rec_sigma, centers, where="mid", color="C2", lw=2.2, marker="s",
            ms=6, label=T("回收模型", "recovered"))
    ax.set_xscale("log")
    ax.set(xlabel=r"$\sigma$ (S/m, log)", ylabel=T("深度 (m)", "Depth (m)"),
           title=T("(b) 对数坐标剖面", "(b) Log-scale profile"))
    ax.invert_yaxis(); ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # (c) 拟合曲线（清洁数据 vs 含噪观测 vs 反演预测）
    ax = axes[2]
    d_rec = fwd.dpred(rec_sigma)
    mat_rec = fwd.to_matrix(d_rec)
    mat_obs = fwd.to_matrix(dobs)
    mat_tru = fwd.to_matrix(fwd.dpred(SIGMA_TRUE))
    freqs = np.array(FREQUENCIES)
    for k, (c, mk) in enumerate(zip(["k", "gray", "C0"], ["o", "s", "^"])):
        ax.semilogx(freqs, np.abs(mat_tru[:, k, 0]), c=c, ls="--",
                     marker=mk, ms=5, lw=1.5,
                     label=T(f"真实 $s$={SPACINGS[k]:.1f}m", f"true $s$={SPACINGS[k]:.1f}m") if c == "k" else None)
        ax.semilogx(freqs, np.abs(mat_obs[:, k, 0]), c="gray", ls=":",
                     marker="x", ms=4, lw=1.0,
                     label=T(f"观测 $s$={SPACINGS[k]:.1f}m", f"obs $s$={SPACINGS[k]:.1f}m") if c == "gray" else None)
        ax.semilogx(freqs, np.abs(mat_rec[:, k, 0]), c=c, ls="-",
                     marker=mk, ms=5, lw=1.8,
                     label=T(f"反演 $s$={SPACINGS[k]:.1f}m", f"inv $s$={SPACINGS[k]:.1f}m") if c == "C0" else None)
    ax.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"), ylabel=r"$|H_z^s|$ (A/m)",
           title=T("(c) 响应拟合对比", "(c) Response fit comparison"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=1)

    fig.suptitle(T(f"图 06  Occam 反演回收（迭代 {len(history['iterations'])} 次，"
                   f"$\\phi_d$/nD={history['final_phi_d']/dobs.size:.2f}）",
                   f"Fig. 06  Occam inversion ({len(history['iterations'])} iters, "
                   f"$\\phi_d$/nD={history['final_phi_d']/dobs.size:.2f})"),
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "06_inversion.png")
    return rec_sigma, history


# ================================================================ #
# 图 07：收敛历史与 β 调度诊断
# ================================================================ #

def fig_07_convergence(history, dobs, fwd):
    """收敛过程诊断：φ_d、φ_m、β 及拟合差残差分布。"""
    it = history["iterations"]
    x = list(range(1, len(it) + 1))
    phi_d = [d["phi_d"] for d in it]
    phi_m = [d["phi_m"] for d in it]
    beta = [d["beta"] for d in it]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) φ_d 收敛
    ax = axes[0, 0]
    ax.semilogy(x, phi_d, "o-", ms=6, lw=2.0, color="C0", label=r"$\phi_d$")
    target = len(dobs) * CHIFACT
    ax.axhline(target, color="crimson", ls="--", lw=1.5,
               label=T(f"目标 = chifact×nD = {target:.0f}",
                       f"target = chifact x nD = {target:.0f}"))
    ax.axvline(x[-1], color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(x[-1] * 1.02, phi_d[-1] * 1.5, T(f"迭代={x[-1]}", f"iter={x[-1]}"),
            fontsize=8, color="gray")
    ax.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\phi_d$",
           title=T("(a) 数据拟合差收敛", "(a) Data-misfit convergence"))
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)

    # (b) β 冷却
    ax = axes[0, 1]
    ax.loglog(x, beta, "s-", ms=6, lw=2.0, color="darkorange")
    ax.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\beta$",
           title=T(f"(b) β 冷却过程（cooling_factor={BETA_COOLING}）",
                   f"(b) $\\beta$ cooling (factor={BETA_COOLING})"))
    ax.grid(True, which="both", alpha=0.3)
    # 在曲线上标注每次冷却的时刻
    prev_b = beta[0]
    for i, b in enumerate(beta):
        if b < prev_b * 0.5:
            ax.annotate("cool", xy=(x[i], b), xytext=(x[i], b * 2.5),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=7.5, color="gray", ha="center")
        prev_b = b

    # (c) 剪刀差 φ_d / φ_m（共享迭代轴）
    ax = axes[1, 0]
    ax.semilogy(x, phi_d, "o-", ms=5, lw=1.8, color="C0", label=T(r"$\phi_d$（拟合差）", r"$\phi_d$ (misfit)"))
    axb = ax.twinx()
    axb.semilogy(x, phi_m, "^-", ms=5, lw=1.8, color="seagreen",
                 label=T(r"$\phi_m$（正则化）", r"$\phi_m$ (reg)"))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\phi_d$",
           title=T("(c) 剪刀差收敛过程", "(c) Misfit vs regularization"))
    axb.set_ylabel(r"$\phi_m$")
    ax.grid(True, which="both", alpha=0.3)

    # (d) 最终预测残差分布
    ax = axes[1, 1]
    # 重建最终模型
    d_rec = fwd.dpred(np.exp(history["model_log"]))
    residual = (dobs - d_rec) / dobs
    ax.hist(residual * 100, bins=20, color="steelblue", alpha=0.78,
            edgecolor="k")
    ax.axvline(-NOISE_LEVEL * 100, color="crimson", ls="--", lw=1.5,
               label=T(f"±{NOISE_LEVEL*100:.0f}% 噪声", f"±{NOISE_LEVEL*100:.0f}% noise"))
    ax.axvline(NOISE_LEVEL * 100, color="crimson", ls="--", lw=1.5)
    ax.axvline(0, color="k", lw=1.2)
    rms = np.sqrt(np.mean(residual ** 2)) * 100
    ax.text(0.97, 0.93, T(f"RMS 相对残差\n= {rms:.2f}%", f"RMS rel. residual\n= {rms:.2f}%"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    ax.set(xlabel=T("相对残差 (%)", "Relative residual (%)"), ylabel=T("频数", "Count"),
           title=T(f"(d) 最终残差分布（nD={len(dobs)}）", f"(d) Final residual (nD={len(dobs)})"))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(T("图 07  反演收敛历史与质量诊断",
                   "Fig. 07  Convergence history & diagnostics"),
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "07_convergence.png")


# ================================================================ #
# 工具函数
# ================================================================ #

def _save(fig, filename: str, dpi: int = 200):
    # 统一走项目级公共保存出口 save_figure（内部已自动 fix_log_labels）
    out = save_figure(fig, FIG_DIR / filename, dpi=dpi)
    print(f"  [保存] {out}")


# ================================================================ #
# 主流程
# ================================================================ #

def main():
    print("=" * 60)
    print("  水平井三层地层 LWD-FDEM 正反演完整流程")
    print("=" * 60)
    print(f"\n地层: {list(zip(LAYER_NAMES[LANG], THICKNESSES.tolist() + ['∞'], SIGMA_TRUE))}")
    print(f"观测: {len(FREQUENCIES)}频 × {len(SPACINGS)}距 × {len(COMPONENTS)}分量"
          f" = {len(FREQUENCIES)*len(SPACINGS)*len(COMPONENTS)} 道")

    fwd, cfg = build_forward()

    print("\n[图 01] 地质概念模型 ...")
    fig_01_concept_model()

    print("[图 02] 计算网格与趋肤深度 ...")
    fig_02_mesh()

    print("[图 03] 观测系统配置 ...")
    fig_03_survey(cfg, fwd)

    print("[图 04] 正演响应曲线 ...")
    fig_04_forward_response(fwd, cfg)

    print("[图 05] 合成含噪观测数据 ...")
    dobs, std = fig_05_synthetic_data(fwd, cfg)

    print("[图 06] Occam 反演 ...")
    rec_sigma, history = fig_06_inversion(fwd, dobs, std)

    print("[图 07] 收敛诊断 ...")
    fig_07_convergence(history, dobs, fwd)

    print(f"\n全部图版已保存至: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
