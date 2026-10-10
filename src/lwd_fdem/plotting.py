"""LWD FDEM 正演可视化：模型图 / 网格图 / 响应图。

绘制风格对齐 external/SimPEG-src/fdem_forward_validation 中的判据图：

  plot_model    —— 模型图：1D 层状示意（空气层+地表+σ 着色层块+σ/ρ/层名+
                   层厚 h+井眼+Tx/Rx 线圈，仿 fdem_model_schematic）；
                   2D/3D 为 log10(σ) 剖面（仿 fdem_fwd_2d_cyl/fdem_fwd_3d）
  plot_mesh     —— 网格图：1D 分层界面；2D/3D 在 log10(σ) 剖面上叠加网格线
                   （plot_image/plot_slice 的 grid=True），并标注收发线圈
  plot_response —— 结果图：1D 为 loglog |响应|-频率 频散曲线（仿 fdem_fwd_1d）；
                   2D/3D 为 |响应|-收发距 曲线
  plot_forward_views —— 一次性输出上述三类图
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from .forward import LWDFDEMForward

__all__ = ["plot_model", "plot_mesh", "plot_response", "plot_forward_views"]


# ------------------------------------------------------------------ #
# 内部工具
# ------------------------------------------------------------------ #
def _new_ax(figsize=(6.4, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _log_sigma(sigma: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(sigma, dtype=float), 1e-12))


def _tool_positions(fwd: LWDFDEMForward):
    """返回发射源坐标与逐收发距接收器坐标 (n_spacing, 3)。"""
    cfg = fwd.tool
    src = np.asarray(cfg.source_location, dtype=float)
    rx = []
    for s in cfg.spacings:
        if cfg.receiver_geometry == "axial":
            rx.append(src + np.array([0.0, 0.0, float(s)]))
        else:
            rx.append(src + np.array([float(s), 0.0, 0.0]))
    return src, np.asarray(rx)


def _mark_tool(ax, src, rx, h_axis: int, v_axis: int):
    """在空间剖面 (h_axis, v_axis) 上标注源与接收线圈。"""
    ax.plot(src[h_axis], src[v_axis], marker="o", ms=10, mfc="crimson",
            mec="k", zorder=6, label="Tx 发射源")
    for i, p in enumerate(rx):
        ax.plot(p[h_axis], p[v_axis], marker="s", ms=7, mfc="dodgerblue",
                mec="k", zorder=6, label="Rx 接收线圈" if i == 0 else None)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _slice_profile(fwd: LWDFDEMForward, vals: np.ndarray, ax, grid: bool,
                   cmap: str):
    """2D/3D 剖面填色（log10 σ），返回 (mappable, 轴标签, 标题)。"""
    if fwd.geometry == "2d_cyl":
        im = fwd.mesh.plot_image(vals, ax=ax, cmap=cmap, grid=grid)[0]
        return im, "r (m)", "z (m)", "2D 轴对称 (r-z, log10 S/m)"
    ny = fwd.mesh.shape_cells[1]
    iy = ny // 2
    im = fwd.mesh.plot_slice(vals, normal="y", ind=iy, ax=ax,
                             cmap=cmap, grid=grid)[0]
    return im, "x (m)", "z (m)", f"3D (y=0 切片 {iy}/{ny}, log10 S/m)"


# ------------------------------------------------------------------ #
# 模型图
# ------------------------------------------------------------------ #
def plot_model(fwd: LWDFDEMForward, sigma: np.ndarray, ax=None,
               labels: Optional[Sequence[str]] = None,
               cmap: str = "viridis") -> plt.Axes:
    """绘制电导率模型：1D 层状示意 / 2D r-z / 3D x-z 剖面（log10 σ）。"""
    sigma = np.asarray(sigma, dtype=float)
    if ax is None:
        _, ax = _new_ax()

    if fwd.geometry == "1d":
        return _plot_1d_schematic(fwd, sigma, ax, labels)

    im, xl, yl, tt = _slice_profile(fwd, _log_sigma(sigma), ax, grid=False,
                                    cmap=cmap)
    ax.figure.colorbar(im, ax=ax, label=r"$\log_{10}\ \sigma$ (S/m)")
    ax.set(xlabel=xl, ylabel=yl, title="电导率模型 " + tt)
    ax.set_aspect("equal", adjustable="box")
    return ax


def _plot_1d_schematic(fwd, sigma, ax, labels=None):
    """1D 层状地电模型示意（仿 fdem_model_schematic 第①栏）。"""
    th = np.asarray(fwd.layer_thicknesses, dtype=float)
    n = len(sigma)
    ls = _log_sigma(sigma)
    norm = (ls - ls.min()) / (ls.max() - ls.min() + 1e-12)
    ztop = 3.0

    # 空气层 + 地表
    ax.add_patch(Rectangle((0, 0), 1.0, ztop, color="lightcyan", ec="none"))
    ax.text(0.78, ztop / 2, "空气\nσ≈0", ha="center", va="center",
            fontsize=10, color="steelblue")
    ax.plot([-0.05, 1.05], [0, 0], color="saddlebrown", lw=2.5)
    ax.text(1.06, 0, " 地表 z=0", color="saddlebrown", fontsize=10, va="center")

    depth = 0.0
    for i in range(n):
        inf = i >= len(th)
        h = 4.0 if inf else th[i]
        ax.add_patch(Rectangle((0, -depth - h), 1.0, h,
                               color=plt.cm.viridis(norm[i]), ec="k", alpha=0.85))
        name = labels[i] if labels and i < len(labels) else (
            "半空间" if inf else f"层 {i+1}")
        ax.text(0.5, -depth - h / 2,
                f"σ{i+1}={sigma[i]:.3g} S/m\nρ{i+1}={1.0/sigma[i]:.3g} Ω·m\n({name})",
                ha="center", va="center", fontsize=9.5,
                color="w" if norm[i] < 0.55 else "k")
        if not inf:
            ax.annotate("", xy=(1.08, -depth), xytext=(1.08, -depth - h),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
            ax.text(1.12, -depth - h / 2, f"h{i+1}={h:.1f} m", va="center",
                    fontsize=9.5)
            depth += h
        else:
            break

    # 井眼 + 共轴收发线圈
    tool_x = 0.22
    ax.plot([tool_x, tool_x], [ztop, -depth], color="k", lw=2, alpha=0.45)
    sp = float(np.max(fwd.tool.spacings)) if len(fwd.tool.spacings) else 1.0
    tx_z, rx_z = -1.0, -1.0 - sp
    ax.add_patch(Ellipse((tool_x, tx_z), 0.16, 0.14, color="crimson", ec="k"))
    ax.text(tool_x - 0.05, tx_z, "Tx", color="crimson", fontsize=10,
            ha="right", va="center", weight="bold")
    ax.add_patch(Ellipse((tool_x, rx_z), 0.16, 0.14, color="blue", ec="k"))
    ax.text(tool_x - 0.05, rx_z, "Rx", color="blue", fontsize=10,
            ha="right", va="center", weight="bold")
    ax.annotate("", xy=(-0.02, tx_z), xytext=(-0.02, rx_z),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
    ax.text(-0.05, (tx_z + rx_z) / 2, f"s={sp:.1f} m", fontsize=9,
            va="center", ha="right")

    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(-(depth + 4.0) - 0.3, ztop + 0.6)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel("径向 r (m)")
    ax.set_ylabel("深度 z (m)")
    ax.set_title("1D 层状地电模型 + 共轴线圈")
    return ax


# ------------------------------------------------------------------ #
# 网格图
# ------------------------------------------------------------------ #
def plot_mesh(fwd: LWDFDEMForward, sigma: Optional[np.ndarray] = None,
              ax=None, mark_tool: bool = True) -> plt.Axes:
    """绘制计算网格：1D 分层界面 / 2D-3D 在 log10(σ) 剖面上叠加网格线。

    Args:
        fwd: 正演对象。
        sigma: 电导率模型；给出时 2D/3D 背景填充 log10(σ)，否则仅画网格线。
        ax: 目标轴；None 时新建图形。
        mark_tool: 是否在空间可对应的几何上标注源/收位置。

    Returns:
        绑定网格剖面的 matplotlib Axes。
    """
    if ax is None:
        _, ax = _new_ax()

    if fwd.geometry == "1d":
        th = np.asarray(fwd.layer_thicknesses, dtype=float)
        zb = np.r_[0.0, np.cumsum(th), th[-1] * 1.5 + th.sum()]
        for i in range(len(zb) - 1):
            ax.fill_between([0, 1], zb[i], zb[i + 1],
                            facecolor=plt.cm.viridis(i / max(len(zb) - 2, 1)),
                            edgecolor="k", lw=0.9, alpha=0.85)
            ax.text(0.5, 0.5 * (zb[i] + zb[i + 1]),
                    f"层 {i+1}" + ("" if i < len(th) else " (半空间)"),
                    ha="center", va="center", fontsize=9, color="w")
            if i < len(th):
                ax.annotate("", xy=(1.05, zb[i]), xytext=(1.05, zb[i + 1]),
                            arrowprops=dict(arrowstyle="<->", color="k", lw=1))
                ax.text(1.1, 0.5 * (zb[i] + zb[i + 1]), f"h{i+1}={th[i]:g} m",
                        va="center", fontsize=9)
        ax.set_xlim(-0.1, 1.9)
        ax.set_ylim(zb[-1], 0)
        ax.set_xticks([])
        ax.set_xlabel("（径向由 Hankel 变换处理）")
        ax.set_ylabel("深度 (m)")
        ax.set_title("1D 层状网格（分层界面）")
        return ax

    # 2D/3D：log10(σ) 填色 + grid=True 叠加网格线（仿 fdem_fwd_2d_cyl/3d）
    vals = (_log_sigma(sigma) if sigma is not None
            else np.zeros(fwd.mesh.n_cells))
    im, xl, yl, tt = _slice_profile(fwd, vals, ax, grid=True, cmap="viridis")
    ax.figure.colorbar(im, ax=ax, label=r"$\log_{10}\ \sigma$ (S/m)")
    ax.set(xlabel=xl, ylabel=yl, title="计算网格 " + tt)
    ax.set_aspect("equal", adjustable="box")
    if mark_tool:
        _mark_tool(ax, *_tool_positions(fwd), h_axis=0, v_axis=2)
    return ax


# ------------------------------------------------------------------ #
# 响应图
# ------------------------------------------------------------------ #
def plot_response(fwd: LWDFDEMForward, d: np.ndarray, ax=None,
                  analytic: Optional[np.ndarray] = None) -> plt.Axes:
    """绘制响应曲线：1D 为 loglog |d|-频率 频散曲线；2D/3D 为 |d|-收发距。

    Args:
        analytic: 可选解析解复数数组，形状 (n_freq, n_spacing, n_comp//2) 或
            (n_freq,)（1D 时叠加对比，仿 fdem_fwd_1d）。
    """
    if ax is None:
        _, ax = _new_ax()
    mat = fwd.to_matrix(d)                 # (n_freq, n_spacing, n_comp)
    cfg = fwd.tool

    if fwd.geometry == "1d":
        for j, s in enumerate(cfg.spacings):
            for k, comp in enumerate(cfg.components):
                ax.loglog(cfg.frequencies, np.abs(mat[:, j, k]), marker="o",
                          label=f"s={s:g} m {comp}")
        if analytic is not None:
            an = np.asarray(analytic)
            if an.ndim == 1:
                an = an.reshape(-1, 1, 1) if np.iscomplexobj(an) else an[..., None]
            ax.loglog(cfg.frequencies, np.abs(an[:, 0, 0].real), "k--",
                      lw=1, label="解析 real")
            ax.loglog(cfg.frequencies, np.abs(an[:, 0, 0].imag), "k:",
                      lw=1, label="解析 imag")
        ax.set(xlabel="频率 (Hz)", ylabel="|响应| (A/m)",
               title="LWD 1D 频散曲线")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        return ax

    fwd.plot_response(d, ax=ax)            # 复用：|响应|-收发距
    ax.set_title("LWD 频域响应")
    return ax


# ------------------------------------------------------------------ #
# 一站式输出
# ------------------------------------------------------------------ #
def plot_forward_views(fwd: LWDFDEMForward, sigma: np.ndarray,
                       d: Optional[np.ndarray] = None, out_dir=".",
                       prefix: str = "fwd", dpi: int = 150,
                       labels: Optional[Sequence[str]] = None,
                       analytic: Optional[np.ndarray] = None) -> dict:
    """绘制并保存模型图 / 网格图 / 响应图三件套。

    Args:
        fwd: 正演对象。
        sigma: 电导率模型向量 (S/m)。
        d: 响应数据向量 (nD,)；None 时自动 fwd.dpred(sigma)。
        out_dir: 输出目录（不存在则创建）。
        prefix: 文件名前缀，产出 ``{prefix}_model/_mesh/_response.png``。
        dpi: 存图分辨率。
        labels: 1D 层名（可选，用于模型示意图）。
        analytic: 1D 解析解（可选，叠加到响应图）。

    Returns:
        dict: {'model': fig, 'mesh': fig, 'response': fig, 'paths': {...}}
    """
    import pathlib

    sigma = np.asarray(sigma, dtype=float)
    d = fwd.dpred(sigma) if d is None else np.asarray(d)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figs, paths = {}, {}
    steps = (
        ("model", lambda a: plot_model(fwd, sigma, ax=a, labels=labels)),
        ("mesh", lambda a: plot_mesh(fwd, sigma=sigma, ax=a)),
        ("response", lambda a: plot_response(fwd, d, ax=a, analytic=analytic)),
    )
    for key, fn in steps:
        fig, ax = _new_ax()
        fn(ax)
        fig.tight_layout()
        p = out / f"{prefix}_{key}.png"
        fig.savefig(p, dpi=dpi)
        figs[key], paths[key] = fig, str(p)
    return {**figs, "paths": paths}
