"""LWD FDEM 正反演可视化 —— SCI 论文图版风格。

设计目标（对标期刊图版）：
  * 统一出版级样式：衬线字体（Times/STIX 数学）、内向刻度、次刻度、
    去边框图例、300 dpi、`bbox_inches='tight'`。
  * 面板以 (a)(b)(c) 标注，英文坐标 + 物理量 LaTeX 记法与单位。
  * `plot_forward_figure`  将 模型 / 网格 / 响应 合成为一张多联图版。
  * `plot_inversion_figure` 将 真实/回收剖面 与 收敛史 合成为一张图版。
  * `plot_model` / `plot_mesh` / `plot_response` 为可绑定到自定义 Axes 的单面板。
  * `plot_forward_views`  分别输出三张单图（向后兼容测试）。
  * `pub_style`  上下文管理器，供任意脚本套用同一出版样式。

绘制内容约定参考 external/SimPEG-src/fdem_forward_validation。
"""

from __future__ import annotations

import contextlib
import pathlib
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Ellipse

from .forward import LWDFDEMForward

__all__ = [
    "pub_style", "plot_model", "plot_mesh", "plot_response",
    "plot_forward_figure", "plot_inversion_figure", "plot_forward_views",
    "set_cjk_rcparams", "fix_log_labels", "apply_rcparams", "t", "LBL",
]

# ------------------------------------------------------------------ #
# 出版级样式
# ------------------------------------------------------------------ #
_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "savefig.dpi": 300,
    "figure.dpi": 110,
    "axes.grid": False,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "image.cmap": "viridis",
}

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]
_LINESTYLES = ["-", "--", "-.", ":"]

# ------------------------------------------------------------------ #
# 双语术语表（图面文字集中管理，供 t() 按语言取用）
# ------------------------------------------------------------------ #
#: 每个条目为 {"en": 英文, "zh": 中文}；英文为 SCI 图版默认，
#: 中文供国内期刊/汇报。数学符号类标签（如 $\sigma$）语言无关，不入表。
LBL = {
    "depth":            {"en": "Depth (m)",                "zh": "深度 (m)"},
    "frequency":        {"en": "Frequency (Hz)",           "zh": "频率 (Hz)"},
    "spacing":          {"en": "Spacing (m)",              "zh": "收发距 (m)"},
    "layered_model":    {"en": "Layered model",            "zh": "层状模型"},
    "conductivity_model": {"en": "Conductivity model",     "zh": "电导率模型"},
    "layer_discretization": {"en": "Layer discretization", "zh": "层状网格离散"},
    "computational_mesh": {"en": "Computational mesh",     "zh": "计算网格"},
    "response":         {"en": "Response",                 "zh": "响应"},
    "true_model":       {"en": "True model",               "zh": "真实模型"},
    "recovered_model":  {"en": "Recovered model",          "zh": "回收模型"},
    "profile_recovery": {"en": "Profile recovery",         "zh": "剖面回收"},
    "convergence":      {"en": "Convergence",              "zh": "收敛"},
    "data_misfit":      {"en": r"$\phi_d$ (data misfit)",  "zh": r"$\phi_d$（数据拟合差）"},
    "model_norm":       {"en": r"$\phi_m$ (model norm)",   "zh": r"$\phi_m$（模型范数）"},
    "iteration":        {"en": "Iteration",                "zh": "迭代次数"},
    "objective":        {"en": "Objective",                "zh": "目标函数值"},
    "air":              {"en": "air",                      "zh": "空气"},
    "true":             {"en": "true",                     "zh": "真实"},
    "recovered":        {"en": "recovered",                "zh": "回收"},
    "analytic":         {"en": "analytic",                 "zh": "解析解"},
    "half_space":       {"en": "(half-space)",             "zh": "（半空间）"},
    "layer":            {"en": "layer",                    "zh": "层"},
    "cell":             {"en": "cell",                     "zh": "单元"},
    "baseline":         {"en": "baseline",                 "zh": "基线"},
}


def t(key: str, lang: str = "en") -> str:
    """按语言取图面文字。

    Args:
        key: `LBL` 中的键。
        lang: 'en'（默认，SCI）或 'zh'。

    Returns:
        对应语言的标签字符串；未知键则回退为 key 本身。
    """
    entry = LBL.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry["en"])


# 语言相关的字体配置（叠加在出版级 _RC 之上）
_RC_LANG = {
    "en": {},  # _RC 已为 serif/Times，适配英文 SCI
    "zh": {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    },
}


def apply_rcparams(lang: str = "en") -> None:
    """按语言设置全局字体（幂等），供手绘脚本顶部调用。

    Args:
        lang: 'en' 用 serif/Times（SCI 默认）；'zh' 用 SimHei 正文 + stix 数学。
    """
    if lang == "zh":
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    else:
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"


@contextlib.contextmanager
def pub_style(lang: str = "en", **overrides):
    """套用出版级 rcParams 的上下文管理器（退出时还原）。

    Args:
        lang: 'en'（默认，英文 SCI）或 'zh'（中文标注，数学仍用 stix）。
        overrides: 额外覆盖的 rcParams。
    """
    rc = {**_RC, **_RC_LANG.get(lang, {}), **overrides}
    old = {k: matplotlib.rcParams[k] for k in rc}
    matplotlib.rcParams.update(rc)
    try:
        yield
    finally:
        matplotlib.rcParams.update(old)


# ------------------------------------------------------------------ #
# 通用样式工具（供所有 example/脚本复用）
# ------------------------------------------------------------------ #
def set_cjk_rcparams() -> None:
    """配置中文字体与数学负号，供所有脚本统一调用（幂等）。

    背景：将 `font.sans-serif` 设为 SimHei 后，普通文本可正常显示中文，
    但 SimHei 缺少数学负号字形 U+2212；且 `axes.unicode_minus=False` 仅对
    普通文本生效，管不到 mathtext（对数轴刻度、公式标签）。本函数将
    mathtext 字体集固定为 stix（自带负号字形），从源头缓解缺字问题；
    对数轴刻度的残留问题再配合 `fix_log_labels()` 彻底解决。
    """
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "stix"


def fix_log_labels(fig):
    """将图内所有对数轴的刻度标签改用纯 mathtext（不含 \\mathdefault）。

    背景：matplotlib 默认对数轴标签为 $\\mathdefault{10^{-4}}$，其中
    \\mathdefault 会解析到正文字体（如 SimHei，缺 U+2212 负号），导致指数
    显示为方框。本函数用 FuncFormatter 输出 $10^{-4}$ 形式的纯 mathtext，
    负号由 mathtext fontset（stix）自带字形渲染，彻底规避缺字问题。

    Args:
        fig: matplotlib Figure 对象；就地修改其下所有 log 尺度坐标轴。

    Returns:
        fig（便于链式调用）。
    """
    import math
    from matplotlib.ticker import FuncFormatter

    def _fmt(x, pos):
        if x is None or x <= 0:
            return ""
        e = math.floor(math.log10(x))
        m = int(round(x / 10 ** e))
        if m == 1:
            return f"$10^{{{e}}}$"
        if m in (2, 5):
            return f"${m}\\times10^{{{e}}}$"
        return ""

    formatter = FuncFormatter(_fmt)
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_major_formatter(formatter)
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_formatter(formatter)
    return fig


# ------------------------------------------------------------------ #
# 内部工具
# ------------------------------------------------------------------ #
def _log_sigma(sigma) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(sigma, dtype=float), 1e-12))


def _panel_title(ax, desc: str, panel: Optional[str] = None):
    ax.set_title(f"({panel}) {desc}" if panel else desc, pad=6)


def _tool_positions(fwd: LWDFDEMForward):
    cfg = fwd.tool
    src = np.asarray(cfg.source_location, dtype=float)
    rx = []
    for s in cfg.spacings:
        off = np.array([0.0, 0.0, float(s)]) if cfg.receiver_geometry == "axial" \
            else np.array([float(s), 0.0, 0.0])
        rx.append(src + off)
    return src, np.asarray(rx)


def _mark_tool(ax, src, rx, h_axis=0, v_axis=2):
    ax.plot(src[h_axis], src[v_axis], marker="o", ms=7, mfc="crimson",
            mec="k", mew=0.6, zorder=6, label="Tx")
    for i, p in enumerate(rx):
        ax.plot(p[h_axis], p[v_axis], marker="s", ms=5.5, mfc="tab:blue",
                mec="k", mew=0.6, zorder=6, label="Rx" if i == 0 else None)
    ax.legend(loc="upper right", fontsize=7.5, handletextpad=0.3)


def _section(fwd, vals, ax, grid, cmap="viridis"):
    """2D/3D 空间剖面填色（vals 为 log10 σ），返回 mappable。"""
    if fwd.geometry == "2d_cyl":
        im = fwd.mesh.plot_image(vals, ax=ax, cmap=cmap, grid=grid)[0]
        ax.set(xlabel="r (m)", ylabel="z (m)")
    else:
        ny = fwd.mesh.shape_cells[1]
        iy = ny // 2
        im = fwd.mesh.plot_slice(vals, normal="y", ind=iy, ax=ax,
                                 cmap=cmap, grid=grid)[0]
        ax.set(xlabel="x (m)", ylabel="z (m)")
    return im


def _add_cbar(fig, im, label):
    cb = fig.colorbar(im, ax=im.axes, pad=0.03)
    cb.set_label(label)
    cb.outline.set_linewidth(0.6)
    return cb


# ------------------------------------------------------------------ #
# 单面板：模型
# ------------------------------------------------------------------ #
def plot_model(fwd, sigma, ax=None, labels=None, panel=None,
               show_title=True, cmap="viridis", colorbar=True, lang="en"):
    """电导率模型：1D 层状示意 / 2D r-z / 3D x-z 剖面（log10 σ）。"""
    sigma = np.asarray(sigma, dtype=float)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(3.6, 3.6))
    if fwd.geometry == "1d":
        _draw_1d_model(fwd, sigma, ax, labels, lang=lang)
        if show_title:
            _panel_title(ax, t("layered_model", lang), panel)
        return ax
    im = _section(fwd, _log_sigma(sigma), ax, grid=False, cmap=cmap)
    if show_title:
        _panel_title(ax, t("conductivity_model", lang), panel)
    ax.set_aspect("equal", adjustable="box")
    if colorbar:
        _add_cbar(ax.figure, im, r"$\log_{10}\,\sigma$  (S/m)")
    return ax


def _draw_1d_model(fwd, sigma, ax, labels=None, lang="en"):
    th = np.asarray(fwd.layer_thicknesses, dtype=float)
    n = len(sigma)
    ls = _log_sigma(sigma)
    norm = (ls - ls.min()) / (ls.max() - ls.min() + 1e-12)
    z_air = -3.0

    ax.add_patch(Rectangle((0, z_air), 1.0, -z_air,
                           facecolor=(0.86, 0.95, 1.0), edgecolor="none", zorder=0))
    ax.text(0.82, z_air / 2, t("air", lang), ha="center", va="center",
            fontsize=8, style="italic", color="steelblue", zorder=3)
    ax.axhline(0, color="k", lw=1.0, zorder=2)

    depth = 0.0
    for i in range(n):
        inf = i >= len(th)
        h = 4.0 if inf else th[i]
        ax.add_patch(Rectangle((0, depth), 1.0, h, facecolor=plt.cm.viridis(norm[i]),
                               edgecolor="k", lw=0.6, alpha=0.92, zorder=1))
        name = labels[i] if labels and i < len(labels) else (
            t("baseline", lang) if inf
            else f"{t('layer', lang)} {i+1}")
        ax.text(0.62, depth + h / 2,
                rf"$\sigma_{i+1}={sigma[i]:.3g}$ S/m" "\n"
                rf"$\rho_{i+1}={1.0/sigma[i]:.3g}$" r" $\Omega$m" "\n"
                rf"({name})",
                ha="center", va="center", fontsize=7.2,
                color="w" if norm[i] < 0.55 else "k", zorder=3)
        if not inf:
            ax.annotate("", xy=(1.05, depth), xytext=(1.05, depth + h),
                        arrowprops=dict(arrowstyle="<->", lw=0.7, color="k"))
            ax.text(1.08, depth + h / 2, rf"${h:.1f}$", va="center", fontsize=7.2)
            depth += h
        else:
            depth += h
            break

    tool_x = 0.25
    ax.plot([tool_x, tool_x], [z_air, depth], color="0.35", lw=1.2, zorder=4)
    sp = float(np.max(fwd.tool.spacings)) if len(fwd.tool.spacings) else 1.0
    txz, rxz = 1.0, 1.0 + sp
    ax.add_patch(Ellipse((tool_x, txz), 0.13, 0.30, facecolor="crimson",
                         edgecolor="k", lw=0.6, zorder=5))
    ax.add_patch(Ellipse((tool_x, rxz), 0.13, 0.30, facecolor="tab:blue",
                         edgecolor="k", lw=0.6, zorder=5))
    ax.annotate("", xy=(tool_x - 0.15, txz), xytext=(tool_x - 0.15, rxz),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="k"))
    ax.text(tool_x - 0.18, (txz + rxz) / 2, rf"$s={sp:.1f}$ m",
            ha="right", va="center", fontsize=7.2)
    ax.text(tool_x + 0.08, txz, "Tx", fontsize=7.2, color="crimson", va="center")
    ax.text(tool_x + 0.08, rxz, "Rx", fontsize=7.2, color="tab:blue", va="center")

    ax.set_xlim(-0.55, 1.45)
    ax.set_ylim(depth, z_air)          # 深度向下
    ax.set_xticks([])
    ax.set_ylabel(t("depth", lang))


# ------------------------------------------------------------------ #
# 单面板：网格
# ------------------------------------------------------------------ #
def plot_mesh(fwd, sigma=None, ax=None, panel=None, show_title=True,
              mark_tool=True, cmap="viridis", colorbar=True, lang="en"):
    """计算网格：1D 分层界面 / 2D-3D 在 log10(σ) 剖面上叠加网格线。"""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(3.6, 3.6))
    if fwd.geometry == "1d":
        th = np.asarray(fwd.layer_thicknesses, dtype=float)
        zb = np.r_[0.0, np.cumsum(th), th[-1] * 1.5 + th.sum()]
        for i in range(len(zb) - 1):
            ax.add_patch(Rectangle((0, zb[i]), 1.0, zb[i + 1] - zb[i],
                                   facecolor="none", edgecolor="k", lw=0.8))
            ax.text(0.5, 0.5 * (zb[i] + zb[i + 1]),
                    f"{t('cell', lang)} {i+1}"
                    + ("" if i < len(th) else " " + t("half_space", lang)),
                    ha="center", va="center", fontsize=7.2)
        ax.set_xlim(0, 1); ax.set_xticks([])
        ax.set_ylim(zb[-1], 0)
        ax.set_ylabel(t("depth", lang))
        if show_title:
            _panel_title(ax, t("layer_discretization", lang), panel)
        return ax

    vals = _log_sigma(sigma) if sigma is not None else np.zeros(fwd.mesh.n_cells)
    im = _section(fwd, vals, ax, grid=True, cmap=cmap)
    ax.set_aspect("equal", adjustable="box")
    if mark_tool:
        _mark_tool(ax, *_tool_positions(fwd))
    if show_title:
        _panel_title(ax, t("computational_mesh", lang), panel)
    if colorbar:
        _add_cbar(ax.figure, im, r"$\log_{10}\,\sigma$  (S/m)")
    return ax


# ------------------------------------------------------------------ #
# 单面板：响应
# ------------------------------------------------------------------ #
def plot_response(fwd, d, ax=None, panel=None, show_title=True, analytic=None,
                  lang="en"):
    """响应曲线：1D 为 log-log |d|-频率；2D/3D 为 |d|-收发距。"""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(4.0, 3.2))
    mat = fwd.to_matrix(d)
    cfg = fwd.tool
    freqs, sps, comps = cfg.frequencies, cfg.spacings, cfg.components

    idx = 0
    if fwd.geometry == "1d":
        for j, s in enumerate(sps):
            for k, c in enumerate(comps):
                ax.loglog(freqs, np.abs(mat[:, j, k]),
                          marker=_MARKERS[idx % len(_MARKERS)],
                          ls=_LINESTYLES[idx % len(_LINESTYLES)],
                          ms=4.5, label=f"{c}, $s$={s:g} m")
                idx += 1
        ax.set(xlabel=t("frequency", lang), ylabel=r"$|H_z^{\mathrm{s}}|$ (A/m)")
        if analytic is not None:
            an = np.abs(np.asarray(analytic))
            ax.loglog(freqs, an.reshape(-1)[::max(1, an.size // len(freqs))]
                      if an.size > len(freqs) else an, "k:", lw=1.2,
                      label=t("analytic", lang))
    else:
        for i, f in enumerate(freqs):
            for k, c in enumerate(comps):
                ax.semilogy(sps, np.abs(mat[i, :, k]),
                            marker=_MARKERS[idx % len(_MARKERS)],
                            ls=_LINESTYLES[idx % len(_LINESTYLES)],
                            ms=4.5, label=f"{c}, $f$={f:g} Hz")
                idx += 1
        ax.set(xlabel=t("spacing", lang), ylabel=r"$|d|$ (A/m)")

    ax.grid(True, which="major")
    ax.legend(loc="best", ncol=2 if idx > 4 else 1)
    if show_title:
        _panel_title(ax, t("response", lang), panel)
    return ax


# ------------------------------------------------------------------ #
# 合成图版
# ------------------------------------------------------------------ #
def plot_forward_figure(fwd, sigma, d=None, out_dir=".", prefix="fwd_plate",
                        dpi=300, labels=None, analytic=None, lang="en"):
    """将模型/网格/响应合成为一张 SCI 多联图版并保存，返回文件路径。"""
    sigma = np.asarray(sigma, dtype=float)
    d = fwd.dpred(sigma) if d is None else np.asarray(d)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    with pub_style(lang=lang):
        if fwd.geometry == "1d":
            fig = plt.figure(figsize=(7.2, 3.6))
            gs = GridSpec(1, 2, width_ratios=[1.0, 1.35], wspace=0.26,
                          figure=fig)
            plot_model(fwd, sigma, ax=fig.add_subplot(gs[0, 0]),
                       labels=labels, panel="a", lang=lang)
            plot_response(fwd, d, ax=fig.add_subplot(gs[0, 1]), panel="b",
                          analytic=analytic, lang=lang)
        else:
            fig = plt.figure(figsize=(10.8, 3.7))
            gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.28,
                          figure=fig)
            plot_model(fwd, sigma, ax=fig.add_subplot(gs[0, 0]), panel="a",
                       lang=lang)
            plot_mesh(fwd, sigma=sigma, ax=fig.add_subplot(gs[0, 1]),
                      panel="b", colorbar=False, lang=lang)
            plot_response(fwd, d, ax=fig.add_subplot(gs[0, 2]), panel="c",
                          lang=lang)

        path = out / f"{prefix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def plot_inversion_figure(fwd, rec_sigma, history, sigma_true=None,
                          out_dir=".", prefix="inv_plate", dpi=300, lang="en"):
    """反演图版：1D 剖面回收 + 收敛史；2D/3D 真实/回收剖面 + 收敛史。"""
    rec_sigma = np.asarray(rec_sigma, dtype=float)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    with pub_style(lang=lang):
        if fwd.geometry == "1d":
            fig = plt.figure(figsize=(7.2, 3.6))
            gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28, figure=fig)
            _profile_1d(fwd, rec_sigma, sigma_true, fig.add_subplot(gs[0, 0]),
                        "a", lang=lang)
            _convergence(history, fig.add_subplot(gs[0, 1]), "b", lang=lang)
        else:
            fig = plt.figure(figsize=(10.8, 3.7))
            gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.3,
                          figure=fig)
            if sigma_true is not None:
                im_t = _section(fwd, _log_sigma(sigma_true),
                                fig.add_subplot(gs[0, 0]), grid=False)
                _panel_title(fig.axes[-1], t("true_model", lang), "a")
                _add_cbar(fig, im_t, r"$\log_{10}\,\sigma$  (S/m)")
            im_r = _section(fwd, _log_sigma(rec_sigma),
                            fig.add_subplot(gs[0, 1]), grid=False)
            _panel_title(fig.axes[-1], t("recovered_model", lang), "b")
            _add_cbar(fig, im_r, r"$\log_{10}\,\sigma$  (S/m)")
            _convergence(history, fig.add_subplot(gs[0, 2]), "c", lang=lang)

        path = out / f"{prefix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def _profile_1d(fwd, rec, true, ax, panel=None, lang="en"):
    th = np.asarray(fwd.layer_thicknesses, dtype=float)
    n = len(rec)
    bounds = np.r_[0.0, np.cumsum(th)]
    centers = np.array([0.5 * (bounds[i] + bounds[i + 1]) if i < len(th)
                        else bounds[-1] + 0.5 * th[-1] for i in range(n)])
    if true is not None:
        ax.step(np.asarray(true, float), centers, where="mid", color="k",
                ls="--", lw=1.6, label=t("true", lang))
    ax.step(rec, centers, where="mid", color="C0", lw=2.0, marker="o",
            ms=4, label=t("recovered", lang))
    ax.set_xscale("log")
    ax.set(xlabel=r"$\sigma$ (S/m)", ylabel=t("depth", lang))
    ax.invert_yaxis()
    ax.legend(loc="best")
    _panel_title(ax, t("profile_recovery", lang), panel)


def _convergence(history, ax, panel=None, lang="en"):
    it = history["iterations"]
    x = list(range(1, len(it) + 1))
    ax.semilogy(x, [d["phi_d"] for d in it], "o-", ms=4,
                label=t("data_misfit", lang))
    ax.semilogy(x, [d["phi_m"] for d in it], "s-", ms=4,
                label=t("model_norm", lang))
    ax.set(xlabel=t("iteration", lang), ylabel=t("objective", lang))
    ax.grid(True, which="both")
    ax.legend(loc="best")
    _panel_title(ax, t("convergence", lang), panel)


# ------------------------------------------------------------------ #
# 三张单图（向后兼容）
# ------------------------------------------------------------------ #
def plot_forward_views(fwd, sigma, d=None, out_dir=".", prefix="fwd",
                       dpi=300, labels=None, analytic=None, lang="en"):
    """分别输出 model / mesh / response 三张单图，返回 fig 与路径字典。"""
    sigma = np.asarray(sigma, dtype=float)
    d = fwd.dpred(sigma) if d is None else np.asarray(d)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    figs, paths = {}, {}
    with pub_style(lang=lang):
        for key, fn in (
            ("model", lambda a: plot_model(fwd, sigma, ax=a, labels=labels,
                                           lang=lang)),
            ("mesh", lambda a: plot_mesh(fwd, sigma=sigma, ax=a, lang=lang)),
            ("response", lambda a: plot_response(fwd, d, ax=a, analytic=analytic,
                                                 lang=lang)),
        ):
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            fn(ax)
            fig.tight_layout()
            p = out / f"{prefix}_{key}.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            figs[key], paths[key] = fig, str(p)
    return {**figs, "paths": paths}
