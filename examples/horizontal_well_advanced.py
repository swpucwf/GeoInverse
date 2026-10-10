"""水平井进阶示例 —— 演示业务层已具备、此前仅在示例中未接入的能力。

对比实验（遵循"频率低/中/高三条代表曲线"约定）：
  图 A  源类型对比：磁偶极子 MagDipole  vs  有限回线源 CircularLoop
        （通过 LWDToolConfig(source_type=...) 切换，fdemcore 原生支持）
  图 B  反演正则化对比：Occam 平滑 (sparse=False)  vs  稀疏块状 (sparse=True)
        （LWDFDEMInversion(sparse=True) 已封装 Sparse + IRLS + ProjectedGNCG）

模型：水平井穿三层水平地层，1D 层状 + 径向偏移（Hankel 解析）。
复用 src/lwd_fdem 公共设施：apply_rcparams / save_figure / case_fig_dir / LANG / T。
仅新增文件，不修改任何原有代码。

运行：conda activate lwd && python examples/horizontal_well_advanced.py
输出：examples/figs/horizontal_well_advanced/
Copyright 2026 — Apache-2.0
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    add_synthetic_noise, apply_rcparams, case_fig_dir, make_data_object,
    save_figure,
)
from src.fdemcore import directives


class _SparseOKInversion(LWDFDEMInversion):
    """业务层封装：修正 LWDFDEMInversion 在 sparse=True 时的指令冲突。

    底层 bug（未改）：build_directives 对 sparse 同时加了 BetaSchedule 与
    UpdateIRLS，而 UpdateIRLS 自管 β 冷却、断言不得共存 BetaSchedule。
    这里不改原模块，仅子类过滤掉 BetaSchedule（属业务层封装）。
    """

    def build_directives(self):
        dl = super().build_directives()
        if self.sparse:
            dl = [d for d in dl if not isinstance(d, directives.BetaSchedule)]
        return dl

FIG_DIR = case_fig_dir(__file__)

# ---- 图版语言 ----
LANG = "en"


def T(zh, en):
    return zh if LANG == "zh" else en


# ---- 模型与观测 ----
THICKNESSES = np.array([3.0, 6.0])
SIGMA_TRUE = np.array([0.20, 0.01, 0.50])
FREQUENCIES = (2e4, 5e4, 1e5)                 # 低/中/高三条代表频
SPACINGS = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
NOISE_LEVEL, SEED = 0.02, 42
LOOP_RADIUS = 0.5                             # 回线源半径 (m)，放大以显出与点偶极差异


def make_forward(source_type):
    cfg = LWDToolConfig(
        frequencies=FREQUENCIES, spacings=SPACINGS, components=("real", "imag"),
        field_kind="h", secondary=True, receiver_geometry="radial",
        source_type=source_type, loop_radius=LOOP_RADIUS,
    )
    return LWDFDEMForward("1d", cfg, layer_thicknesses=THICKNESSES)


# ================================================================ #
# 图 A：接收场量对比（二次场 vs 总场）
# 注：回线源 CircularLoop 在 1D 层状仅支持零偏移中心回线，其源型对比见
#     垂直井 2D 示例 vertical_well_advanced.py。
# ================================================================ #

def fig_A_field_compare():
    def mk(secondary):
        cfg = LWDToolConfig(
            frequencies=FREQUENCIES, spacings=SPACINGS, components=("real", "imag"),
            field_kind="h", secondary=secondary, receiver_geometry="radial",
        )
        return LWDFDEMForward("1d", cfg, layer_thicknesses=THICKNESSES)
    fwd_sec = mk(True)
    fwd_tot = mk(False)
    m_sec = fwd_sec.to_matrix(fwd_sec.dpred(SIGMA_TRUE))
    m_tot = fwd_tot.to_matrix(fwd_tot.dpred(SIGMA_TRUE))
    freqs = np.array(FREQUENCIES)
    j = 2                                                     # 取 s=1.5 m

    amp_sec = np.hypot(m_sec[:, j, 0], m_sec[:, j, 1])
    amp_tot = np.hypot(m_tot[:, j, 0], m_tot[:, j, 1])
    rel = (amp_tot - amp_sec) / amp_sec * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.loglog(freqs, amp_sec, "o-", ms=7, lw=1.8, color="C0",
               label=T("二次场 secondary", "secondary"))
    ax1.loglog(freqs, amp_tot, "s--", ms=7, lw=1.8, color="C3",
               label=T("总场 total", "total"))
    ax1.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
            ylabel=r"$|H_z^{\mathrm{s}}|$ (A/m)",
            title=T(f"(a) 二次场 vs 总场（$s$={SPACINGS[j]:.1f} m）",
                    f"(a) secondary vs total ($s$={SPACINGS[j]:.1f} m)"))
    ax1.grid(True, which="both", alpha=0.3); ax1.legend()

    ax2.semilogx(freqs, rel, "D-", ms=7, lw=1.8, color="C4")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set(xlabel=T("频率 (Hz)", "Frequency (Hz)"),
            ylabel=T("相对差异 (%)", "Relative diff (%)"),
            title=T("(b) 总场相对二次场的差异", "(b) total vs secondary difference"))
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(T("图 A  接收场量对比：二次场 vs 总场（LWDToolConfig secondary 开关）",
                   "Fig. A  secondary vs total field (LWDToolConfig secondary flag)"),
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "A_field_compare.png", dpi=200)


# ================================================================ #
# 图 B：反演正则化对比（Occam 平滑 vs 稀疏）
# ================================================================ #

def _run_inv(fwd, data, sparse):
    inv = _SparseOKInversion(
        fwd, data, m0=np.log(0.1 * np.ones(len(SIGMA_TRUE))),
        max_iter=15, sparse=sparse,
        alpha_s=1e-3, alpha_x=1.0,
        beta0_ratio=1e-3, beta_cooling=3.0, chifact=1.0,
        sensitivity_weights=False, preconditioner=not sparse,
        optimizer="pgncg" if sparse else "gn",
    )
    return inv.run()


def fig_B_inversion_compare():
    fwd = make_forward("mag_dipole")
    dobs, std = add_synthetic_noise(fwd.dpred(SIGMA_TRUE), NOISE_LEVEL, seed=SEED)
    data = make_data_object(fwd.survey, dobs, absolute_std=std)

    rec_smooth, h_smooth = _run_inv(fwd, data, sparse=False)
    rec_sparse, h_sparse = _run_inv(fwd, data, sparse=True)

    th = THICKNESSES
    n = len(SIGMA_TRUE)
    bounds = np.r_[0.0, np.cumsum(th)]
    centers = np.array([0.5 * (bounds[i] + bounds[i + 1]) if i < len(th)
                        else bounds[-1] + 0.5 * th[-1] for i in range(n)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.step(SIGMA_TRUE, centers, where="mid", color="k", ls="--", lw=2.0,
             label=T("真实模型", "true"))
    ax1.step(rec_smooth, centers, where="mid", color="C0", lw=1.8, marker="o",
             ms=4, label=T("Occam 平滑", "Occam (smooth)"))
    ax1.step(rec_sparse, centers, where="mid", color="C3", lw=1.8, marker="s",
             ms=4, label=T("稀疏块状", "Sparse (blocky)"))
    ax1.set_xscale("log")
    ax1.set(xlabel=r"$\sigma$ (S/m)", ylabel=T("深度 (m)", "Depth (m)"),
            title=T("(a) 回收剖面对比", "(a) Recovered profile"))
    ax1.invert_yaxis(); ax1.grid(True, which="both", alpha=0.3); ax1.legend()

    for h, lab, c, mk in [(h_smooth, T("Occam 平滑", "smooth"), "C0", "o"),
                          (h_sparse, T("稀疏", "sparse"), "C3", "s")]:
        it = h["iterations"]
        x = list(range(1, len(it) + 1))
        ax2.semilogx(x, [d["phi_d"] for d in it], mk + "-", ms=4, lw=1.6,
                     color=c, label=lab)
    ax2.axhline(dobs.size, color="gray", ls=":", lw=1.2, label="nD")
    ax2.set(xlabel=T("迭代次数", "Iteration"), ylabel=r"$\phi_d$",
            title=T("(b) 数据拟合差收敛", "(b) Data-misfit convergence"))
    ax2.grid(True, which="both", alpha=0.3); ax2.legend()

    fig.suptitle(T("图 B  反演正则化对比：Occam 平滑 vs 稀疏块状（LWDFDEMInversion sparse 开关）",
                   "Fig. B  Inversion regularization: Occam smooth vs Sparse "
                   "(LWDFDEMInversion sparse flag)"),
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "B_inversion_compare.png", dpi=200)
    print("  平滑回收:", np.round(rec_smooth, 4), " 稀疏回收:", np.round(rec_sparse, 4))


def main():
    apply_rcparams(LANG)
    print("=" * 58)
    print("  水平井进阶：源类型对比 + 反演正则化对比")
    print("=" * 58)
    print("[图 A] 二次场 vs 总场对比 ...")
    fig_A_field_compare()
    print("[图 B] 反演正则化对比 (Occam vs Sparse) ...")
    fig_B_inversion_compare()
    print(f"\n图版已保存至: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
