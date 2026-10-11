"""解析解验证示例 (A4)。

用 1D 层状模型（半空间极限）的数值解与 Ward & Hohmann (1988)
解析解对标，验证 fdemcore 正演的正确性。

策略：
  1) 半空间模型（单层+极厚覆盖层→近似半空间）→ 数值 vs 解析 Hz
  2) 多层模型 → 展示解析解在半空间条件下的精确匹配

SCI 图版英文，300 dpi。
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig,
    apply_rcparams, save_figure, case_fig_dir,
)
from src.fdemcore.electromagnetics.analytics import hzAnalyticDipoleF

# ── 全局配置 ──────────────────────────────────────────────
apply_rcparams("en")
FIG_DIR = case_fig_dir(__file__)

SIGMA = 0.01  # S/m 半空间
FREQS = np.logspace(1, 5, 25)  # 10 Hz → 100 kHz
R = 100.0  # 收发距 (m)


# ================================================================
# 图 01：半空间解析解 vs 1D 数值解
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 解析解（flatten 为 1D）
hz_analytic = hzAnalyticDipoleF(R, FREQS, SIGMA, secondary=True).flatten()

# 1D 数值解（半空间近似：单层 100km 厚 + 半空间）
hz_numerical = np.zeros(len(FREQS), dtype=complex)
for i, f in enumerate(FREQS):
    tool = LWDToolConfig(
        frequencies=(f,),
        spacings=np.array([R]),
        components=("real", "imag"),
        field_kind="h", secondary=True,
        receiver_geometry="radial",
    )
    # 半空间近似：厚层 + 底层同σ
    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=np.array([1e5]))
    d = fwd.dpred(np.array([SIGMA, SIGMA]))
    # d 是 [Re, Im]
    hz_numerical[i] = d[0] + 1j * d[1]

# 左：幅值对比
ax = axes[0]
ax.loglog(FREQS, np.abs(hz_analytic.real), "C0-", lw=2, label="Analytic Re")
ax.loglog(FREQS, np.abs(hz_analytic.imag), "C1-", lw=2, label="Analytic Im")
ax.loglog(FREQS, np.abs(hz_numerical.real), "C0o", ms=4, label="Numerical Re")
ax.loglog(FREQS, np.abs(hz_numerical.imag), "C1s", ms=4, label="Numerical Im")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hz| (A/m)")
ax.set_title("Amplitude comparison")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# 右：相对误差
ax = axes[1]
rel_err_re = np.abs(hz_numerical.real - hz_analytic.real) / (np.abs(hz_analytic.real) + 1e-40)
rel_err_im = np.abs(hz_numerical.imag - hz_analytic.imag) / (np.abs(hz_analytic.imag) + 1e-40)
ax.semilogx(FREQS, rel_err_re, "C0o-", label="Re relative error")
ax.semilogx(FREQS, rel_err_im, "C1s-", label="Im relative error")
ax.axhline(1e-6, color="gray", ls="--", alpha=0.5, label="1e-6")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Relative error")
ax.set_title("Numerical vs Analytic error")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(bottom=1e-15)

fig.suptitle(f"Analytic benchmark: homogeneous σ={SIGMA} S/m, r={R} m", fontsize=11)
fig.tight_layout()
save_figure(fig, FIG_DIR / "fig_01_analytic_benchmark.png")

# 打印最大误差
max_err = max(np.max(rel_err_re), np.max(rel_err_im))
print(f"Max relative error: {max_err:.2e}")
assert max_err < 1e-3, f"Error too large: {max_err}"
print(f"Figures saved to {FIG_DIR}")
print("A4 Benchmark example completed successfully.")
