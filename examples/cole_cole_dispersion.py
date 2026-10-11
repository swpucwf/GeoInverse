"""Cole-Cole 频散效应示例 (A2)。

对比 1D 层状模型在无色散 vs 不同 Cole-Cole 参数下的电磁响应，
展示频散对幅值/相位的影响规律。

模型：三层地层 [0.1, 0.01, 0.5] S/m，厚度 [5, 10] m。
仪器：3 频率 × 3 收发距 × 2 分量 (real/imag)。

输出：
  fig_01  无色散 vs 有频散响应对比（幅值 + 分量值）
  fig_02  不同 η/τ/c 参数扫描（单变量）

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

# ── 全局配置 ──────────────────────────────────────────────
apply_rcparams("en")
FIG_DIR = case_fig_dir(__file__)

SIGMA = np.array([0.1, 0.01, 0.5])
THICK = np.array([5.0, 10.0])
FREQS = (1e3, 1e4, 1e5)
SPACINGS = np.array([0.5, 1.0, 2.0])

TOOL = LWDToolConfig(
    frequencies=FREQS, spacings=SPACINGS,
    components=("real", "imag"),
    field_kind="h", secondary=True,
    receiver_geometry="radial",
)


def _run_fwd(cole_cole=None):
    fwd = LWDFDEMForward("1d", TOOL, layer_thicknesses=THICK, cole_cole=cole_cole)
    return fwd, fwd.dpred(SIGMA)


# ================================================================
# 图 01：无色散 vs 有频散
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fwd0, d0 = _run_fwd()
fwd1, d1 = _run_fwd(cole_cole=dict(eta=0.3, tau=0.01, c=0.5))

mat0 = fwd0.to_matrix(d0)
mat1 = fwd1.to_matrix(d1)

# 左：幅值
ax = axes[0]
for i, f in enumerate(FREQS):
    ax.semilogx(SPACINGS, np.abs(mat0[i, :, 0]), "o-", color=f"C{i}",
                label=f"{f/1e3:.0f} kHz (no disp)")
    ax.semilogx(SPACINGS, np.abs(mat1[i, :, 0]), "s--", color=f"C{i}",
                label=f"{f/1e3:.0f} kHz (η=0.3)")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("|Re(Hs)|")
ax.set_title("Amplitude")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, which="both", alpha=0.3)

# 右：实部分量
ax = axes[1]
for i, f in enumerate(FREQS):
    ax.semilogx(SPACINGS, mat0[i, :, 0], "o-", color=f"C{i}",
                label=f"{f/1e3:.0f} kHz (no disp)")
    ax.semilogx(SPACINGS, mat1[i, :, 0], "s--", color=f"C{i}",
                label=f"{f/1e3:.0f} kHz (η=0.3)")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("Re(Hs)")
ax.set_title("Real component")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Cole-Cole dispersion effect (η=0.3, τ=0.01 s, c=0.5)", fontsize=11)
fig.tight_layout()
save_figure(fig, FIG_DIR / "fig_01_cole_cole_compare.png")


# ================================================================
# 图 02：参数扫描（单变量）
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 扫描 η
etas = [0.0, 0.1, 0.2, 0.3, 0.5]
ax = axes[0]
for eta in etas:
    _, d = _run_fwd(cole_cole=dict(eta=eta, tau=0.01, c=0.5))
    mat = fwd0.to_matrix(d)
    ax.semilogx(SPACINGS, np.abs(mat[1, :, 0]), "o-", label=f"η={eta}")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("|Re(Hs)| (10 kHz)")
ax.set_title("Varying η (τ=0.01, c=0.5)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# 扫描 τ
taus = [0.001, 0.005, 0.01, 0.05, 0.1]
ax = axes[1]
for tau in taus:
    _, d = _run_fwd(cole_cole=dict(eta=0.3, tau=tau, c=0.5))
    mat = fwd0.to_matrix(d)
    ax.semilogx(SPACINGS, np.abs(mat[1, :, 0]), "o-", label=f"τ={tau}")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("|Re(Hs)| (10 kHz)")
ax.set_title("Varying τ (η=0.3, c=0.5)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# 扫描 c
cs = [0.2, 0.4, 0.5, 0.7, 0.9]
ax = axes[2]
for c in cs:
    _, d = _run_fwd(cole_cole=dict(eta=0.3, tau=0.01, c=c))
    mat = fwd0.to_matrix(d)
    ax.semilogx(SPACINGS, np.abs(mat[1, :, 0]), "o-", label=f"c={c}")
ax.set_xlabel("Spacing (m)")
ax.set_ylabel("|Re(Hs)| (10 kHz)")
ax.set_title("Varying c (η=0.3, τ=0.01)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Cole-Cole parameter sensitivity scan", fontsize=11)
fig.tight_layout()
save_figure(fig, FIG_DIR / "fig_02_cole_cole_scan.png")

print(f"Figures saved to {FIG_DIR}")
print("A2 Cole-Cole example completed successfully.")
