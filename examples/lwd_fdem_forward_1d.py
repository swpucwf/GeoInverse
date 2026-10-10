"""示例：LWD 1D 层状介质频域正演（多频率 × 多收发距）。

演示内容：
  1) LWDToolConfig 定义仪器（频率、收发距、real/imag 分量）
  2) LWDFDEMForward('1d') 计算清洁响应
  3) 与均匀半空间准静态解析解对标（Ward & Hohmann 1988, Eq 4.56）
  4) 响应-收发距曲线与幅相曲线绘图

运行（仓库根目录）:  python examples/lwd_fdem_forward_1d.py
输出:  examples/figs/lwd_fdem_forward_1d.png

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from src.lwd_fdem import LWDFDEMForward, LWDToolConfig, set_cjk_rcparams, case_fig_dir

# 中文字体 + 数学负号统一配置（复用项目级公共工具）
set_cjk_rcparams()

FIG_DIR = case_fig_dir(__file__)   # 按案例隔离输出到 figs/lwd_fdem_forward_1d/


def hz_analytic_halfspace(r, freqs, sigma):
    """均匀全空间垂直磁偶极子二次场 Hz（准静态，moment=1 A·m^2）。"""
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    k = np.sqrt(-1j * w * mu_0 * sigma)
    hz = (1.0 / (2.0 * np.pi * k**2 * r**5)
          * (9.0 - (9.0 + 9j * k * r - 4.0 * (k * r) ** 2
                    - 1j * (k * r) ** 3) * np.exp(-1j * k * r)))
    return hz - (-1.0 / (4.0 * np.pi * r**3))


def main():
    # ---- 1) 仪器配置：4 个频率、3 个收发距、记录 Hz 二次场实/虚部 ----
    cfg = LWDToolConfig(
        frequencies=(1e4, 1e5, 5e5, 1e6),
        spacings=np.array([0.3, 0.6, 1.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",   # 1D 层状必须径向偏移
    )
    # 三层地电模型：2 m / 4 m 层 + 半空间
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=np.array([2.0, 4.0]))
    sigma_model = np.array([0.25, 0.01, 0.5])

    # ---- 2) 正演 ----
    d = fwd.dpred(sigma_model)
    mat = fwd.to_matrix(d)   # (n_freq, n_spacing, n_comp)
    print("数据形状:", mat.shape, " 总道数:", d.size)
    print("响应矩阵 (real, A/m):")
    for i, f in enumerate(cfg.frequencies):
        print(f"  {f:>9.0f} Hz: " + "  ".join(f"{v:+.3e}" for v in mat[i, :, 0]))

    # ---- 3) 半空间退化对标（三层同电导率 -> 全空间解析解） ----
    d_half = fwd.to_matrix(fwd.dpred(0.1 * np.ones(3)))
    hz = hz_analytic_halfspace(1.0, cfg.frequencies, 0.1)
    err_r = np.max(np.abs(d_half[:, 2, 0] - hz.real) / np.abs(hz.real))
    err_i = np.max(np.abs(d_half[:, 2, 1] - hz.imag) / np.abs(hz.imag))
    print(f"\n半空间对标最大相对误差: real {err_r:.2e}, imag {err_i:.2e} (tol 1e-3)")

    # ---- 4) 绘图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, f in enumerate(cfg.frequencies):
        ax1.semilogx(cfg.spacings, np.abs(mat[i, :, 0]), "o-", label=f"{f/1e3:.0f} kHz real")
        ax1.semilogx(cfg.spacings, np.abs(mat[i, :, 1]), "s--", label=f"{f/1e3:.0f} kHz imag")
    ax1.set(xlabel="收发距 (m)", ylabel=r"|Hz secondary| (A/m)",
            title="1D 层状响应 (0.25/0.01/0.5 S/m)", xticks=cfg.spacings)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    freqs = np.logspace(3, 6.5, 60)
    hz_cont = hz_analytic_halfspace(1.0, freqs, 0.1)
    ax2.loglog(freqs, np.abs(hz_cont.real), "C0-", label="解析 real")
    ax2.loglog(freqs, np.abs(hz_cont.imag), "C3-", label="解析 imag")
    ax2.loglog(cfg.frequencies, np.abs(d_half[:, 2, 0]), "C0o", label="SimPEG real")
    ax2.loglog(cfg.frequencies, np.abs(d_half[:, 2, 1]), "C3s", label="SimPEG imag")
    ax2.set(xlabel="频率 (Hz)", ylabel=r"|Hz secondary| (A/m)",
            title=r"半空间对标 (0.1 S/m, r=1 m)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "lwd_fdem_forward_1d.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")

    # ---- 5) SCI 图版（模型 + 响应，多联合成）----
    plate = fwd.plot_figure(sigma_model, d, out_dir=FIG_DIR,
                            prefix="lwd_fdem_forward_1d_plate",
                            labels=("overburden", "resistive", "baseline"))
    print("图版:", plate)


if __name__ == "__main__":
    main()
