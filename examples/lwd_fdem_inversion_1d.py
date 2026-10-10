"""示例：LWD 1D 层状介质 Occam 风格反演（合成数据回收）。

演示内容：
  1) 由真实三层模型生成含噪合成观测
  2) LWDFDEMInversion 组装反演（对数电导率 + ExpMap + WLS 平滑 +
     beta 冷却 + 目标拟合差停止）
  3) 回收电导率剖面与逐迭代收敛史绘图

运行（仓库根目录）:  python examples/lwd_fdem_inversion_1d.py
输出:  examples/figs/lwd_fdem_inversion_1d.png

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
# 中文标注字体（Windows 自带黑体，避免 CJK Glyph missing 警告）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    add_synthetic_noise, build_1d_mesh, make_data_object,
)

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    # ---- 1) 真实模型与合成数据 ----
    cfg = LWDToolConfig(
        frequencies=(1e3, 1e4, 1e5, 1e6),
        spacings=np.array([0.3, 0.6, 1.0, 2.0]),
        components=("real", "imag"),
        field_kind="h", secondary=True, data_type="field",
        receiver_geometry="radial",
    )
    thicknesses = np.array([2.0, 4.0])
    fwd = LWDFDEMForward("1d", cfg, layer_thicknesses=thicknesses)
    true_sigma = np.array([0.25, 0.01, 0.5])
    dobs, std = add_synthetic_noise(fwd.dpred(true_sigma), 0.02, seed=1)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    # ---- 2) 反演 ----
    inv = LWDFDEMInversion(
        fwd, data_obj, m0=np.log(0.1 * np.ones(3)),
        max_iter=15, beta0_ratio=1e-4, beta_cooling=2.0,
        sensitivity_weights=False, preconditioner=False,
    )
    rec_sigma, history = inv.run()
    print("真实 sigma :", true_sigma)
    print("回收 sigma :", np.round(rec_sigma, 4))
    print("相对误差   :", np.round(np.abs(rec_sigma - true_sigma) / true_sigma, 4))

    # ---- 3) 绘图：左阶梯剖面，右收敛史 ----
    mesh1d = build_1d_mesh(thicknesses)
    zc = np.asarray(mesh1d.cell_centers).reshape(-1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    ax1.step(rec_sigma, zc, where="mid", color="C0", lw=2, label="反演")
    ax1.step(true_sigma, zc, where="mid", color="k", ls="--", lw=1.5, label="真模型")
    ax1.invert_yaxis()
    ax1.set(xlabel="sigma (S/m)", ylabel="z (m)", title="1D 层状反演回收")
    ax1.legend()
    ax1.grid(alpha=0.3)

    it = history["iterations"]
    ax2.semilogy(range(1, len(it) + 1), [x["phi_d"] for x in it], "o-", label="phi_d")
    ax2.semilogy(range(1, len(it) + 1), [x["phi_m"] for x in it], "s-", label="phi_m")
    ax2.set(xlabel="迭代", ylabel="目标函数", title="收敛史")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    out = FIG_DIR / "lwd_fdem_inversion_1d.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
