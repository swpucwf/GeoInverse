"""示例：LWD 3D 直角网格 FVM 正演冒烟（单频点）。

演示内容：
  1) build_3d_tensor_mesh 构建 core+padding 张量网格
  2) e-form 3D FVM 正演（总电场 E 分量）
  3) 均匀背景 vs 三维高导方块体的响应差异

3D 全量精细模拟成本高，本示例用粗网格演示功能链路；
性能优化（迭代求解器、并行、跨区域求解）为后续迭代项。

运行（仓库根目录）:  python examples/lwd_fdem_forward_3d.py
输出:  examples/figs/lwd_fdem_forward_3d.png

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
    LWDFDEMForward, LWDToolConfig, build_3d_tensor_mesh, skin_depth,
)

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    freq, sigma_bg = 1e4, 0.1
    print(f"趋肤深度 @{freq:.0f} Hz, {sigma_bg} S/m: "
          f"{skin_depth(freq, sigma_bg):.1f} m")

    # ---- 1) 粗网格：核心 8m 立方、1m 单元 + 3 层 padding ----
    mesh = build_3d_tensor_mesh(core_extent=(8.0, 8.0, 8.0), cell_size=1.0,
                                n_pad=3, pad_factor=1.3)
    print(f"3D 网格单元数: {mesh.n_cells}")

    # ---- 2) 观测系统：单频率、轴向偏移、总电场 Ex/Ez 实虚部 ----
    cfg = LWDToolConfig(
        frequencies=(freq,),
        spacings=np.array([1.5, 3.0]),
        components=("real", "imag"),
        field_kind="e", secondary=False, data_type="field",
        receiver_geometry="axial", orientation="x",
    )
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh)   # e-form（field_kind=e 默认）

    # ---- 3) 背景 vs 方块异常 ----
    # 注：轴向偏移 + 与源同轴的 Ex 接收器在均匀背景中总场对称抵消，
    # 背景 real/imag 均为 ~1e-18 的数值噪声；异常体的响应即散射场本身，
    # 因此这里比较绝对幅度而非相对变化。
    d_bg = fwd.to_matrix(fwd.dpred(sigma_bg * np.ones(mesh.n_cells)))

    sigma_blk = sigma_bg * np.ones(mesh.n_cells)
    cc = mesh.cell_centers
    block = ((np.abs(cc[:, 0] - 2.0) < 1.5)
             & (np.abs(cc[:, 1]) < 1.5)
             & (np.abs(cc[:, 2] + 1.0) < 1.5))
    sigma_blk[block] = 0.4
    d_blk = fwd.to_matrix(fwd.dpred(sigma_blk))
    print(f"方块异常单元数: {int(block.sum())}")

    for k, sp in enumerate(cfg.spacings):
        print(f"  收发距 {sp} m: 背景 Ex real/imag {d_bg[0, k, 0]:+.2e} / "
              f"{d_bg[0, k, 1]:+.2e} (数值噪声) | 异常 Ex real/imag "
              f"{d_blk[0, k, 0]:+.3e} / {d_blk[0, k, 1]:+.3e}")

    # ---- 4) 绘图：核心切片电导率 + 响应柱状 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    sel = np.abs(cc[:, 1]) < 0.6
    sc = ax1.scatter(cc[sel, 0], cc[sel, 2], c=sigma_blk[sel],
                     cmap="viridis", s=26, marker="s")
    fig.colorbar(sc, ax=ax1, label="sigma (S/m)")
    ax1.set(xlabel="x (m)", ylabel="z (m)", title="y=0 切片：井旁方块异常")

    x = np.arange(len(cfg.spacings))
    ax2.bar(x - 0.20, np.abs(d_bg[0, :, 0]), 0.18, label="背景 real")
    ax2.bar(x - 0.02, np.abs(d_bg[0, :, 1]), 0.18, label="背景 imag")
    ax2.bar(x + 0.10, np.abs(d_blk[0, :, 0]), 0.18, label="异常 real")
    ax2.bar(x + 0.28, np.abs(d_blk[0, :, 1]), 0.18, label="异常 imag")
    ax2.set_yscale("log")
    ax2.set(xticks=x, xticklabels=[f"{s} m" for s in cfg.spacings],
            xlabel="收发距", ylabel="|Ex| (V/m)",
            title="3D 单频点响应（背景为数值噪声底）")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    out = FIG_DIR / "lwd_fdem_forward_3d.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")

    # ---- 5) 模型图 / 网格图 / 响应图 三件套 ----
    d_blk_full = fwd.dpred(sigma_blk)
    views = fwd.plot_views(sigma_blk, d_blk_full, out_dir=FIG_DIR,
                           prefix="lwd_fdem_forward_3d")
    print("三件套图:", views["paths"])


if __name__ == "__main__":
    main()
