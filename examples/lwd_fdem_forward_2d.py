"""示例：LWD 2D 轴对称（圆柱网格）FVM 正演 + 井旁环状异常响应。

演示内容：
  1) build_2d_cyl_mesh 自动趋肤深度校核建网格
  2) b-form FVM 正演（轴向偏移接收器 + B 二次场）
  3) 均匀背景 vs 井旁高导环的响应差异
  4) 电导率模型 r-z 切片绘图

运行（仓库根目录）:  python examples/lwd_fdem_forward_2d.py
输出:  examples/figs/lwd_fdem_forward_2d.png

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
    LWDFDEMForward, LWDToolConfig,
    build_2d_cyl_mesh, mesh_quality_report, recommend_cell_size,
)

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    frequencies = (1e4, 5e4)
    sigma_bg = 0.1

    # ---- 1) 网格：趋肤深度给出粗化下限，实际按异常尺度取 1 m ----
    cell_skin = recommend_cell_size(frequencies, [sigma_bg, 0.5])
    cell = min(1.0, cell_skin)
    mesh = build_2d_cyl_mesh(r_core=8.0, z_core=30.0, cell_size=cell,
                             n_pad=6, pad_factor=1.3)
    rep = mesh_quality_report(mesh, frequencies, [sigma_bg])
    print(f"趋肤深度建议单元 {cell_skin:.1f} m -> 实际取 {cell:.2f} m, "
          f"网格 {mesh.n_cells} 单元, 校核 ok={rep['ok']:.0f}")

    # ---- 2) 观测系统：轴向偏移（沿井轴），B 二次场实/虚部 ----
    cfg = LWDToolConfig(
        frequencies=frequencies,
        spacings=np.array([1.0, 2.0, 3.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)

    # ---- 3) 均匀背景 vs 环状高导异常 ----
    d_bg = fwd.to_matrix(fwd.dpred(sigma_bg * np.ones(mesh.n_cells)))

    sigma_ring = sigma_bg * np.ones(mesh.n_cells)
    cc = mesh.cell_centers
    ring = (np.abs(cc[:, 0] - 4.0) < 1.5) & (np.abs(cc[:, 2] + 5.0) < 2.0)
    sigma_ring[ring] = 0.5
    d_ring = fwd.to_matrix(fwd.dpred(sigma_ring))
    print(f"环状异常体单元数: {int(ring.sum())}")

    for i, f in enumerate(cfg.frequencies):
        for k, comp in enumerate(cfg.components):
            rel = (d_ring[i, :, k] - d_bg[i, :, k]) / d_bg[i, :, k]
            print(f"  {f:>6.0f} Hz {comp:<4}: 背景 "
                  + " ".join(f"{v:+.3e}" for v in d_bg[i, :, k])
                  + " | 异常相对变化 " + " ".join(f"{v:+.1%}" for v in rel))

    # ---- 4) 绘图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    cc = mesh.cell_centers
    ax1.tripcolor(cc[:, 0], cc[:, 2], sigma_ring, cmap="viridis",
                  shading="flat")
    ax1.set(xlabel="r (m)", ylabel="z (m)", title="模型：井旁高导环 (S/m)")

    x = np.arange(len(cfg.spacings))
    for i, f in enumerate(cfg.frequencies):
        ax2.bar(x + 0.18 * (i - 0.5),
                np.abs(d_ring[i, :, 1] - d_bg[i, :, 1]) / np.abs(d_bg[i, :, 1]),
                0.16, label=f"{f/1e3:.0f} kHz imag")
    ax2.set(xticks=x, xlabel="收发距 (m)",
            ylabel="异常引起的相对变化", title="环状异常探测灵敏度")
    ax2.set_xticklabels([str(s) for s in cfg.spacings])
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    out = FIG_DIR / "lwd_fdem_forward_2d.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")

    # ---- 5) 模型图 / 网格图 / 响应图 三件套 ----
    d_ring_full = fwd.dpred(sigma_ring)
    views = fwd.plot_views(sigma_ring, d_ring_full, out_dir=FIG_DIR,
                           prefix="lwd_fdem_forward_2d")
    print("三件套图:", views["paths"])


if __name__ == "__main__":
    main()
