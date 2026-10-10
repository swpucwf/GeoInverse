"""示例：LWD 2D 轴对称（圆柱网格）反演 —— 井旁环状高导异常回收。

演示内容：
  1) 2D 轴对称正演生成含噪合成数据（环状异常体）
  2) LWDFDEMInversion 在 (r,z) 截面参数化下反演（自动规避
     CylindricalMesh 不支持 cell_grads 的问题）
  3) 回收电导率 r-z 切片绘图

运行（仓库根目录）:  python examples/lwd_fdem_inversion_2d.py
输出:  examples/figs/lwd_fdem_inversion_2d.png

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
    add_synthetic_noise, build_2d_cyl_mesh, make_data_object,
)

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    sigma_bg = 0.1
    cfg = LWDToolConfig(
        frequencies=(1e4, 3e4),
        spacings=np.array([1.0, 2.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    mesh = build_2d_cyl_mesh(r_core=5.0, z_core=12.0, cell_size=1.0,
                             n_pad=4, pad_factor=1.4)
    fwd = LWDFDEMForward("2d_cyl", cfg, mesh=mesh)

    # 真实模型：井旁 (r≈3, z≈-3) 环状高导异常
    cc = mesh.cell_centers
    ring = (np.abs(cc[:, 0] - 3.0) < 1.5) & (np.abs(cc[:, 2] + 3.0) < 1.5)
    sigma_true = sigma_bg * np.ones(mesh.n_cells)
    sigma_true[ring] = 0.5
    print(f"环状异常单元数: {int(ring.sum())} / {mesh.n_cells}")

    dobs, std = add_synthetic_noise(fwd.dpred(sigma_true), 0.03, seed=7)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    inv = LWDFDEMInversion(
        fwd, data_obj, m0=np.log(sigma_bg * np.ones(mesh.n_cells)),
        max_iter=5, beta0_ratio=0.1, chifact=0.5,
        sensitivity_weights=False, preconditioner=False,
    )
    rec_sigma, history = inv.run()
    print("迭代数:", len(history["iterations"]),
          "| 末次 phi_d:", round(history["final_phi_d"], 3))
    print(f"环区均值 {rec_sigma[ring].mean():.3f} S/m, "
          f"背景均值 {rec_sigma[~ring].mean():.3f} S/m")

    # ---- SCI 图版：真实/回收剖面 + 收敛史 ----
    from src.lwd_fdem import plot_inversion_figure
    plate = plot_inversion_figure(
        fwd, rec_sigma, history, sigma_true=sigma_true,
        out_dir=FIG_DIR, prefix="lwd_fdem_inversion_2d")
    print("图版:", plate)


if __name__ == "__main__":
    main()
