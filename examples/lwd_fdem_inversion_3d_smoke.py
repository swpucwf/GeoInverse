"""示例：LWD 3D 反演冒烟（粗网格、单频点、少量迭代）。

3D 反演成本高（复数不定矩阵、自由度三次方增长、每迭代需多次伴随求解），
本示例仅验证管线端到端可用；生产级 3D 反演需迭代求解器与更优网格，
留待后续性能优化。

运行（仓库根目录）:  python examples/lwd_fdem_inversion_3d_smoke.py
输出:  examples/figs/lwd_fdem_inversion_3d_smoke.png

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    add_synthetic_noise, build_3d_tensor_mesh, make_data_object,
    set_cjk_rcparams,
)

# 中文字体 + 数学负号统一配置（复用项目级公共工具）
set_cjk_rcparams()

FIG_DIR = pathlib.Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    cfg = LWDToolConfig(
        frequencies=(1e4,),
        spacings=np.array([1.5, 3.0]),
        components=("real", "imag"),
        field_kind="b", secondary=True, data_type="field",
        receiver_geometry="axial",
    )
    mesh = build_3d_tensor_mesh(core_extent=(8.0, 8.0, 8.0), cell_size=1.0,
                                n_pad=3, pad_factor=1.3)
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh)
    print(f"3D 网格单元数: {mesh.n_cells}")

    cc = mesh.cell_centers
    sigma_true = 0.1 * np.ones(mesh.n_cells)
    blob = (np.abs(cc[:, 0] - 2.0) < 1.5) & (np.abs(cc[:, 2]) < 1.5)
    sigma_true[blob] = 0.4

    dobs, std = add_synthetic_noise(fwd.dpred(sigma_true), 0.05, seed=3)
    data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)

    inv = LWDFDEMInversion(
        fwd, data_obj, m0=np.log(0.1 * np.ones(mesh.n_cells)),
        max_iter=2, sensitivity_weights=False, preconditioner=False,
    )
    rec_sigma, history = inv.run()
    print("迭代数:", len(history["iterations"]),
          "| 末次 phi_d:", round(history["final_phi_d"], 3))
    print(f"异常区均值 {rec_sigma[blob].mean():.3f} S/m, "
          f"背景均值 {rec_sigma[~blob].mean():.3f} S/m (功能冒烟，不追求定量回收)")

    sel = np.abs(cc[:, 1]) < 0.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    s1 = ax1.scatter(cc[sel, 0], cc[sel, 2], c=sigma_true[sel],
                     cmap="viridis", s=24, marker="s")
    fig.colorbar(s1, ax=ax1, label="sigma (S/m)")
    ax1.set(xlabel="x (m)", ylabel="z (m)", title="真实 (y=0 切片)")
    s2 = ax2.scatter(cc[sel, 0], cc[sel, 2], c=rec_sigma[sel],
                    cmap="viridis", s=24, marker="s")
    fig.colorbar(s2, ax=ax2, label="sigma (S/m)")
    ax2.set(xlabel="x (m)", ylabel="z (m)", title="反演回收 (冒烟)")
    fig.tight_layout()
    out = FIG_DIR / "lwd_fdem_inversion_3d_smoke.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
