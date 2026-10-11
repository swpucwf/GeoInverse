# -*- coding: utf-8 -*-
"""论文1正演复现：Multi-task U-net look-ahead LWD (Zhang et al., 2026)。

期刊: Petroleum Science 23 (2026) 1908-1928

使用本项目 lwd_fdem 框架的 3D TensorMesh FVM 路径（h-form）复现论文中的
多分量感应测井正演响应与衰减比计算：
  * 仪器: 单发射 + 双接收，三轴正交线圈磁偶极子；
  * 收发距: TR1=10 m, TR2=14 m；
  * 频率: 10, 20, 30, 50 kHz（quick 档取 10, 30 kHz 代表值）；
  * 地层: 5 层各向异性 VTI（σh, σv）；
  * 测量: 3x3 电压/磁场张量；衰减比 Att_ij = -20*log10(|H_R1_ij| / |H_R2_ij|)。

关键结论复现: yy 分量对各向异性最敏感；同轴(xx,zz)+共面(yy)对电导率最敏感；
交叉(xz,zx)+同轴对界面最敏感（本脚本给出全部 9 分量的 Att 供分析）。

计算量说明:
  3D FVM 正演每个 (频率, 源取向) 需一次稀疏 LU 分解求解。quick 档采用粗网格
  (cell_size=6 m, n_pad=2, ~1.5k 单元) 与 2 个频点、2 个界面平移，秒级—分钟级
  即可完成；full 档 (cell_size=2 m, n_pad=5, ~1e5 单元, 4 频点, 5 平移) 需数十
  分钟至数小时，建议在多核大内存机器上运行。

运行:
  python paper/paper1_lwd_fdem.py --quick
  python paper/paper1_lwd_fdem.py --full

输出:
  paper/paper1_lwd_fdem.npz      —— 复数张量与衰减比，供 compare_results.py 加载
  paper/figs/paper1_*.png        —— 图件

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (  # noqa: E402
    PAPER1_SPACINGS, attenuation_ratio, paper1_meta,
    paper1_sigma_tensor, to_complex_tensor,
)
from src.lwd_fdem import (  # noqa: E402
    LWDFDEMForward, LWDToolConfig, build_3d_tensor_mesh, build_tensor_survey,
)

# 抑制求解器性能提示与稀疏格式转换告警，保持输出整洁
logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SolverLU.*")
warnings.filterwarnings("ignore", message=".*splu converted.*")

FIG_DIR = os.path.join(_HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def build_forward(mesh, freqs, spacings):
    """构建 3D 各向异性正演器，并注入三轴源 x 三轴接收的张量 Survey。

    Args:
        mesh: discretize.TensorMesh。
        freqs: 频率元组 (Hz)。
        spacings: 收发距数组 (m)。

    Returns:
        已注入张量 Survey 的 LWDFDEMForward 实例（h-form）。
    """
    cfg = LWDToolConfig(
        frequencies=tuple(freqs),
        spacings=np.asarray(spacings, dtype=float),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        data_type="field",
        receiver_geometry="axial",
        orientation="z",
    )
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh, anisotropic=True)
    # 用张量 Survey（3 源取向 x 3 接收取向）覆盖默认单取向 Survey
    fwd.survey = build_tensor_survey(
        frequencies=tuple(freqs),
        spacings=np.asarray(spacings, dtype=float),
        field_kind="h",
        secondary=False,
        components=("real", "imag"),
        receiver_geometry="axial",
        rx_orientations=("x", "y", "z"),
        src_orientations=("x", "y", "z"),
        source_location=(0.0, 0.0, 0.0),
        form="h",
    )
    fwd._simulation = None  # 强制按新 Survey 重建仿真
    return fwd


def run_all(meta, verbose=True):
    """在给定配置下计算全部界面平移的张量响应与衰减比。

    Args:
        meta: paper1_meta(quick) 返回的配置字典。
        verbose: 是否打印进度。

    Returns:
        dict，含 freqs/spacings/shifts/h_complex/att。
    """
    mesh_spec = meta["mesh"]
    freqs = meta["freqs"]
    shifts = meta["shifts"]

    mesh = build_3d_tensor_mesh(
        core_extent=mesh_spec.core, cell_size=mesh_spec.cell_size,
        n_pad=mesh_spec.n_pad, pad_factor=mesh_spec.pad_factor,
    )
    if verbose:
        print(f"[paper1-lwd] 网格单元数: {mesh.n_cells} "
              f"(cell={mesh_spec.cell_size} m, n_pad={mesh_spec.n_pad})")

    fwd = build_forward(mesh, freqs, PAPER1_SPACINGS)
    z_centers = mesh.cell_centers[:, 2]
    n_freq = len(freqs)

    h_all = []
    att_all = []
    for k, shift in enumerate(shifts):
        t0 = time.time()
        sigma = paper1_sigma_tensor(z_centers, shift)
        d = fwd.dpred(sigma)
        hc = to_complex_tensor(d, n_freq, len(PAPER1_SPACINGS))
        att = attenuation_ratio(hc)
        h_all.append(hc)
        att_all.append(att)
        if verbose:
            print(f"[paper1-lwd] shift={shift:5.1f} m  用时 {time.time()-t0:6.1f} s  "
                  f"Att_zz(f)={np.round(att[:, 2, 2], 3)}")

    return {
        "freqs": np.asarray(freqs, dtype=float),
        "spacings": PAPER1_SPACINGS.astype(float),
        "shifts": np.asarray(shifts, dtype=float),
        "h_complex": np.asarray(h_all),   # (n_shift, n_freq, 3, n_sp, 3)
        "att": np.asarray(att_all),        # (n_shift, n_freq, 3, 3)
    }


def save_results(res, path):
    """保存结果到 .npz。"""
    np.savez(path, **res)
    print(f"[paper1-lwd] 结果已保存: {path}")


def make_figures(res):
    """绘制衰减比热图与响应-深度曲线。"""
    freqs = res["freqs"]
    att = res["att"]          # (n_shift, n_freq, 3, 3)
    shifts = res["shifts"]
    labels = ["x", "y", "z"]

    # 图1: 每个频率的 3x3 衰减比热图（取首个界面平移）
    n_freq = len(freqs)
    fig, axes = plt.subplots(1, n_freq, figsize=(4.2 * n_freq, 3.8), squeeze=False)
    a0 = att[0]
    vmax = np.nanmax(np.abs(a0))
    for i in range(n_freq):
        ax = axes[0, i]
        im = ax.imshow(a0[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3), [f"Rx-{c}" for c in labels])
        ax.set_yticks(range(3), [f"Tx-{r}" for r in labels])
        ax.set_title(f"{freqs[i]/1e3:.0f} kHz  Att$_{{ij}}$ (dB)")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{a0[i, r, c]:.1f}", ha="center", va="center",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper1 (lwd_fdem): attenuation ratio tensor Att$_{ij}$", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "paper1_lwd_attenuation.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper1-lwd] 图件已保存: {p1}")

    # 图2: 代表性分量 Att 随界面平移（前视深度）变化
    comps = [("yy", 1, 1), ("zz", 2, 2), ("xz", 0, 2), ("zx", 2, 0)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = ["-", "--", "-.", ":"]
    for (name, si, rj), ls in zip(comps, styles):
        for fi, f in enumerate(freqs):
            curve = att[:, fi, si, rj]
            ax.plot(shifts, curve, ls, marker="o", ms=4,
                    label=f"Att_{name} @{f/1e3:.0f}kHz")
    ax.set_xlabel("Interface shift / look-ahead depth (m)")
    ax.set_ylabel("Attenuation ratio Att (dB)")
    ax.set_title("Paper1 (lwd_fdem): Att vs depth for key components")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "paper1_lwd_att_vs_depth.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper1-lwd] 图件已保存: {p2}")


def main():
    parser = argparse.ArgumentParser(description="Paper1 forward reproduction (lwd_fdem)")
    parser.add_argument("--quick", action="store_true", help="快速演示档（默认）")
    parser.add_argument("--full", action="store_true", help="完整复现档（计算量大）")
    args = parser.parse_args()
    quick = not args.full  # 默认 quick，仅显式 --full 时用完整档

    meta = paper1_meta(quick)
    mode = "QUICK" if quick else "FULL"
    print("=" * 64)
    print(f"论文1 正演复现 (lwd_fdem) —— {mode} 档")
    print(f"  频率: {[f/1e3 for f in meta['freqs']]} kHz")
    print(f"  收发距: {PAPER1_SPACINGS.tolist()} m")
    print(f"  界面平移: {list(meta['shifts'])} m")
    print("=" * 64)

    t0 = time.time()
    res = run_all(meta)
    print(f"[paper1-lwd] 正演总用时: {time.time()-t0:.1f} s")

    save_results(res, os.path.join(_HERE, "paper1_lwd_fdem.npz"))
    make_figures(res)
    print("[paper1-lwd] 完成。")


if __name__ == "__main__":
    main()
