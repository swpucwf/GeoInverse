# -*- coding: utf-8 -*-
"""论文1正演复现（SimPEG 版本）：Multi-task U-net look-ahead LWD (Zhang et al., 2026)。

直接使用 external/simpeg-main 的底层 API 复现论文1 的多分量感应测井正演与
衰减比计算，**不经过 lwd_fdem 业务层**，用于与 paper1_lwd_fdem.py 的结果做
逐分量数值对比（预期二者一致到机器精度，因 fdemcore 与 SimPEG FDEM 核心
字节级一致）。

物理模型与 paper1_lwd_fdem.py 完全相同（由 _common.py 统一定义）：
  * 5 层各向异性 VTI 地层；TR1=10 m, TR2=14 m；
  * 频率 10/20/30/50 kHz（quick 档取 10/30 kHz）；
  * 三轴磁偶极子源 x 三轴磁场接收器（9 分量）；
  * Att_ij = -20*log10(|H_R1_ij| / |H_R2_ij|)。

与 lwd_fdem 版本的对应关系：
  build_3d_tensor_mesh        -> discretize.TensorMesh（相同 h 列表, origin=CCC）
  build_tensor_survey         -> 手动构建 MagDipole 源 + PointMagneticField 接收器
  _AnisotropicSigmaMap        -> 本地 _AniSigmaMap（flat -> (n_cells, 3)）
  Simulation3DMagneticField   -> simpeg ... Simulation3DMagneticField（h-form）

运行:
  python paper/paper1_simpeg.py --quick
  python paper/paper1_simpeg.py --full

输出:
  paper/paper1_simpeg.npz      —— 供 compare_results.py 加载
  paper/figs/paper1_simpeg_*.png

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import argparse
import logging
import os
import sys
import time
import types
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SIMPEG = os.path.join(_ROOT, "external", "simpeg-main")
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMPEG)

# external/simpeg-main 归档缺少构建期生成的 version.py，注入占位模块
_vmod = types.ModuleType("simpeg.version")
_vmod.__version__ = "0.25.1"
sys.modules.setdefault("simpeg.version", _vmod)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discretize import TensorMesh  # noqa: E402

from _common import (  # noqa: E402
    PAPER1_SPACINGS, attenuation_ratio, paper1_meta,
    paper1_sigma_tensor, to_complex_tensor,
)

from simpeg import maps as smaps  # noqa: E402
from simpeg.electromagnetics import frequency_domain as fdem  # noqa: E402

logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SolverLU.*")
warnings.filterwarnings("ignore", message=".*splu converted.*")

FIG_DIR = os.path.join(_HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


class _AniSigmaMap(smaps.IdentityMap):
    """各向异性电导率映射：flat (n_cells*3,) -> (n_cells, 3)。

    与 lwd_fdem.forward._AnisotropicSigmaMap 行为一致，按单元 C 序解释
    [σxx, σyy, σzz]，供 discretize 的 get_face_inner_product 识别为对角各向异性。
    """

    def _transform(self, m):
        return np.asarray(m, dtype=float).reshape(-1, 3)

    @property
    def shape_output(self):
        return (self.nP // 3, 3)


def build_mesh(mesh_spec):
    """按 MeshSpec 构建与 lwd_fdem.build_3d_tensor_mesh 完全一致的 TensorMesh。"""
    hx, hy, hz = mesh_spec.h_arrays()
    return TensorMesh([hx, hy, hz], origin="CCC")


def build_tensor_survey(freqs, spacings):
    """手动构建三轴源 x 三轴接收的张量 Survey（h-form 总磁场）。

    数据排列：频率 -> 源取向(x,y,z) -> 收发距 -> 接收取向(x,y,z) -> 分量(real,imag)，
    与 lwd_fdem.tensor.build_tensor_survey 完全一致。
    """
    rx_list = []
    for sp in np.asarray(spacings, dtype=float):
        loc = np.array([0.0, 0.0, float(sp)])
        for rx_ori in ("x", "y", "z"):
            for comp in ("real", "imag"):
                rx_list.append(
                    fdem.receivers.PointMagneticField(
                        loc, orientation=rx_ori, component=comp, data_type="field"
                    )
                )
    src_list = []
    for f in freqs:
        for src_ori in ("x", "y", "z"):
            src_list.append(
                fdem.sources.MagDipole(
                    rx_list, frequency=float(f),
                    location=np.array([0.0, 0.0, 0.0]),
                    orientation=src_ori, moment=1.0,
                )
            )
    return fdem.Survey(src_list)


def run_all(meta, verbose=True):
    """计算全部界面平移的张量响应与衰减比（SimPEG 路径）。"""
    mesh_spec = meta["mesh"]
    freqs = meta["freqs"]
    shifts = meta["shifts"]

    mesh = build_mesh(mesh_spec)
    if verbose:
        print(f"[paper1-simpeg] 网格单元数: {mesh.n_cells} "
              f"(cell={mesh_spec.cell_size} m, n_pad={mesh_spec.n_pad})")

    survey = build_tensor_survey(freqs, PAPER1_SPACINGS)
    sigma_map = _AniSigmaMap(nP=int(mesh.n_cells) * 3)
    sim = fdem.Simulation3DMagneticField(mesh=mesh, survey=survey, sigmaMap=sigma_map)

    z_centers = mesh.cell_centers[:, 2]
    n_freq = len(freqs)

    h_all = []
    att_all = []
    for shift in shifts:
        t0 = time.time()
        sigma = paper1_sigma_tensor(z_centers, shift)
        d = sim.dpred(sigma.ravel())
        hc = to_complex_tensor(d, n_freq, len(PAPER1_SPACINGS))
        att = attenuation_ratio(hc)
        h_all.append(hc)
        att_all.append(att)
        if verbose:
            print(f"[paper1-simpeg] shift={shift:5.1f} m  用时 {time.time()-t0:6.1f} s  "
                  f"Att_zz(f)={np.round(att[:, 2, 2], 3)}")

    return {
        "freqs": np.asarray(freqs, dtype=float),
        "spacings": PAPER1_SPACINGS.astype(float),
        "shifts": np.asarray(shifts, dtype=float),
        "h_complex": np.asarray(h_all),
        "att": np.asarray(att_all),
    }


def save_results(res, path):
    """保存结果到 .npz。"""
    np.savez(path, **res)
    print(f"[paper1-simpeg] 结果已保存: {path}")


def make_figures(res):
    """绘制衰减比热图（与 lwd_fdem 版对应）。"""
    freqs = res["freqs"]
    att = res["att"]
    labels = ["x", "y", "z"]
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
                ax.text(c, r, f"{a0[i, r, c]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper1 (SimPEG): attenuation ratio tensor Att$_{ij}$", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "paper1_simpeg_attenuation.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper1-simpeg] 图件已保存: {p1}")


def main():
    parser = argparse.ArgumentParser(description="Paper1 forward reproduction (SimPEG)")
    parser.add_argument("--quick", action="store_true", help="快速演示档（默认）")
    parser.add_argument("--full", action="store_true", help="完整复现档（计算量大）")
    args = parser.parse_args()
    quick = not args.full

    meta = paper1_meta(quick)
    mode = "QUICK" if quick else "FULL"
    print("=" * 64)
    print(f"论文1 正演复现 (SimPEG) —— {mode} 档")
    print(f"  频率: {[f/1e3 for f in meta['freqs']]} kHz")
    print(f"  收发距: {PAPER1_SPACINGS.tolist()} m")
    print(f"  界面平移: {list(meta['shifts'])} m")
    print("=" * 64)

    t0 = time.time()
    res = run_all(meta)
    print(f"[paper1-simpeg] 正演总用时: {time.time()-t0:.1f} s")

    save_results(res, os.path.join(_HERE, "paper1_simpeg.npz"))
    make_figures(res)
    print("[paper1-simpeg] 完成。")


if __name__ == "__main__":
    main()
