# -*- coding: utf-8 -*-
"""论文2正演复现（SimPEG 版本）：Knowledge-guided DL look-ahead LWD (2026)。

直接使用 external/simpeg-main 的底层 API 复现论文2 的前视敏感性实验，
**不经过 lwd_fdem 业务层**，用于与 paper2_lwd_fdem.py 的结果做逐分量对比
（预期一致到机器精度）。

物理模型与 paper2_lwd_fdem.py 完全相同（由 _common.py 统一定义）：
  * 2 层各向同性模型，上层 ρ1=100 Ω·m，下层 ρ2 扫描，DTB=5/15/30 m；
  * 收发距 R1/R2/R3=10/20/30 m；频率 2/6/12/24/48/96 kHz（quick 取 6/24 kHz）；
  * 三轴磁偶极子源 x 三轴磁场接收器，主分量 Hxx/Hyy/Hzz；
  * UHRA=20log10|2Vzz/(Vxx+Vyy)|，UHRP=angle[...]。

与 lwd_fdem 版本的对应关系：
  build_3d_tensor_mesh -> discretize.TensorMesh（相同 h 列表, origin=CCC）
  build_tensor_survey  -> 手动 MagDipole 源 + PointMagneticField 接收器
  IdentityMap          -> simpeg.maps.IdentityMap（各向同性, nP=n_cells）

运行:
  python paper/paper2_simpeg.py --quick
  python paper/paper2_simpeg.py --full

输出:
  paper/paper2_simpeg.npz
  paper/figs/paper2_simpeg_*.png

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

_vmod = types.ModuleType("simpeg.version")
_vmod.__version__ = "0.25.1"
sys.modules.setdefault("simpeg.version", _vmod)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discretize import TensorMesh  # noqa: E402

from _common import (  # noqa: E402
    normalized_sensitivity, paper2_meta, paper2_sigma, to_complex_tensor,
    uhra_uhrp,
)

from simpeg import maps as smaps  # noqa: E402
from simpeg.electromagnetics import frequency_domain as fdem  # noqa: E402

logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SolverLU.*")
warnings.filterwarnings("ignore", message=".*splu converted.*")

FIG_DIR = os.path.join(_HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def build_mesh(mesh_spec):
    """按 MeshSpec 构建与 lwd_fdem.build_3d_tensor_mesh 一致的 TensorMesh。"""
    hx, hy, hz = mesh_spec.h_arrays()
    return TensorMesh([hx, hy, hz], origin="CCC")


def build_tensor_survey(freqs, spacings):
    """手动构建三轴源 x 三轴接收张量 Survey（h-form 总磁场）。"""
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
    """在 DTB x rho2 网格上计算张量响应、UHRA/UHRP（SimPEG 路径）。"""
    mesh_spec = meta["mesh"]
    freqs = meta["freqs"]
    dtbs = meta["dtbs"]
    rho2_scan = np.asarray(meta["rho2_scan"], dtype=float)
    spacings = np.asarray(meta["spacings"], dtype=float)

    mesh = build_mesh(mesh_spec)
    if verbose:
        print(f"[paper2-simpeg] 网格单元数: {mesh.n_cells} "
              f"(cell={mesh_spec.cell_size} m, n_pad={mesh_spec.n_pad})")

    survey = build_tensor_survey(freqs, spacings)
    sigma_map = smaps.IdentityMap(nP=int(mesh.n_cells))
    sim = fdem.Simulation3DMagneticField(mesh=mesh, survey=survey, sigmaMap=sigma_map)

    z_centers = mesh.cell_centers[:, 2]
    n_freq = len(freqs)
    n_sp = len(spacings)

    h_all = np.zeros((len(dtbs), rho2_scan.size, n_freq, 3, n_sp, 3), dtype=complex)
    uhra_all = np.zeros((len(dtbs), rho2_scan.size, n_freq, n_sp))
    uhrp_all = np.zeros((len(dtbs), rho2_scan.size, n_freq, n_sp))

    for di, dtb in enumerate(dtbs):
        for ri, rho2 in enumerate(rho2_scan):
            t0 = time.time()
            sigma = paper2_sigma(z_centers, dtb, rho2)
            d = sim.dpred(sigma)
            hc = to_complex_tensor(d, n_freq, n_sp)
            h_all[di, ri] = hc
            for si in range(n_sp):
                uhra, uhrp = uhra_uhrp(hc, spacing_idx=si)
                uhra_all[di, ri, :, si] = uhra
                uhrp_all[di, ri, :, si] = uhrp
            if verbose:
                print(f"[paper2-simpeg] DTB={dtb:4.1f} m  rho2={rho2:8.1f} "
                      f"用时 {time.time()-t0:5.1f} s  "
                      f"|Hzz(R1)|={np.abs(hc[0, 2, 0, 2]):.3e}")

    h_store = np.transpose(h_all, (0, 1, 2, 4, 3, 5))
    return {
        "freqs": np.asarray(freqs, dtype=float),
        "spacings": spacings.astype(float),
        "dtbs": np.asarray(dtbs, dtype=float),
        "rho2_scan": rho2_scan.astype(float),
        "h_complex": h_store,
        "uhra": uhra_all,
        "uhrp": uhrp_all,
    }


def save_results(res, path):
    """保存结果到 .npz。"""
    np.savez(path, **res)
    print(f"[paper2-simpeg] 结果已保存: {path}")


def make_figures(res):
    """绘制主分量随电阻率扫描曲线与敏感度对比。"""
    rho2 = res["rho2_scan"]
    freqs = res["freqs"]
    h_store = res["h_complex"]
    di, si = 0, 0
    sub = h_store[di, :, :, si, :, :]
    diag = {"Hxx": sub[:, :, 0, 0], "Hyy": sub[:, :, 1, 1], "Hzz": sub[:, :, 2, 2]}

    n_freq = len(freqs)
    fig, axes = plt.subplots(2, n_freq, figsize=(4.6 * n_freq, 7.0), squeeze=False)
    for fi in range(n_freq):
        for name, color in [("Hxx", "C0"), ("Hyy", "C1"), ("Hzz", "C2")]:
            v = diag[name][:, fi]
            axes[0, fi].semilogx(rho2, v.real, "-o", ms=3, color=color, label=name)
            axes[1, fi].semilogx(rho2, v.imag, "-o", ms=3, color=color, label=name)
        axes[0, fi].set_title(f"{freqs[fi]/1e3:.0f} kHz  Real(H)")
        axes[1, fi].set_title(f"{freqs[fi]/1e3:.0f} kHz  Imag(H)")
        axes[1, fi].set_xlabel(r"$\rho_2$ ($\Omega\cdot$m)")
        for ax in (axes[0, fi], axes[1, fi]):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Real part (A/m)")
    axes[1, 0].set_ylabel("Imag part (A/m)")
    fig.suptitle(f"Paper2 (SimPEG): main components vs $\\rho_2$ "
                 f"(DTB={res['dtbs'][di]:.0f} m, R1={res['spacings'][si]:.0f} m)",
                 fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "paper2_simpeg_components.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper2-simpeg] 图件已保存: {p1}")

    names = ["Hxx", "Hyy", "Hzz"]
    sens_real, sens_imag = [], []
    for fi in range(n_freq):
        for name in names:
            v = diag[name][:, fi]
            sens_real.append(normalized_sensitivity(v.real))
            sens_imag.append(normalized_sensitivity(v.imag))
    x = np.arange(len(sens_real))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, sens_real, 0.4, label="Real part")
    ax.bar(x + 0.2, sens_imag, 0.4, label="Imag part")
    ax.set_xticks(x, [f"{n}\n{freqs[fi]/1e3:.0f}k"
                      for fi in range(n_freq) for n in names], fontsize=7)
    ax.set_ylabel("Normalized sensitivity  (max-min)/rms")
    ax.set_title("Paper2 (SimPEG): imaginary part is more sensitive (key conclusion)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "paper2_simpeg_sensitivity.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper2-simpeg] 图件已保存: {p2}")


def main():
    parser = argparse.ArgumentParser(description="Paper2 sensitivity reproduction (SimPEG)")
    parser.add_argument("--quick", action="store_true", help="快速演示档（默认）")
    parser.add_argument("--full", action="store_true", help="完整复现档（计算量大）")
    args = parser.parse_args()
    quick = not args.full

    meta = paper2_meta(quick)
    mode = "QUICK" if quick else "FULL"
    print("=" * 64)
    print(f"论文2 敏感性复现 (SimPEG) —— {mode} 档")
    print(f"  频率: {[f/1e3 for f in meta['freqs']]} kHz")
    print(f"  收发距: {np.asarray(meta['spacings']).tolist()} m")
    print(f"  DTB: {list(meta['dtbs'])} m   rho2 扫描: "
          f"{np.asarray(meta['rho2_scan']).tolist()} Ohm·m")
    print("=" * 64)

    t0 = time.time()
    res = run_all(meta)
    print(f"[paper2-simpeg] 正演总用时: {time.time()-t0:.1f} s")

    save_results(res, os.path.join(_HERE, "paper2_simpeg.npz"))
    make_figures(res)
    print("[paper2-simpeg] 完成。")


if __name__ == "__main__":
    main()
