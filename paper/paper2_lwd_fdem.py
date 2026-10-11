# -*- coding: utf-8 -*-
"""论文2正演复现：Knowledge-guided DL inversion of EM look-ahead LWD (2026)。

期刊: Petroleum Science (2026), DOI 10.1016/j.petsci.2026.04.041

使用本项目 lwd_fdem 框架的 3D TensorMesh FVM 路径（h-form）复现论文中的
前视敏感性实验：
  * 仪器: 单发射 + 三接收 R1/R2/R3，三轴磁偶极子；
  * 收发距: R1=10 m, R2=20 m, R3=30 m；
  * 频率: 2/6/12/24/48/96 kHz（quick 档取 6/24 kHz 代表值）；
  * 模型: 2 层，上层 ρ1=100 Ω·m，下层 ρ2 扫描；界面深度 DTB=5/15/30 m；
  * 测量: 主分量 Hxx/Hyy/Hzz（实/虚部）；
    UHRA = 20*log10|2Vzz/(Vxx+Vyy)|；UHRP = angle[2Vzz/(Vxx+Vyy)]。

关键结论复现: 三个主分量的**虚部**对前视电阻率变化最敏感
（本脚本用归一化敏感度指标 S=(max-min)/rms 定量对比实部与虚部）。

计算量说明:
  正演次数 = n_DTB x n_rho2 x n_freq x n_src(3)。quick 档采用粗网格
  (cell_size=8 m, ~1.3k 单元)、单 DTB、4 个 ρ2、2 个频点，约 1-2 分钟；
  full 档覆盖全部 DTB / ρ2 / 频率，计算量急剧上升，需在高性能机器上运行。

运行:
  python paper/paper2_lwd_fdem.py --quick
  python paper/paper2_lwd_fdem.py --full

输出:
  paper/paper2_lwd_fdem.npz      —— 供 compare_results.py 加载
  paper/figs/paper2_lwd_*.png

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
    normalized_sensitivity, paper2_meta, paper2_sigma, to_complex_tensor,
    uhra_uhrp,
)
from src.lwd_fdem import (  # noqa: E402
    LWDFDEMForward, LWDToolConfig, build_3d_tensor_mesh, build_tensor_survey,
)

logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SolverLU.*")
warnings.filterwarnings("ignore", message=".*splu converted.*")

FIG_DIR = os.path.join(_HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def build_forward(mesh, freqs, spacings):
    """构建 3D 各向同性正演器，并注入三轴源 x 三轴接收的张量 Survey。"""
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
    fwd = LWDFDEMForward("3d", cfg, mesh=mesh, anisotropic=False)
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
    fwd._simulation = None
    return fwd


def run_all(meta, verbose=True):
    """在 DTB x rho2 网格上计算张量响应、UHRA/UHRP。"""
    mesh_spec = meta["mesh"]
    freqs = meta["freqs"]
    dtbs = meta["dtbs"]
    rho2_scan = np.asarray(meta["rho2_scan"], dtype=float)
    spacings = np.asarray(meta["spacings"], dtype=float)

    mesh = build_3d_tensor_mesh(
        core_extent=mesh_spec.core, cell_size=mesh_spec.cell_size,
        n_pad=mesh_spec.n_pad, pad_factor=mesh_spec.pad_factor,
    )
    if verbose:
        print(f"[paper2-lwd] 网格单元数: {mesh.n_cells} "
              f"(cell={mesh_spec.cell_size} m, n_pad={mesh_spec.n_pad})")

    fwd = build_forward(mesh, freqs, spacings)
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
            d = fwd.dpred(sigma)
            hc = to_complex_tensor(d, n_freq, n_sp)   # (n_freq,3,n_sp,3)
            h_all[di, ri] = hc
            for si in range(n_sp):
                uhra, uhrp = uhra_uhrp(hc, spacing_idx=si)
                uhra_all[di, ri, :, si] = uhra
                uhrp_all[di, ri, :, si] = uhrp
            if verbose:
                print(f"[paper2-lwd] DTB={dtb:4.1f} m  rho2={rho2:8.1f} "
                      f"用时 {time.time()-t0:5.1f} s  "
                      f"|Hzz(R1)|={np.abs(hc[0, 2, 0, 2]):.3e}")

    # 统一存储形状: (n_dtb, n_rho2, n_freq, n_sp, 3src, 3rx)
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
    print(f"[paper2-lwd] 结果已保存: {path}")


def _diag(h_store, di, si):
    """从存储张量取某 DTB / 某收发距的主对角分量 Hxx/Hyy/Hzz。

    h_store 形状 (n_dtb, n_rho2, n_freq, n_sp, 3src, 3rx)。
    返回 dict，每个值为 (n_rho2, n_freq) 复数数组。
    """
    sub = h_store[di, :, :, si, :, :]      # (n_rho2, n_freq, 3, 3)
    return {
        "Hxx": sub[:, :, 0, 0],
        "Hyy": sub[:, :, 1, 1],
        "Hzz": sub[:, :, 2, 2],
    }


def make_figures(res):
    """绘制主分量随电阻率扫描曲线与虚/实部敏感度对比。"""
    rho2 = res["rho2_scan"]
    freqs = res["freqs"]
    h_store = res["h_complex"]
    di, si = 0, 0   # 首个 DTB、近接收器 R1

    diag = _diag(h_store, di, si)

    # 图1: Hxx/Hyy/Hzz 实部与虚部随 rho2 变化（每个频率一列）
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
    fig.suptitle(f"Paper2 (lwd_fdem): main components vs $\\rho_2$ "
                 f"(DTB={res['dtbs'][di]:.0f} m, R1={res['spacings'][si]:.0f} m)",
                 fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "paper2_lwd_components.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper2-lwd] 图件已保存: {p1}")

    # 图2: 归一化敏感度柱状图（实部 vs 虚部）
    names = ["Hxx", "Hyy", "Hzz"]
    sens_real = []
    sens_imag = []
    for fi in range(n_freq):
        for name in names:
            v = diag[name][:, fi]
            sens_real.append(normalized_sensitivity(v.real))
            sens_imag.append(normalized_sensitivity(v.imag))
    x = np.arange(len(sens_real))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, sens_real, 0.4, label="Real part")
    ax.bar(x + 0.2, sens_imag, 0.4, label="Imag part")
    tick_labels = [f"{n}\n{freqs[fi]/1e3:.0f}k"
                   for fi in range(n_freq) for n in names]
    ax.set_xticks(x, tick_labels, fontsize=7)
    ax.set_ylabel("Normalized sensitivity  (max-min)/rms")
    ax.set_title("Paper2 (lwd_fdem): imaginary part is more sensitive "
                 "(key conclusion)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "paper2_lwd_sensitivity.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper2-lwd] 图件已保存: {p2}")

    # 控制台打印敏感度对比结论
    print("[paper2-lwd] 归一化敏感度（实部 / 虚部）:")
    k = 0
    for fi in range(n_freq):
        for name in names:
            print(f"    {freqs[fi]/1e3:5.0f} kHz {name}: "
                  f"real={sens_real[k]:.3f}  imag={sens_imag[k]:.3f}")
            k += 1


def main():
    parser = argparse.ArgumentParser(description="Paper2 sensitivity reproduction (lwd_fdem)")
    parser.add_argument("--quick", action="store_true", help="快速演示档（默认）")
    parser.add_argument("--full", action="store_true", help="完整复现档（计算量大）")
    args = parser.parse_args()
    quick = not args.full

    meta = paper2_meta(quick)
    mode = "QUICK" if quick else "FULL"
    print("=" * 64)
    print(f"论文2 敏感性复现 (lwd_fdem) —— {mode} 档")
    print(f"  频率: {[f/1e3 for f in meta['freqs']]} kHz")
    print(f"  收发距: {np.asarray(meta['spacings']).tolist()} m")
    print(f"  DTB: {list(meta['dtbs'])} m   rho2 扫描: "
          f"{np.asarray(meta['rho2_scan']).tolist()} Ohm·m")
    print("=" * 64)

    t0 = time.time()
    res = run_all(meta)
    print(f"[paper2-lwd] 正演总用时: {time.time()-t0:.1f} s")

    save_results(res, os.path.join(_HERE, "paper2_lwd_fdem.npz"))
    make_figures(res)
    print("[paper2-lwd] 完成。")


if __name__ == "__main__":
    main()
