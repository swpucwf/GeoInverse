# -*- coding: utf-8 -*-
"""两版本（lwd_fdem vs SimPEG）数据对比脚本。

加载 paper1/paper2 两个版本各自保存的 .npz 结果，逐分量对比数值差异，
计算相对误差统计并绘制对比图件。

预期结论: fdemcore 是 SimPEG FDEM 核心的字节级一致精简版，二者在相同网格、
相同源/接收器、相同电导率下应给出**完全一致**的正演结果，相对误差 < 1e-10
（实际常为 0.0，即比特级一致）。

前置条件: 先运行以下脚本生成 .npz（同一档位，quick 或 full 需一致）:
  python paper/paper1_lwd_fdem.py --quick
  python paper/paper1_simpeg.py   --quick
  python paper/paper2_lwd_fdem.py --quick
  python paper/paper2_simpeg.py   --quick

运行:
  python paper/compare_results.py

输出:
  paper/figs/compare_paper1.png
  paper/figs/compare_paper2.png

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(_HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# 相对误差判定阈值（fdemcore 与 SimPEG 应比特级一致）
REL_TOL = 1e-10


def _load(name):
    """加载 .npz，缺失时返回 None 并提示。"""
    path = os.path.join(_HERE, name)
    if not os.path.exists(path):
        print(f"  [缺失] {path} —— 请先运行对应脚本生成结果。")
        return None
    return np.load(path)


def _rel_err(a, b):
    """计算最大绝对差与相对误差（以 b 的最大幅值为标度）。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return np.inf, np.inf, f"shape mismatch {a.shape} vs {b.shape}"
    abs_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    rel = abs_diff / scale if scale > 0 else abs_diff
    return abs_diff, rel, "ok"


def compare_pair(tag, keys, lwd_name, sim_name, results):
    """对比一组 .npz 中若干关键字段，追加统计到 results。"""
    print(f"\n[{tag}] 加载 {lwd_name} 与 {sim_name}")
    a = _load(lwd_name)
    b = _load(sim_name)
    if a is None or b is None:
        results.append((tag, "—", np.inf, "MISSING", False))
        return None, None
    for key in keys:
        if key not in a or key not in b:
            print(f"  跳过字段 {key}（不存在）")
            continue
        abs_diff, rel, note = _rel_err(a[key], b[key])
        ok = rel < REL_TOL and note == "ok"
        print(f"  {key:12s} max|diff|={abs_diff:.3e}  rel_err={rel:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
        results.append((tag, key, rel, f"{abs_diff:.3e}", ok))
    return a, b


def plot_paper1(a, b):
    """绘制论文1 衰减比 lwd vs SimPEG 对比图。"""
    if a is None or b is None:
        return
    att_l = a["att"].ravel()
    att_s = b["att"].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    ax.scatter(att_s, att_l, s=26, c="C0", alpha=0.8, label="data points")
    lo = min(att_s.min(), att_l.min())
    hi = max(att_s.max(), att_l.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (exact)")
    ax.set_xlabel("SimPEG  Att$_{ij}$ (dB)")
    ax.set_ylabel("lwd_fdem  Att$_{ij}$ (dB)")
    ax.set_title("Paper1: attenuation ratio, lwd_fdem vs SimPEG")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = np.abs(att_l - att_s)
    ax.hist(diff, bins=30, color="C3", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("|Att(lwd) - Att(SimPEG)| (dB)")
    ax.set_ylabel("count")
    ax.set_title(f"Paper1: abs difference (max={diff.max():.2e})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(FIG_DIR, "compare_paper1.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] 图件已保存: {p}")


def plot_paper2(a, b):
    """绘制论文2 主分量 lwd vs SimPEG 对比图。"""
    if a is None or b is None:
        return
    rho2 = a["rho2_scan"]
    hl = a["h_complex"]   # (n_dtb, n_rho2, n_freq, n_sp, 3, 3)
    hs = b["h_complex"]
    # 取首 DTB / 首收发距 / Hzz 主分量
    hzz_l = hl[0, :, :, 0, 2, 2]
    hzz_s = hs[0, :, :, 0, 2, 2]
    freqs = a["freqs"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for fi in range(len(freqs)):
        axes[0].semilogx(rho2, hzz_l[:, fi].imag, "-o", ms=4,
                         label=f"lwd {freqs[fi]/1e3:.0f}k imag")
        axes[0].semilogx(rho2, hzz_s[:, fi].imag, "--x", ms=5,
                         label=f"SimPEG {freqs[fi]/1e3:.0f}k imag")
        axes[1].semilogx(rho2, hzz_l[:, fi].real, "-o", ms=4,
                         label=f"lwd {freqs[fi]/1e3:.0f}k real")
        axes[1].semilogx(rho2, hzz_s[:, fi].real, "--x", ms=5,
                         label=f"SimPEG {freqs[fi]/1e3:.0f}k real")
    axes[0].set_title("Paper2: Imag(Hzz) vs $\\rho_2$ (overlap = identical)")
    axes[1].set_title("Paper2: Real(Hzz) vs $\\rho_2$")
    for ax in axes:
        ax.set_xlabel(r"$\rho_2$ ($\Omega\cdot$m)")
        ax.set_ylabel("H (A/m)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "compare_paper2.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] 图件已保存: {p}")


def main():
    print("=" * 68)
    print("lwd_fdem vs SimPEG 正演结果对比")
    print(f"相对误差阈值 REL_TOL = {REL_TOL:.0e}")
    print("=" * 68)

    results = []
    a1, b1 = compare_pair("Paper1", ("h_complex", "att"),
                          "paper1_lwd_fdem.npz", "paper1_simpeg.npz", results)
    a2, b2 = compare_pair("Paper2", ("h_complex", "uhra", "uhrp"),
                          "paper2_lwd_fdem.npz", "paper2_simpeg.npz", results)

    plot_paper1(a1, b1)
    plot_paper2(a2, b2)

    # 汇总
    print("\n" + "=" * 68)
    print("对比汇总")
    print("-" * 68)
    print(f"{'case':8s} {'field':12s} {'rel_err':>12s} {'max|diff|':>12s}  result")
    all_ok = True
    for tag, key, rel, absd, ok in results:
        rel_s = "MISSING" if rel == np.inf else f"{rel:.2e}"
        print(f"{tag:8s} {key:12s} {rel_s:>12s} {absd:>12s}  "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print("-" * 68)
    print(f"总体判定: {'ALL PASS —— 两版本数值一致（fdemcore ≡ SimPEG）' if all_ok else 'FAIL'}")
    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
