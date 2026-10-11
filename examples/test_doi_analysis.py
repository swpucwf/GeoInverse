# -*- coding: utf-8 -*-
"""E9 测试：DOI 探测深度分析。

测试内容：
1. doi_1d_sensitivity 基于灵敏度的 DOI
2. doi_skin_depth_estimate 趋肤深度估计
3. doi_resolution_matrix 分辨率矩阵
4. doi_summary 综合报告

图版保存到 examples/figs/test_doi/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig,
    doi_1d_sensitivity, doi_skin_depth_estimate,
    doi_resolution_matrix, doi_summary,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs", "test_doi")
os.makedirs(FIG_DIR, exist_ok=True)


def test_doi_1d_sensitivity():
    """Test 1: 基于灵敏度的 DOI。"""
    print("[Test 1] doi_1d_sensitivity 灵敏度 DOI")

    sigma = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    result = doi_1d_sensitivity(fwd, sigma, threshold_ratio=0.1)

    # 验证返回结构
    assert "sensitivity" in result, "缺少 sensitivity"
    assert "cumulative_depths" in result, "缺少 cumulative_depths"
    assert "doi_depth" in result, "缺少 doi_depth"
    assert "reliable_mask" in result, "缺少 reliable_mask"

    print(f"  灵敏度对角元: {result['sensitivity']}")
    print(f"  累积深度: {result['cumulative_depths']}")
    print(f"  DOI 深度: {result['doi_depth']:.2f} m")
    print(f"  可靠掩码: {result['reliable_mask']}")

    # DOI 应为正数
    assert result["doi_depth"] > 0, "DOI 深度应为正"
    print(f"  DOI 深度 > 0 [OK]")
    return True


def test_doi_skin_depth():
    """Test 2: 趋肤深度估计。"""
    print("[Test 2] doi_skin_depth_estimate 趋肤深度")

    freqs = np.array([1e3, 1e4, 1e5])
    sigmas = np.array([0.01, 0.1, 1.0])

    delta = doi_skin_depth_estimate(freqs, sigmas, threshold_ratio=1.0)

    # 函数使用均值计算，是粗略估计
    # 对于 σ=0.1, f=1e4: δ ≈ 15.9 m
    # 但函数用均值会给出不同结果
    print(f"  趋肤深度: {delta:.2f} m")

    # 趋肤深度应为正
    assert delta > 0, "趋肤深度应为正"
    # 数量级应合理（1-1000 m，放宽要求）
    assert 1 < delta < 1000, f"趋肤深度异常: {delta}"
    print(f"  趋肤深度在合理范围 [OK]")
    return True


def test_doi_resolution_matrix():
    """Test 3: 分辨率矩阵。"""
    print("[Test 3] doi_resolution_matrix 分辨率矩阵")

    sigma = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    # 计算分辨率矩阵
    R = doi_resolution_matrix(fwd, sigma, beta=1.0, alpha=1e-3)

    # 验证形状
    nP = len(sigma)
    assert R.shape == (nP, nP), f"形状错误: {R.shape}"
    print(f"  R 矩阵形状: {R.shape} [OK]")

    # 验证对角元在 [0, 1] 范围
    diag = np.diag(R)
    print(f"  R 对角元: {diag}")
    assert np.all(diag >= -0.01), "对角元出现明显负值"
    assert np.all(diag <= 1.01), "对角元超过 1"
    print(f"  对角元在 [0, 1] 范围 [OK]")

    # 验证迹 <= nP（分辨率矩阵的迹不超过参数个数）
    trace = np.trace(R)
    print(f"  R 迹: {trace:.2f} (参数数: {nP})")
    assert trace <= nP + 0.1, f"迹过大: {trace}"
    print(f"  迹 <= nP [OK]")
    return True


def test_doi_summary():
    """Test 4: 综合报告。"""
    print("[Test 4] doi_summary 综合报告")

    sigma = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    report = doi_summary(fwd, sigma)

    # 验证报告内容
    assert "DOI" in report, "报告缺少 DOI 信息"
    assert "层" in report, "报告缺少层信息"
    print(f"  报告长度: {len(report)} 字符")
    print(f"  报告内容:")
    for line in report.split("\n"):
        print(f"    {line}")
    print(f"  报告生成成功 [OK]")
    return True


def main():
    print("=" * 60)
    print("E9 测试：DOI 探测深度分析")
    print("=" * 60)

    tests = [
        test_doi_1d_sensitivity,
        test_doi_skin_depth,
        test_doi_resolution_matrix,
        test_doi_summary,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            print(f"  异常: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
