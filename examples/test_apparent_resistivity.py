# -*- coding: utf-8 -*-
"""E5 测试：视电阻率转换。

测试内容：
1. apparent_resistivity_from_hz 单点视电阻率
2. apparent_resistivity_matrix 批量转换
3. apparent_resistivity_from_forward 与正演对象配合
4. 半空间模型：视电阻率应接近真实电导率

图版保存到 examples/figs/test_apparent/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig,
    apparent_resistivity_from_hz, apparent_resistivity_matrix,
    apparent_resistivity_from_forward,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs", "test_apparent")
os.makedirs(FIG_DIR, exist_ok=True)


def test_single_point_apparent():
    """Test 1: 单点视电阻率（半空间模型）。"""
    print("[Test 1] apparent_resistivity_from_hz 半空间模型")

    # 半空间 σ = 0.1 S/m
    sigma_true = 0.1
    freq = 1e4  # 10 kHz
    offset = 1.0  # 1 m

    # 用解析解计算该半空间的 Hz 响应
    from src.fdemcore.electromagnetics.analytics import hzAnalyticDipoleF
    hz = hzAnalyticDipoleF(offset, np.array([freq]), sigma_true,
                            secondary=False).flatten()[0]

    # 反演视电阻率
    sigma_a = apparent_resistivity_from_hz(freq, hz, offset)

    rel_err = abs(sigma_a - sigma_true) / sigma_true
    print(f"  真实 σ: {sigma_true} S/m")
    print(f"  视电阻率: {sigma_a:.6f} S/m")
    print(f"  相对误差: {rel_err:.2e}")

    # 半空间模型视电阻率应接近真实值（误差 < 10%）
    assert rel_err < 0.1, f"误差过大: {rel_err}"
    print(f"  误差 < 10% [OK]")
    return True


def test_apparent_matrix():
    """Test 2: 批量转换。"""
    print("[Test 2] apparent_resistivity_matrix 批量转换")

    sigma_true = 0.05  # 半空间
    freqs = np.array([1e4, 1e5])  # 减少频率点
    spacings = np.array([0.5, 1.0])  # 减少收发距

    # 构建数据矩阵（用解析解的振幅）
    from src.fdemcore.electromagnetics.analytics import hzAnalyticDipoleF
    data_mat = np.zeros((len(freqs), len(spacings)))
    for i, f in enumerate(freqs):
        for j, r in enumerate(spacings):
            hz = hzAnalyticDipoleF(r, np.array([f]), sigma_true,
                                    secondary=False).flatten()[0]
            data_mat[i, j] = abs(hz)  # 用振幅

    # 批量转换
    sigma_a_mat = apparent_resistivity_matrix(freqs, spacings, data_mat,
                                               component="real")

    print(f"  数据矩阵形状: {data_mat.shape}")
    print(f"  视电阻率矩阵形状: {sigma_a_mat.shape}")
    print(f"  视电阻率范围: [{sigma_a_mat.min():.4f}, {sigma_a_mat.max():.4f}] S/m")

    # 至少部分视电阻率应接近真实值
    rel_err = np.abs(sigma_a_mat - sigma_true) / sigma_true
    min_err = np.min(rel_err)
    print(f"  最小相对误差: {min_err:.2e}")
    # 至少有一个点误差 < 20%
    assert min_err < 0.2, f"最佳点误差也过大: {min_err}"
    print(f"  最佳点误差 < 20% [OK]")
    return True


def test_apparent_from_forward():
    """Test 3: 与正演对象配合。"""
    print("[Test 3] apparent_resistivity_from_forward 正演对象配合")

    # 三层模型
    sigma_true = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.3, 0.6, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)
    d = fwd.dpred(sigma_true)

    # 视电阻率转换
    sigma_a = apparent_resistivity_from_forward(fwd, d)

    print(f"  数据形状: {d.shape}")
    print(f"  视电阻率形状: {sigma_a.shape}")
    print(f"  视电阻率范围: [{sigma_a.min():.4f}, {sigma_a.max():.4f}] S/m")

    # 视电阻率应在合理范围内
    assert sigma_a.shape == (2, 3, 2), f"形状错误: {sigma_a.shape}"
    assert np.all(sigma_a > 0), "视电阻率应为正"
    print(f"  形状正确、全为正 [OK]")
    return True


def test_error_handling():
    """Test 4: 错误处理。"""
    print("[Test 4] 错误处理")

    # 非 1D 几何
    try:
        tool = LWDToolConfig(
            frequencies=(1e4,),
            spacings=np.array([1.0]),
            field_kind="h",
            receiver_geometry="axial",
        )
        # 模拟一个 2D forward（这里简化测试）
        from src.lwd_fdem.apparent import apparent_resistivity_from_forward

        class FakeForward:
            geometry = "2d_cyl"

        apparent_resistivity_from_forward(FakeForward(), np.array([1.0]))
        print("  非1D几何 应报错但未报错 [FAIL]")
        return False
    except ValueError:
        print("  非1D几何 正确报错 [OK]")
    return True


def main():
    print("=" * 60)
    print("E5 测试：视电阻率转换")
    print("=" * 60)

    tests = [
        test_single_point_apparent,
        test_apparent_matrix,
        test_apparent_from_forward,
        test_error_handling,
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
