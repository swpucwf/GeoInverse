# -*- coding: utf-8 -*-
"""E6+E7+E8 测试：LAS导出、并行频率、灵敏度矩阵。

测试内容：
1. export_las2 LAS 2.0 格式导出
2. forward_parallel_frequencies 并行频率计算
3. compute_sensitivity_matrix 灵敏度矩阵
4. compute_sensitivity_diagonal 灵敏度对角元

图版保存到 examples/figs/test_advanced/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig,
    export_las2, forward_parallel_frequencies,
    compute_sensitivity_matrix, compute_sensitivity_diagonal,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs", "test_advanced")
os.makedirs(FIG_DIR, exist_ok=True)


def test_las_export():
    """Test 1: LAS 2.0 格式导出。"""
    print("[Test 1] export_las2 LAS 2.0 导出")

    # 模拟数据
    depths = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    data_dict = {
        "DEPT": depths,
        "RHOL": np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]),
        "GR": np.array([50, 55, 60, 65, 70, 75, 80]),
        "RES": np.array([10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0]),
    }

    filepath = os.path.join(FIG_DIR, "test_export.las")
    result = export_las2(depths, data_dict, filepath, well_name="TEST-WELL")

    # 验证文件存在
    assert os.path.exists(filepath), "LAS 文件未生成"
    print(f"  LAS 文件已生成: {filepath} [OK]")

    # 读取并验证内容
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    assert "TEST-WELL" in content, "井名未写入"
    assert "DEPT" in content, "深度曲线未写入"
    assert "RHOL" in content, "密度曲线未写入"
    assert "GR" in content, "GR 曲线未写入"
    assert "RES" in content, "电阻率曲线未写入"
    print(f"  内容验证: 井名、深度、3条曲线 [OK]")

    # 清理
    os.remove(filepath)
    print(f"  测试文件已清理 [OK]")
    return True


def test_parallel_frequencies():
    """Test 2: 并行频率计算。"""
    print("[Test 2] forward_parallel_frequencies 并行计算")

    # 三层模型
    sigma_true = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e3, 1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",  # 1D 必须用 radial
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    # 串行计算（基准）
    d_serial = fwd.dpred(sigma_true)

    # 并行计算
    d_parallel = forward_parallel_frequencies(fwd, sigma_true, max_workers=2)

    # 验证结果一致
    assert d_parallel.shape == d_serial.shape, f"形状不匹配: {d_parallel.shape} vs {d_serial.shape}"
    max_diff = np.max(np.abs(d_parallel - d_serial))
    print(f"  串行形状: {d_serial.shape}")
    print(f"  并行形状: {d_parallel.shape}")
    print(f"  最大差异: {max_diff:.2e}")

    assert max_diff < 1e-10, f"结果不一致: {max_diff}"
    print(f"  串行/并行结果一致 [OK]")
    return True


def test_sensitivity_matrix():
    """Test 3: 灵敏度矩阵。"""
    print("[Test 3] compute_sensitivity_matrix 灵敏度矩阵")

    sigma = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",  # 1D 必须用 radial
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    # 计算灵敏度矩阵
    J = compute_sensitivity_matrix(fwd, sigma)

    # 验证形状
    nD = fwd.survey.nD
    nP = len(sigma)
    expected_shape = (nD, nP)
    assert J.shape == expected_shape, f"形状错误: {J.shape} vs {expected_shape}"
    print(f"  数据维度 nD: {nD}")
    print(f"  参数维度 nP: {nP}")
    print(f"  J 矩阵形状: {J.shape} [OK]")

    # 验证非零
    assert np.any(J != 0), "J 矩阵全为零"
    print(f"  J 矩阵非零元素: {np.count_nonzero(J)} [OK]")
    return True


def test_sensitivity_diagonal():
    """Test 4: 灵敏度对角元（内存友好版）。"""
    print("[Test 4] compute_sensitivity_diagonal 灵敏度对角元")

    sigma = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",  # 1D 必须用 radial
    )

    fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    # 计算对角元
    diag = compute_sensitivity_diagonal(fwd, sigma)

    # 验证形状
    nP = len(sigma)
    assert diag.shape == (nP,), f"形状错误: {diag.shape}"
    print(f"  对角元形状: {diag.shape} [OK]")

    # 验证非负（JᵀJ 对角元 >= 0）
    assert np.all(diag >= 0), "对角元出现负值"
    print(f"  对角元值: {diag}")
    print(f"  全为非负 [OK]")
    return True


def main():
    print("=" * 60)
    print("E6+E7+E8 测试：LAS导出、并行频率、灵敏度矩阵")
    print("=" * 60)

    tests = [
        test_las_export,
        test_parallel_frequencies,
        test_sensitivity_matrix,
        test_sensitivity_diagonal,
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
