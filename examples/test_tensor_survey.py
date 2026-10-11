# -*- coding: utf-8 -*-
"""E1+E2 测试：多分量张量接收与多分量源阵列。

测试内容：
1. build_tensor_receivers 构建三轴接收器
2. build_tensor_survey 构建三轴源×三轴接收 Survey
3. tensor_data_order 数据维度计算
4. 与原始单分量 survey 的兼容性
5. 错误处理

图版保存到 examples/figs/test_tensor/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lwd_fdem import (
    build_tensor_receivers, build_tensor_survey, tensor_data_order,
    LWDToolConfig, build_survey,
)

# 图版目录
FIG_DIR = os.path.join(os.path.dirname(__file__), "figs", "test_tensor")
os.makedirs(FIG_DIR, exist_ok=True)


def test_tensor_receivers():
    """Test 1: 三轴接收器构建。"""
    print("[Test 1] build_tensor_receivers 三轴接收")
    rx_list = build_tensor_receivers(
        spacings=np.array([0.5, 1.0]),
        field_kind="h",
        secondary=False,
        components=("real", "imag"),
        rx_orientations=("x", "y", "z"),
        form="h",
    )
    expected = 12  # 2 spacings × 3 orientations × 2 components
    assert len(rx_list) == expected, f"接收器数量: {len(rx_list)} != {expected}"
    print(f"  接收器数量: {len(rx_list)} (预期 {expected}) [OK]")

    # 检查取向覆盖
    orientations = set()
    for rx in rx_list:
        ori = rx.orientation
        if isinstance(ori, np.ndarray):
            ori = tuple(ori.tolist())
        orientations.add(ori)
    print(f"  取向覆盖: {len(orientations)} 种 [OK]")
    return True


def test_tensor_survey():
    """Test 2: 三轴源 × 三轴接收 Survey。"""
    print("[Test 2] build_tensor_survey 三轴源×三轴接收")
    survey = build_tensor_survey(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        field_kind="h",
        secondary=False,
        components=("real", "imag"),
        rx_orientations=("x", "y", "z"),
        src_orientations=("x", "y", "z"),
        form="h",
    )
    # 2 频率 × 3 源取向 = 6 源
    assert len(survey.source_list) == 6, f"源数量: {len(survey.source_list)}"
    # 每源 2 收发距 × 3 取向 × 2 分量 = 12 接收器
    n_rx = len(survey.source_list[0].receiver_list)
    assert n_rx == 12, f"每源接收器: {n_rx}"
    print(f"  源数量: {len(survey.source_list)}, 每源接收器: {n_rx} [OK]")
    return True


def test_tensor_data_order():
    """Test 3: 数据维度计算。"""
    print("[Test 3] tensor_data_order")
    order = tensor_data_order(
        frequencies=(1e4, 1e5, 1e6),
        src_orientations=("x", "y", "z"),
        spacings=np.array([0.3, 0.6, 1.0]),
        rx_orientations=("x", "y", "z"),
        components=("real", "imag"),
    )
    expected_shape = (3, 3, 3, 3, 2)
    expected_total = 162
    assert order["shape"] == expected_shape, f"形状: {order['shape']}"
    assert order["total"] == expected_total, f"总数: {order['total']}"
    print(f"  数据形状: {order['shape']}, 总数: {order['total']} [OK]")
    return True


def test_single_orientation_degenerate():
    """Test 4: 单取向退化兼容。"""
    print("[Test 4] 单取向退化（与原始 survey 等价）")
    survey_single = build_tensor_survey(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        field_kind="h",
        secondary=False,
        components=("real", "imag"),
        rx_orientations=("z",),
        src_orientations=("z",),
        form="h",
    )
    assert len(survey_single.source_list) == 2
    assert len(survey_single.source_list[0].receiver_list) == 4
    print(f"  单轴退化: {len(survey_single.source_list)} 源 × "
          f"{len(survey_single.source_list[0].receiver_list)} 接收器 [OK]")
    return True


def test_original_survey_unchanged():
    """Test 5: 原始 build_survey 未受影响。"""
    print("[Test 5] 原始 build_survey 未受影响")
    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="axial",
    )
    orig_survey = build_survey(tool, form="h")
    assert len(orig_survey.source_list) == 2
    assert len(orig_survey.source_list[0].receiver_list) == 4
    print(f"  原始 survey: {len(orig_survey.source_list)} 源 × "
          f"{len(orig_survey.source_list[0].receiver_list)} 接收器 [OK]")
    return True


def test_error_handling():
    """Test 6: 错误处理。"""
    print("[Test 6] 错误处理")
    # 电场+二次场
    try:
        build_tensor_receivers(
            spacings=np.array([0.5]),
            field_kind="e",
            secondary=True,
            form="e",
        )
        print("  电场+二次场 应报错但未报错 [FAIL]")
        return False
    except ValueError:
        print("  电场+二次场 正确报错 [OK]")

    # 非法取向
    try:
        build_tensor_receivers(
            spacings=np.array([0.5]),
            field_kind="h",
            rx_orientations=("invalid",),
            form="h",
        )
        print("  非法取向 应报错但未报错 [FAIL]")
        return False
    except ValueError:
        print("  非法取向 正确报错 [OK]")
    return True


def main():
    print("=" * 60)
    print("E1+E2 测试：多分量张量接收与多分量源阵列")
    print("=" * 60)

    tests = [
        test_tensor_receivers,
        test_tensor_survey,
        test_tensor_data_order,
        test_single_orientation_degenerate,
        test_original_survey_unchanged,
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
            print(f"  异常: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
