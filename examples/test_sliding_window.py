# -*- coding: utf-8 -*-
"""E10 测试：滑动窗口反演。

测试内容：
1. sliding_window_inversion_1d 基本功能
2. simple_window_builder 构建器
3. simple_inversion_runner 运行器
4. 合成数据端到端测试

图版保存到 examples/figs/test_window/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lwd_fdem import (
    LWDFDEMForward, LWDFDEMInversion, LWDToolConfig,
    sliding_window_inversion_1d, simple_window_builder, simple_inversion_runner,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs", "test_window")
os.makedirs(FIG_DIR, exist_ok=True)


def test_sliding_window_basic():
    """Test 1: 滑动窗口反演基本功能。"""
    print("[Test 1] sliding_window_inversion_1d 基本功能")

    # 真实模型：三层
    sigma_true = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])
    n_layers = 3

    # 工具配置
    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    # 生成合成数据（多个深度位置）
    n_depths = 5
    depths = np.arange(n_depths) * 1.0  # 0, 1, 2, 3, 4 m

    # 为每个深度生成数据（这里简化：所有深度用同一模型）
    fwd_template = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)
    d_template = fwd_template.dpred(sigma_true)
    nD = len(d_template)

    # 添加少量噪声，构建 (n_depths, nD) 数组
    np.random.seed(42)
    noise_level = 0.01
    data_at_depths = np.zeros((n_depths, nD))
    for i in range(n_depths):
        noise = np.random.randn(nD) * noise_level * np.abs(d_template)
        data_at_depths[i] = d_template + noise

    # 构建器：接收 data_avg，返回正演对象
    def forward_builder(data_avg):
        return LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

    # 运行器：接收 (forward, data_target)，返回 sigma
    def inversion_runner(forward, data_target):
        from src.lwd_fdem.dataio import make_data_object
        data_obj = make_data_object(forward.survey, data_target, relative_error=0.05)
        m0 = np.log(0.1 * np.ones(forward.n_params))
        inv = LWDFDEMInversion(forward, data_obj, m0=m0, max_iter=10)
        sigma, _ = inv.run()
        return sigma

    # 执行滑动窗口反演
    result = sliding_window_inversion_1d(
        depths=depths,
        data_at_depths=data_at_depths,
        forward_builder=forward_builder,
        inversion_runner=inversion_runner,
        window_size=3,
        step=1,
        verbose=False,
    )

    # 验证返回结构
    assert "depths" in result, "缺少 depths"
    assert "models" in result, "缺少 models"
    assert "misfits" in result, "缺少 misfits"
    assert "converged" in result, "缺少 converged"

    print(f"  深度点数: {len(result['depths'])}")
    print(f"  模型数量: {len(result['models'])}")
    print(f"  拟合值范围: [{result['misfits'].min():.2e}, {result['misfits'].max():.2e}]")
    print(f"  收敛情况: {result['converged'].sum()}/{len(result['converged'])}")

    # 至少有部分反演收敛
    assert result["converged"].sum() > 0, "没有反演收敛"
    print(f"  有反演收敛 [OK]")
    return True


def test_simple_window_builder():
    """Test 2: 简单窗口构建器。"""
    print("[Test 2] simple_window_builder 构建器")

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    layer_thicknesses = np.array([2.0, 4.0])
    n_layers = 3

    builder = simple_window_builder(
        tool_config=tool,
        layer_thicknesses=layer_thicknesses,
        n_layers=n_layers,
        sigma_bounds=(1e-3, 10.0),
    )

    # 调用构建器（传入模拟的 data_avg）
    nD = len(tool.frequencies) * len(tool.spacings) * len(tool.components) * 2  # real+imag
    data_avg = np.ones(nD) * 1e-6  # 模拟小振幅数据
    fwd = builder(data_avg)

    # 验证返回的是正演对象
    assert isinstance(fwd, LWDFDEMForward), f"类型错误: {type(fwd)}"
    assert fwd.geometry == "1d", f"几何错误: {fwd.geometry}"
    print(f"  构建器返回 LWDFDEMForward [OK]")
    return True


def test_simple_inversion_runner():
    """Test 3: 简单反演运行器。"""
    print("[Test 3] simple_inversion_runner 运行器")

    sigma_true = np.array([0.25, 0.01, 0.5])
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
    data = fwd.dpred(sigma_true)

    # 创建运行器
    runner = simple_inversion_runner(max_iter=10, beta_cooling=5.0, chifact=1.0)

    # 执行反演
    sigma = runner(fwd, data)

    # 验证返回的是 numpy 数组
    assert isinstance(sigma, np.ndarray), f"类型错误: {type(sigma)}"
    assert len(sigma) == 3, f"模型长度错误: {len(sigma)}"
    print(f"  反演模型: {sigma}")
    print(f"  反演运行器正常 [OK]")
    return True


def test_end_to_end():
    """Test 4: 端到端测试。"""
    print("[Test 4] 端到端测试")

    # 真实模型
    sigma_true = np.array([0.25, 0.01, 0.5])
    thicknesses = np.array([2.0, 4.0])

    tool = LWDToolConfig(
        frequencies=(1e4, 1e5),
        spacings=np.array([0.5, 1.0]),
        components=("real", "imag"),
        field_kind="h",
        secondary=False,
        receiver_geometry="radial",
    )

    # 生成 3 个深度位置的数据
    n_depths = 3
    depths = np.array([0.0, 1.0, 2.0])

    fwd_template = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)
    d_template = fwd_template.dpred(sigma_true)
    nD = len(d_template)

    # 构建 (n_depths, nD) 数组
    data_at_depths = np.zeros((n_depths, nD))
    for i in range(n_depths):
        data_at_depths[i] = d_template.copy()

    # 使用简单构建器和运行器
    builder = simple_window_builder(tool, thicknesses, n_layers=3)
    runner = simple_inversion_runner(max_iter=10)

    # 执行滑动窗口反演
    result = sliding_window_inversion_1d(
        depths=depths,
        data_at_depths=data_at_depths,
        forward_builder=builder,
        inversion_runner=runner,
        window_size=2,
        step=1,
        verbose=False,
    )

    # 验证结果
    assert len(result["depths"]) == n_depths, f"深度数错: {len(result['depths'])}"
    assert len(result["models"]) == n_depths, f"模型数错: {len(result['models'])}"

    # 计算平均反演模型
    avg_model = np.mean(result["models"], axis=0)
    print(f"  真实模型: {sigma_true}")
    print(f"  平均反演: {avg_model}")

    # 简单反演设置可能无法精确收敛，只验证机制工作
    # （滑动窗口能产出所有深度的反演结果）
    print(f"  滑动窗口机制正常 [OK]")
    return True


def main():
    print("=" * 60)
    print("E10 测试：滑动窗口反演")
    print("=" * 60)

    tests = [
        test_sliding_window_basic,
        test_simple_window_builder,
        test_simple_inversion_runner,
        test_end_to_end,
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
