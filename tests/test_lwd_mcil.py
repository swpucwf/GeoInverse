"""多分量感应测井扩展功能测试。

验证姿态角（井斜/方位/工具面）、各向异性 sigma、
1d_tensor/2d_tensor 新路径的正确性。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""
import numpy as np
import pytest

from src.lwd_fdem import (
    LWDToolConfig,
    rotation_matrix,
    build_2d_tensor_mesh,
    build_3d_tensor_mesh,
    build_survey_tilted,
    build_tensor_survey,
    LWDFDEMForward,
)

# --- 容差常量 ---
TOL_ORTHO = 1e-12       # 旋转矩阵正交性（浮点三角函数残差量级 ~1e-16）
TOL_COMPAT = 1e-10      # 向后兼容退化（orientation_vector 与字符串取向一致）


class TestRotationMatrix:
    """rotation_matrix 函数测试。"""

    def test_orthogonality(self):
        """R.T @ R == I, det(R) == 1，多组角度。"""
        angles = [(0, 0, 0), (75, 0, 0), (45, 30, 15), (90, 90, 90), (180, 360, 0)]
        for inc, azi, tf in angles:
            R = rotation_matrix(inc, azi, tf)
            assert np.allclose(R.T @ R, np.eye(3), atol=TOL_ORTHO)
            assert np.isclose(np.linalg.det(R), 1.0, atol=TOL_ORTHO)

    def test_identity(self):
        """inc=0, azi=0, tf=0 → R == I。"""
        R = rotation_matrix(0.0, 0.0, 0.0)
        assert np.allclose(R, np.eye(3), atol=TOL_ORTHO)

    def test_known_values(self):
        """inc=90, azi=0, tf=0 → 工具 z 轴指向地层 x 方向。"""
        R = rotation_matrix(90.0, 0.0, 0.0)
        # Ry(90°) = [[0,0,1],[0,1,0],[-1,0,0]]
        # R[:,2] 应为 [1, 0, 0]（工具 z 轴 = 地层 x 方向）
        assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=TOL_ORTHO)

    def test_inclination_75(self):
        """inc=75, azi=0, tf=0 → 验证方向向量。"""
        R = rotation_matrix(75.0, 0.0, 0.0)
        expected_z = np.array([np.sin(np.radians(75)), 0.0, np.cos(np.radians(75))])
        assert np.allclose(R[:, 2], expected_z, atol=TOL_ORTHO)


class TestLWDToolConfigExtension:
    """LWDToolConfig 姿态角扩展测试。"""

    def test_orientation_vector_backward_compat(self):
        """不传姿态角时 orientation_vector 退化为原 orientation 字符串对应向量。"""
        cfg = LWDToolConfig(frequencies=(2e3,), spacings=np.array([1.0]), orientation="z")
        assert np.allclose(cfg.orientation_vector, [0.0, 0.0, 1.0], atol=TOL_COMPAT)

        cfg_x = LWDToolConfig(frequencies=(2e3,), spacings=np.array([1.0]), orientation="x")
        assert np.allclose(cfg_x.orientation_vector, [1.0, 0.0, 0.0], atol=TOL_COMPAT)

    def test_orientation_vector_with_inclination(self):
        """传入 inclination=75 时 orientation_vector 正确计算。"""
        cfg = LWDToolConfig(
            frequencies=(12e3,), spacings=np.array([10.0]),
            inclination=75.0, azimuth=0.0, tool_face=0.0
        )
        expected = np.array([np.sin(np.radians(75)), 0.0, np.cos(np.radians(75))])
        assert np.allclose(cfg.orientation_vector, expected, atol=TOL_COMPAT)

    def test_rotation_matrix_property(self):
        """config.rotation_matrix 与 rotation_matrix() 函数结果一致。"""
        cfg = LWDToolConfig(
            frequencies=(2e3,), spacings=np.array([1.0]),
            inclination=45.0, azimuth=30.0, tool_face=15.0
        )
        R_func = rotation_matrix(45.0, 30.0, 15.0)
        assert np.allclose(cfg.rotation_matrix, R_func, atol=TOL_ORTHO)

    def test_inclination_validation(self):
        """inclination 超范围应报错。"""
        with pytest.raises((ValueError, Exception)):
            LWDToolConfig(frequencies=(2e3,), spacings=np.array([1.0]), inclination=200.0)

    def test_default_values_preserve_behavior(self):
        """默认参数下行为与扩展前一致。"""
        cfg = LWDToolConfig(frequencies=(2e3,), spacings=np.array([1.0]))
        assert cfg.inclination == 0.0
        assert cfg.azimuth == 0.0
        assert cfg.tool_face == 0.0
        # orientation_vector 应等于 orientation="z" 对应的 [0,0,1]
        assert np.allclose(cfg.orientation_vector, [0.0, 0.0, 1.0])


class TestDimensionGuards:
    """维度能力校验测试。"""

    def test_2d_cyl_rejects_tilted(self):
        """geometry='2d_cyl' + 倾斜仪器 → NotImplementedError。"""
        cfg = LWDToolConfig(
            frequencies=(2e3,), spacings=np.array([1.0]),
            inclination=45.0, field_kind="b", secondary=False
        )
        with pytest.raises(NotImplementedError):
            LWDFDEMForward(geometry="2d_cyl", tool=cfg, layer_thicknesses=[10.0, 20.0])

    def test_1d_rejects_anisotropic(self):
        """geometry='1d' + anisotropic=True → NotImplementedError。"""
        cfg = LWDToolConfig(frequencies=(2e3,), spacings=np.array([1.0]),
                            receiver_geometry="radial")
        with pytest.raises(NotImplementedError):
            LWDFDEMForward(geometry="1d", tool=cfg,
                           layer_thicknesses=[10.0, 20.0], anisotropic=True)


class TestBuild2DTensorMesh:
    """build_2d_tensor_mesh 函数测试。"""

    def test_mesh_shape(self):
        """网格维度正确：y 方向为 ny 个 cell。"""
        mesh = build_2d_tensor_mesh(
            core_extent_xz=(20.0, 30.0),
            cell_size=2.0,
            ny=3
        )
        # y 方向应为 ny=3 个 cell
        assert mesh.shape_cells[1] == 3
        # x 和 z 方向应有核心区 + padding
        assert mesh.shape_cells[0] > 10  # nx_core = ceil(20/2) = 10, + padding
        assert mesh.shape_cells[2] > 15  # nz_core = ceil(30/2) = 15, + padding

    def test_mesh_is_tensor(self):
        """返回 TensorMesh 类型。"""
        from discretize import TensorMesh
        mesh = build_2d_tensor_mesh(core_extent_xz=(10.0, 10.0), cell_size=1.0)
        assert isinstance(mesh, TensorMesh)


class TestTensorOrientations:
    """tensor.py 向量 orientations 支持测试。"""

    def test_vector_orientations_accepted(self):
        """传入 ndarray 向量作为 orientations 不报错。"""
        ori_vecs = [np.r_[1.0, 0.0, 0.0], np.r_[0.0, 1.0, 0.0], np.r_[0.0, 0.0, 1.0]]
        # 应不抛异常
        survey = build_tensor_survey(
            frequencies=(2e3,),
            spacings=np.array([1.0]),
            rx_orientations=ori_vecs,
            src_orientations=ori_vecs,
        )
        assert len(survey.source_list) == 3  # 3 个源方向

    def test_string_orientations_still_work(self):
        """字符串 orientations 保持原有行为。"""
        survey = build_tensor_survey(
            frequencies=(2e3,),
            spacings=np.array([1.0]),
            rx_orientations=("x", "y", "z"),
            src_orientations=("z",),
        )
        assert len(survey.source_list) == 1


class TestBuildSurveyTilted:
    """build_survey_tilted 函数测试。"""

    def test_survey_structure(self):
        """倾斜 survey 的源和接收器 orientation 为向量。"""
        cfg = LWDToolConfig(
            frequencies=(12e3,), spacings=np.array([5.0, 10.0]),
            inclination=75.0, azimuth=0.0, tool_face=0.0,
            field_kind="h", secondary=False
        )
        survey = build_survey_tilted(cfg, "h")
        assert len(survey.source_list) == 1  # 单频率
        src = survey.source_list[0]
        # 源的 orientation 应为向量（非字符串）
        assert isinstance(src.orientation, np.ndarray)
        assert src.orientation.shape == (3,)


# --- 以下为 slow 测试（3D 正演） ---

@pytest.mark.slow
class TestForward1DTensor:
    """1d_tensor 路径正演测试。"""

    def test_isotropic_smoke(self):
        """1d_tensor 各向同性 smoke test：正演不崩溃，输出非零。"""
        cfg = LWDToolConfig(
            frequencies=(2e3,), spacings=np.array([2.0]),
            inclination=0.0, field_kind="h", secondary=False,
            receiver_geometry="axial"
        )
        # 显式提供小网格以约束求解规模（自动网格约 5e4 单元，direct 求解过慢）
        mesh = build_3d_tensor_mesh(
            core_extent=(30.0, 6.0, 40.0), cell_size=3.0, n_pad=3
        )
        fwd = LWDFDEMForward(
            geometry="1d_tensor", tool=cfg, mesh=mesh,
            layer_thicknesses=[20.0, 40.0],
            anisotropic=False
        )
        sigma = np.array([0.1, 0.05, 0.01])  # 3 层
        d = fwd.dpred(sigma)
        assert d is not None
        assert len(d) > 0
        assert np.any(np.abs(d) > 0)

    def test_anisotropic_smoke(self):
        """1d_tensor 各向异性 smoke test。"""
        cfg = LWDToolConfig(
            frequencies=(2e3,), spacings=np.array([2.0]),
            inclination=30.0, field_kind="h", secondary=False,
            receiver_geometry="axial"
        )
        mesh = build_3d_tensor_mesh(
            core_extent=(30.0, 6.0, 40.0), cell_size=3.0, n_pad=3
        )
        fwd = LWDFDEMForward(
            geometry="1d_tensor", tool=cfg, mesh=mesh,
            layer_thicknesses=[20.0, 40.0],
            anisotropic=True
        )
        # 各向异性：n_params = n_layers * 2 = 6
        sigma_h = np.array([0.1, 0.05, 0.01])
        sigma_v = np.array([0.05, 0.05, 0.005])
        m = np.concatenate([sigma_h, sigma_v])
        d = fwd.dpred(m)
        assert d is not None
        assert np.any(np.abs(d) > 0)


@pytest.mark.slow
class TestForward3DTilted:
    """3D 路径倾斜+各向异性测试。"""

    def test_3d_tilted_anisotropic(self):
        """3D + inc=75° + VTI → 正演不崩溃。"""
        cfg = LWDToolConfig(
            frequencies=(2e3,), spacings=np.array([2.0]),
            inclination=75.0, field_kind="h", secondary=False,
            receiver_geometry="axial"
        )
        mesh = build_3d_tensor_mesh(
            core_extent=(20.0, 20.0, 30.0),
            cell_size=2.0, n_pad=4
        )
        fwd = LWDFDEMForward(
            geometry="3d", tool=cfg, mesh=mesh, anisotropic=True
        )
        sigma = np.ones((mesh.n_cells, 3)) * 0.05
        sigma[:, 2] = 0.025  # VTI
        d = fwd.dpred(sigma.ravel())
        assert d is not None
        assert np.any(np.abs(d) > 0)
