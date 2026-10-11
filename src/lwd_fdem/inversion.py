"""LWD 频域电磁反演管线。

基于 fdemcore 的目标函数体系（data_misfit + regularization + optimization
+ directives）组装反演外循环，默认 Occam 风格：对数电导率参数化 +
加权最小二乘平滑 + beta 冷却 + 目标拟合差停止（决策清单 E/G）。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh

from ..fdemcore import data_misfit, directives, inverse_problem, inversion
from ..fdemcore import maps, optimization, regularization
from ..fdemcore.utils import validate_string
from .forward import LWDFDEMForward
from .meshes import build_1d_mesh

logger = logging.getLogger(__name__)


def _cyl_reduced_mesh(mesh):
    """将 CylindricalMesh 的 (r,z) 截面提取为 2D TensorMesh + 单元重复索引。

    CylindricalMesh 不支持 cell_grads（正则化平滑所需），故正则化使用
    等价的 (r,z) 平面张量网格；phi 分段通过索引展开映射回 3D 单元。
    单元宽度由 cell_centers 差分推得（mesh.h 属性存在惰性求值歧义，
    不予依赖）。

    Args:
        mesh: discretize CylindricalMesh。

    Returns:
        (mesh2d, index)：TensorMesh 与长度 mesh.n_cells 的整数索引数组，
        满足 model_3d = model_2d[index]。
    """
    cc = np.asarray(mesh.cell_centers)
    hw = np.asarray(mesh.h_gridded)          # (nC, 3) 每单元 (dr, dphi, dz)
    r_key = np.round(cc[:, 0], 10)
    z_key = np.round(cc[:, 2], 10)
    ur = np.unique(r_key)
    # 第一个 r 环带内的 z 序列（自下而上）即为全部 z 中心
    uz = np.unique(z_key[r_key == ur[0]])
    # 从实际单元宽度数组取精确的 r/z 宽度序列（phi 分段时取首个出现）
    sel_r = np.stack([np.flatnonzero(r_key == c)[0] for c in ur])
    sel_z = np.stack([np.flatnonzero((r_key == ur[0]) & (z_key == c))[0]
                      for c in uz])
    hr, hz = hw[sel_r, 0], hw[sel_z, 2]
    mesh2d = TensorMesh([hr, hz], x0="CC")
    ir = np.searchsorted(ur, r_key)
    iz = np.searchsorted(uz, z_key)
    index = ir + iz * ur.size
    return mesh2d, index


class LWDFDEMInversion:
    """LWD FDEM 反演器（1D / 2D 轴对称 / 3D 通用组装）。

    本类是反演计算的统一入口，基于 fdemcore 的目标函数体系
    （data_misfit + regularization + optimization + directives）
    组装 Occam 风格反演外循环。

    反演参数化为 log(sigma)（ExpMap 保证电导率始终为正）。
    正则化网格：1D 使用层状张量网格，2D/3D 使用正演网格。

    主要方法:
        - ``run()``: 执行完整反演流程，返回反演结果模型。
        - ``compute_sensitivity(m)``: 计算并返回灵敏度矩阵 J。

    Args:
        forward: 正演求解器。
            *物理含义*: 封装了正演计算与灵敏度矩阵乘积的完整链路。
            *使用方式*: 传入已配置好的 ``LWDFDEMForward`` 实例。
            *应用场景*: 所有反演均必需。

        data_obj: 观测数据对象。
            *物理含义*: 包含实测数据向量 dobs 和标准差 uncertainty，
            用于构建数据目标函数 φ_d = ||W(d_pred - d_obs)||²。
            *使用方式*: ``fdemcore.data.Data`` 实例，须已设置 dobs 和 standard_deviation。
            *应用场景*: 所有反演均必需。

        m0: 初始模型（log-电导率）。
            *物理含义*: 反演迭代的起始模型，为 log(σ) 形式。
            *使用方式*: 1D numpy 数组，形状 (n_params,)。
            ``None`` 时为 log(0.1) 常模型（即初始电导率 0.1 S/m）。
            *应用场景*: 初始模型影响反演收敛路径，建议根据先验信息设置。

        mref: 参考模型（log-电导率）。
            *物理含义*: 正则化项中的参考模型，正则化度量模型与参考模型的偏离。
            *使用方式*: 1D numpy 数组，形状 (n_params,)。``None`` 时为 m0。
            *应用场景*: 平滑约束反演中常设为背景值或先验模型。

        sparse: 是否使用稀疏正则化。
            *物理含义*: True 时使用 IRLS（迭代重加权最小二乘）求 Lp 范数稀疏解，
            促进块状/边界尖锐模型；False 时使用 L2 范数平滑约束。
            *使用方式*: 布尔值，默认 False。
            *应用场景*: 地层边界尖锐、异常体定位时开启；平滑地层用 False。

        alpha_s: 小扰动（最小结构）权重。
            *物理含义*: 正则化中模型与参考模型偏离的惩罚系数。
            *使用方式*: 非负浮点数，默认 1e-3。
            *应用场景*: 值越小越允许模型偏离参考，数据拟合优先。

        alpha_x / alpha_y / alpha_z: 各方向一阶平滑权重。
            *物理含义*: 正则化中模型在 x/y/z 方向空间变化的惩罚系数。
            *使用方式*: 非负浮点数。1D 仅用 alpha_z；2D 用 alpha_x (r) + alpha_z (z)；
            3D 全用。``None`` 时默认 1.0。
            *应用场景*: 控制模型平滑方向偏好；各向同性平滑全设 1.0；
            倾向平滑可增大某个方向。

        norms: 稀疏正则化的范数阶数。
            *物理含义*: ``[p, q]`` 分别对应模型目标函数 p-范数和梯度目标函数 q-范数。
            *使用方式*: 列表，如 ``[0, 2]``（L0 + L2，默认）或 ``[1, 1]``（L1 + TV）。
            *应用场景*: 仅 ``sparse=True`` 时生效。

        optimizer: 优化器类型。
            *物理含义*: ``'gn'`` 为不精确 Gauss-Newton；``'pgncg'`` 为投影 GN 共轭梯度。
            *使用方式*: 字符串，默认 ``'gn'``。
            *应用场景*: 中小规模用 ``'gn'``；大规模/带约束用 ``'pgncg'``。

        beta0_ratio: 初始 beta 比例。
            *物理含义*: 数据目标函数与正则化的初始权衡系数（由特征值估计缩放）。
            *使用方式*: 正浮点数，默认 1.0。
            *应用场景*: 调大则初始更重数据拟合，调小则更重正则化。

        beta_cooling: Beta 冷却因子。
            *物理含义*: 每轮迭代后 beta 乘以的因子（逐步降低正则化权重）。
            *使用方式*: 大于 1 的浮点数，默认 5.0。
            *应用场景*: Occam 风格冷却策略，逐步从平滑解过渡到数据拟合解。

        chifact: 目标拟合差系数。
            *物理含义*: 目标 phi_d = chifact × nD（nD 为数据个数）。
            *使用方式*: 正浮点数，默认 1.0。
            *应用场景*: 当 phi_d 降至目标值以下时停止反演。
            标准差估计准确时取 1.0。

        max_iter: 最大迭代次数。
            *物理含义*: 反演迭代上限。
            *使用方式*: 正整数，默认 40。
            *应用场景*: 防止不收敛时无限循环。

        sensitivity_weights: 是否启用灵敏度加权。
            *物理含义*: 基于灵敏度矩阵的对角元构造加权，补偿深部灵敏度衰减。
            *使用方式*: 布尔值，默认 False。
            *应用场景*: 深部地层反演时开启可改善分辨率。

        preconditioner: 是否启用 Jacobi 预条件。
            *物理含义*: 用 Gauss-Newton 矩阵对角元作预条件，加速共轭梯度收敛。
            *使用方式*: 布尔值，默认 False。
            *应用场景*: 仅 ``optimizer='pgncg'`` 时生效。

    Attributes:
        forward: 已配置的正演求解器。
        data_obj: fdemcore data.Data 观测数据。
        m0: 初始模型（log-sigma），形状 (n_params,)。
    """

    def __init__(self, forward: LWDFDEMForward, data_obj,
                 m0: Optional[np.ndarray] = None,
                 mref: Optional[np.ndarray] = None,
                 sparse: bool = False,
                 alpha_s: float = 1e-3,
                 alpha_x: float = 1.0,
                 alpha_y: Optional[float] = None,
                 alpha_z: Optional[float] = None,
                 norms: Optional[list] = None,
                 optimizer: str = "gn",
                 beta0_ratio: float = 1.0,
                 beta_cooling: float = 5.0,
                 chifact: float = 1.0,
                 max_iter: int = 25,
                 sensitivity_weights: bool = True,
                 preconditioner: bool = True):
        validate_string("optimizer", optimizer, string_list=("gn", "pgncg"))
        self.forward = forward
        self.data_obj = data_obj
        self.m0 = None if m0 is None else np.asarray(m0, dtype=float)
        self.mref = None if mref is None else np.asarray(mref, dtype=float)
        self.sparse = sparse
        self.alpha_s = alpha_s
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z
        self.norms = [0, 2] if norms is None else list(norms)
        self.optimizer = optimizer
        self.beta0_ratio = beta0_ratio
        self.beta_cooling = beta_cooling
        self.chifact = chifact
        self.max_iter = max_iter
        self.sensitivity_weights = sensitivity_weights
        self.preconditioner = preconditioner

        self._components: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # 组件组装
    # ------------------------------------------------------------------ #
    @property
    def reg_mesh(self):
        """正则化网格（1D：层状张量网格；2D 轴对称：(r,z) 截面网格；3D：正演网格）。"""
        if self.forward.geometry == "1d":
            return build_1d_mesh(self.forward.layer_thicknesses)
        if self.forward.geometry == "2d_cyl":
            mesh2d, _ = _cyl_reduced_mesh(self.forward.mesh)
            return mesh2d
        return self.forward.mesh

    @property
    def nP_model(self) -> int:
        """反演模型参数个数（2D 轴对称按 (r,z) 截面计，其余 = n_params）。"""
        if self.forward.geometry == "2d_cyl":
            mesh2d, _ = _cyl_reduced_mesh(self.forward.mesh)
            return int(mesh2d.n_cells)
        return self.forward.n_params

    def _cells_expand(self) -> Optional[np.ndarray]:
        """2D 轴对称：截面模型 -> 3D 单元的顺序索引；其余几何返回 None。"""
        if self.forward.geometry == "2d_cyl":
            _, index = _cyl_reduced_mesh(self.forward.mesh)
            return index
        return None

    def build_mapping(self):
        """log-sigma 模型 -> 单元电导率的映射（含 2D 轴对称的截面向展开）。

        Returns:
            非轴对称几何返回 ExpMap(nP=n_params)；2D 轴对称返回组合映射
            ExpMap(n_cells) * LinearMap(P)，其中 P 将截面 log-sigma 模型
            展开到 3D 单元，nP = 截面模型维数。
        """
        index = self._cells_expand()
        if index is None:
            return maps.ExpMap(nP=self.forward.n_params)
        n_out = int(self.forward.mesh.n_cells)
        # P: (n_out, nP_model)，P[i, index[i]] = 1（截面 -> 3D 单元选择展开）
        P = sp.csr_matrix(
            (np.ones(index.size), (np.arange(index.size), index)),
            shape=(n_out, self.nP_model),
        )
        return maps.ExpMap(nP=n_out) * maps.LinearMap(P)

    def build_simulation(self):
        """反演用仿真（电导率走 ExpMap）。"""
        return self.forward.build_simulation(sigma_map=self.build_mapping())

    def _sparse_norms(self, dim: int) -> list:
        """把稀疏 norms 规整为长度 1+dim（小扰动 + 各方向平滑范数）。

        用户显式给了正确长度则原样返回；给默认两元素 [p_s, p_g] 时，
        按网格维度复制平滑范数（如 2D→[p_s, p_g, p_g]）。
        """
        norms = list(self.norms) if self.norms is not None else [0, 2]
        n = 1 + dim
        if len(norms) == n:
            return norms
        if len(norms) == 2:
            return [norms[0]] + [norms[1]] * dim
        return (norms + [norms[-1]] * n)[:n]

    def build_regularization(self):
        """构建正则化目标函数。"""
        mesh = self.reg_mesh
        mapping = maps.IdentityMap(nP=self.nP_model)
        if self.sparse:
            reg = regularization.Sparse(
                mesh, mapping=mapping,
                alpha_s=self.alpha_s, alpha_x=self.alpha_x,
                alpha_y=self.alpha_y if self.alpha_y is not None else 1.0,
                alpha_z=self.alpha_z if self.alpha_z is not None else 1.0,
            )
            reg.norms = self._sparse_norms(mesh.dim)
        else:
            reg = regularization.WeightedLeastSquares(
                mesh, mapping=mapping,
                alpha_s=self.alpha_s, alpha_x=self.alpha_x,
                alpha_y=self.alpha_y if self.alpha_y is not None else 1.0,
                alpha_z=self.alpha_z if self.alpha_z is not None else 1.0,
            )
        mref = self._to_model_space(self.mref, "mref")
        if mref is not None:
            reg.reference_model = mref
        return reg

    def build_optimizer(self):
        """构建优化器。"""
        if self.optimizer == "pgncg" or self.sparse:
            return optimization.ProjectedGNCG(
                maxIter=self.max_iter, maxIterLS=20,
                cg_maxiter=30, cg_rtol=1e-3, tolF=1e-6, tolG=1e-6,
            )
        return optimization.InexactGaussNewton(maxIter=self.max_iter,
                                               tolF=1e-6, tolG=1e-6)

    def build_directives(self):
        """构建反演指令列表（G4）。"""
        save = directives.SaveOutputEveryIteration(on_disk=False)
        directive_list = []
        if self.sensitivity_weights and self.reg_mesh.dim > 1:
            directive_list.append(
                directives.UpdateSensitivityWeights(every_iteration=False)
            )
        directive_list.append(directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio))
        # sparse 时 β 冷却由 UpdateIRLS 自管，不能再加 BetaSchedule（否则与 IRLS 冲突）
        if not self.sparse:
            directive_list.append(directives.BetaSchedule(coolingFactor=self.beta_cooling))
        if self.preconditioner and not self.sparse:
            directive_list.append(directives.UpdatePreconditioner())
        directive_list.append(save)
        directive_list.append(directives.TargetMisfit(chifact=self.chifact))
        if self.sparse:
            directive_list.insert(
                len(directive_list) - 1,
                directives.UpdateIRLS(max_irls_iterations=10,
                                      irls_cooling_factor=1.5),
            )
            directive_list.insert(
                len(directive_list) - 1, directives.UpdatePreconditioner(),
            )
        self._components["save"] = save
        return directive_list

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def _to_model_space(self, vec: Optional[np.ndarray],
                        name: str) -> Optional[np.ndarray]:
        """将用户向量规整到截面参数化维度（2D 轴对称支持 3D 单元向量输入）。

        Args:
            vec: 输入向量（log-sigma），可为 None。
            name: 变量名（用于报错信息）。

        Returns:
            形状 (nP_model,) 的向量或 None。

        Raises:
            ValueError: 维度既不是 nP_model 也不是（可投影的）3D 单元数时。
        """
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=float)
        index = self._cells_expand()
        if index is not None and vec.size == self.forward.n_params \
                and self.nP_model != self.forward.n_params:
            first = np.zeros(self.nP_model, dtype=int)
            seen = np.zeros(self.nP_model, dtype=bool)
            for k, j in enumerate(index):
                if not seen[j]:
                    first[j] = k
                    seen[j] = True
            return vec[first]
        if vec.size != self.nP_model:
            raise ValueError(
                f"{name} 长度 {vec.size} 与模型参数维度 {self.nP_model} 不符。")
        return vec

    def _normalize_m0(self) -> np.ndarray:
        """初始模型规整；None 时取 log(0.1) 常模型。"""
        if self.m0 is None:
            return np.log(0.1 * np.ones(self.nP_model))
        return self._to_model_space(self.m0, "m0")

    def run(self) -> Tuple[np.ndarray, Dict]:
        """执行完整反演流程。

        *物理含义*: 组装数据目标函数 + 正则化 + 优化器 + 指令链，
        从初始模型出发迭代求解非线性反演问题，直至达到目标拟合差或最大迭代。
        *使用方式*: 无需参数，所有配置在构造时已设定。调用后返回反演结果。
        *应用场景*: 反演计算主入口。

        Returns:
            (recovered_sigma, history)：
              recovered_sigma: 恢复的电导率模型 (S/m)，形状 (n_params,)。
              history: 字典，含 'model_log'（最终 log-sigma）、'iterations'
                （每次迭代的 phi_d/phi_m/beta 记录列表）、'final_phi_d'、
                'final_phi_m'、'final_beta'。
        """
        simulation = self.build_simulation()
        dmisfit = data_misfit.L2DataMisfit(data=self.data_obj,
                                           simulation=simulation)
        reg = self.build_regularization()
        opt = self.build_optimizer()
        inv_prob = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
        directive_list = self.build_directives()
        inv = inversion.BaseInversion(inv_prob, directive_list)

        m_start = self._normalize_m0()
        logger.info("开始 LWD FDEM 反演：nD=%d, nP=%d, geometry=%s",
                    self.data_obj.dobs.size, self.nP_model,
                    self.forward.geometry)
        recovered_log = inv.run(m_start)
        sigma = self.build_mapping() * recovered_log
        save = self._components["save"]

        iterations = [
            {"beta": b, "phi_d": pd, "phi_m": pm}
            for b, pd, pm in zip(save.beta, save.phi_d, save.phi_m)
        ]
        history = {
            "model_log": recovered_log,
            "iterations": iterations,
            "final_phi_d": float(save.phi_d[-1]) if iterations else 0.0,
            "final_phi_m": float(save.phi_m[-1]) if iterations else 0.0,
            "final_beta": float(inv_prob.beta),
        }
        return sigma, history
