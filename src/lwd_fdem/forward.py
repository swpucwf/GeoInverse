"""LWD 频域电磁正演统一入口。

按几何维度（1D 层状 / 2D 轴对称 / 3D 直角）自动选择 fdemcore 中的
Simulation1DLayered（Hankel 解析）或 FVM 体元仿真（e/b/h/j-form），
封装 dpred / Jvec / Jtvec 与响应矩阵变换（对应决策清单 A1/B1/B2/F）。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..fdemcore import maps
from ..fdemcore.electromagnetics import frequency_domain as fdem
from ..fdemcore.utils import validate_string
from .survey import LWDToolConfig, build_survey, build_survey_tilted, data_order

logger = logging.getLogger(__name__)

Geometry = Literal["1d", "2d_cyl", "2d_tensor", "1d_tensor", "3d"]

# 场量到 FVM 离散形式的默认映射
_DEFAULT_FORM = {"e": "e", "b": "b", "h": "h", "j": "j"}

_SIM_CLASSES = {
    "e": fdem.Simulation3DElectricField,
    "b": fdem.Simulation3DMagneticFluxDensity,
    "h": fdem.Simulation3DMagneticField,
    "j": fdem.Simulation3DCurrentDensity,
}


class _AnisotropicSigmaMap(maps.IdentityMap):
    """各向异性电导率映射：flat 模型向量 (n_cells*3,) → sigma (n_cells, 3)。

    *物理含义*: 反演/正演的模型向量必须是一维的——fdemcore 的
    ``simulation.model`` 经 ``props.validate_ndarray_with_shape(shape=('*',))``
    校验，只接受 1D 数组；而 discretize 的 ``get_edge_inner_product`` /
    ``get_face_inner_product`` 仅当 sigma 形状**严格**为 ``(n_cells, 3)`` 时
    才按对角各向异性张量解释。扁平 ``(3*n_cells,)`` 数组既不报错也不等价于
    ``(n_cells, 3)``：``make_property_tensor`` 内部用 ``mkvc``（Fortran 列主序）
    展平，会把按单元 C 序排列的分量错配到错误的方向分组，静默给出错误物理结果。
    本映射把"1D 模型向量"与"2D 张量 sigma"两层解耦：模型按单元 C 序排列
    ``[σ_xx, σ_yy, σ_zz]``，映射输出 ``(n_cells, 3)``。

    *使用方式*: ``LWDFDEMForward(anisotropic=True)`` 时由 ``default_sigma_map()``
    自动创建；反演中若需 ``ExpMap``，应写作
    ``_AnisotropicSigmaMap(nP=3*n_cells) * maps.ExpMap(nP=3*n_cells)``
    （本映射置于链末端，负责最终的形状变换）。

    Args:
        nP: 模型参数个数，须为 ``3 * n_cells``（``'*'`` 时按输入长度推断）。
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)
        self._permutation = None

    def _transform(self, m):
        """flat (n_cells*3,) → (n_cells, 3)，按单元 C 序解释分量。"""
        return np.asarray(m, dtype=float).reshape(-1, 3)

    @property
    def shape_output(self):
        """映射输出形状 ``(n_cells, 3)``。"""
        nP = self.nP
        if not isinstance(nP, (int, np.integer)):
            return ("*", 3)
        return (int(nP) // 3, 3)

    def _n_cells_from(self, m=None) -> int:
        """由 nP（或模型长度）推断单元数，并校验可被 3 整除。"""
        nP = self.nP
        if not isinstance(nP, (int, np.integer)):
            if m is None:
                raise ValueError(
                    "_AnisotropicSigmaMap 需要确定的 nP 才能构造导数矩阵"
                    "（当前 nP='*' 且未提供模型向量）。"
                )
            nP = int(np.asarray(m).ravel().size)
        nP = int(nP)
        if nP % 3 != 0:
            raise ValueError(
                f"各向异性映射的参数个数须为 3 的倍数（3*n_cells），当前 nP={nP}。"
            )
        return nP // 3

    def _permutation_matrix(self, m=None) -> sp.csr_matrix:
        """构造 (3*n_cells, nP) 置换矩阵 P，使 ``P @ m == mkvc(self._transform(m))``。

        discretize 的内积导数算子以**列主序扁平 sigma**
        （``[σ_xx 全体, σ_yy 全体, σ_zz 全体]``）为列空间，故导数矩阵不能是
        单位阵，而必须是该 C 序 → F 序的置换。
        """
        if self._permutation is None:
            n_cells = self._n_cells_from(m)
            nP = 3 * n_cells
            cell_idx = np.tile(np.arange(n_cells), 3)          # i
            comp_idx = np.repeat(np.arange(3), n_cells)        # j
            rows = comp_idx * n_cells + cell_idx               # F 序输出索引
            cols = cell_idx * 3 + comp_idx                     # C 序输入索引
            self._permutation = sp.csr_matrix(
                (np.ones(nP), (rows, cols)), shape=(nP, nP)
            )
        return self._permutation

    def deriv(self, m, v=None):
        """映射导数。

        ``v is None`` 时返回稀疏置换矩阵 (3*n_cells, nP)：fdemcore 的
        ``sigmaDeriv`` 以 ``mapping.deriv(model)`` 形式取矩阵，并与
        ``mesh.get_edge_inner_product_deriv`` 的列空间（列主序扁平 sigma）对接，
        因此这里必须给出 C 序模型 → F 序 sigma 的置换，而非单位阵。
        ``v`` 非 None 时返回同一矩阵与 ``v`` 的乘积，保持与矩阵形式自洽。
        """
        P = self._permutation_matrix(m)
        if v is not None:
            return P @ np.asarray(v, dtype=float).ravel()
        return P


class LWDFDEMForward:
    """LWD 频域电磁波正演求解器。

    本类是正演计算的统一入口，根据几何维度自动选择底层求解器：

    - **1D 层状介质**: 使用 ``Simulation1DLayered``（Hankel 解析解 + 层矩阵递推），
      速度最快，适用于垂直井层状地层模型。
    - **2D 轴对称**: 使用 ``Simulation3DMagneticField`` + ``CylindricalMesh``，
      FVM 离散，适用于含井眼/侵入带的轴对称模型。
    - **3D 直角**: 使用 ``Simulation3D*`` + ``TensorMesh``，FVM 离散，
      适用于倾斜井/非对称结构。

    主要方法:
        - ``run(model)``: 计算给定制模型的正演响应（返回 dpred）。
        - ``Jvec(model, v)``: 计算灵敏度矩阵与向量的乘积 J·v。
        - ``Jtvec(model, v)``: 计算灵敏度矩阵转置与向量的乘积 Jᵀ·v。
        - ``build_simulation()``: 构建底层 fdemcore Simulation 对象。

    Args:
        geometry: 几何维度。
            *物理含义*: 决定底层求解器类型与网格结构。
            *使用方式*: ``'1d'`` / ``'2d_cyl'`` / ``'3d'``。
            *应用场景*: 垂直井层状近似 → ``'1d'``；含井眼轴对称 → ``'2d_cyl'``；
            倾斜井/复杂构造 → ``'3d'``。

        tool: LWD 仪器与观测系统配置。
            *物理含义*: 描述发射源、接收器布局、观测频率等全部仪器参数。
            *使用方式*: 传入 ``LWDToolConfig`` 实例。
            *应用场景*: 所有几何维度均必需。

        layer_thicknesses: 层厚度数组 (m)。
            *物理含义*: 1D 层状介质中每层的厚度，自上而下排列；最底层为半空间。
            *使用方式*: 1D numpy 数组，如 ``np.array([5.0, 10.0])`` 表示 3 层
            （第 1 层厚 5 m，第 2 层厚 10 m，第 3 层半空间）。
            *应用场景*: 仅 ``geometry='1d'`` 时必需；2D/3D 几何忽略。

        mesh: 计算网格对象。
            *物理含义*: 离散化后的空间网格，用于 FVM 求解。
            *使用方式*: ``discretize.CylindricalMesh``（2D）或
            ``discretize.TensorMesh``（3D）。
            *应用场景*: 仅 ``geometry='2d_cyl'`` 或 ``'3d'`` 时必需。
            可用 ``lwd_fdem.meshes`` 模块中的工具函数生成。

        form: 离散形式（FVM 场量未知量类型）。
            *物理含义*: 决定 FVM 离散后求解的线性方程组以哪种场量为未知量。
            *使用方式*: ``'e'``（电场）/ ``'b'``（磁通密度）/ ``'h'``（磁场）/
            ``'j'``（电流密度）/ ``None``。
            *应用场景*: ``None``（默认）时按 ``tool.field_kind`` 自动匹配
            （h→'h', b→'b', e→'e', j→'j'）；一般无需手动指定。
            不同 form 的数值精度和边界条件处理略有差异。

        sigma_map: 电导率映射。
            *物理含义*: 从模型向量到网格单元电导率的映射关系。
            *使用方式*: ``fdemcore.maps.IdentityMap``（默认）或其他映射。
            ``None`` 时使用恒等映射（模型向量 = 单元电导率 S/m）。
            *应用场景*: 反演中常使用 ``ExpMap``（模型向量 = ln(σ)）
            以保证电导率始终为正。

        solver: 线性求解器类。
            *物理含义*: 求解 FVM 线性方程组 A·x = b 的方法。
            *使用方式*: 传入求解器类（如 ``scipy.sparse.linalg.spsolve``）；
            ``None`` 时使用 fdemcore 默认直接法。
            *应用场景*: 大规模 3D 问题可换用迭代求解器以节省内存。

        cole_cole: Cole-Cole 频散参数（仅 1D 几何生效）。
            *物理含义*: 描述岩石极化效应引起的电导率随频率变化（频散）。
            基于 Pelton (1978) 公式：
            ``σ(ω) = σ∞ · [1 − η·(1 − 1/(1+(iωτ)^c))]``
            其中 σ∞ 为无穷频率电导率，η 为充电子率，τ 为弛豫时间，c 为频率指数。
            *使用方式*: 字典格式 ``{'eta': float|array, 'tau': float|array,
            'c': float|array}``。标量时各层相同；数组时长度须等于层数。
            ``None`` 时无频散（默认，即电导率与频率无关）。
            *应用场景*: 激发极化 (IP) 效应建模；页岩/金属矿等含极化矿物地层。
            参数范围：0 ≤ η < 1（充电子率），τ > 0（弛豫时间, s），0 < c < 1（频率指数）。

    Raises:
        ValueError: 几何/网格/参数不匹配时。

    Attributes:
        geometry: 几何维度，``'1d'`` / ``'2d_cyl'`` / ``'3d'``。
        tool: LWD 仪器配置（``LWDToolConfig``）。
        survey: 构建好的 ``fdem.Survey``。
        mesh: FVM 网格（1D 几何时为 ``None``）。
        form: 实际使用的离散形式，``'1d'`` / ``'e'`` / ``'b'`` / ``'h'`` / ``'j'``。
    """

    def __init__(self, geometry: str, tool: LWDToolConfig,
                 layer_thicknesses: Optional[np.ndarray] = None,
                 mesh=None, form: Optional[str] = None,
                 sigma_map: Optional[maps.IdentityMap] = None,
                 solver=None,
                 cole_cole: Optional[Dict] = None,
                 anisotropic: bool = False):
        validate_string("geometry", geometry,
                        string_list=("1d", "2d_cyl", "2d_tensor",
                                     "1d_tensor", "3d"))
        self.geometry = geometry
        self.tool = tool
        self._anisotropic = anisotropic
        self._layer_thicknesses = None

        # ---- 维度能力校验（仅对新参数触发，不影响既有路径） ----
        if geometry == '2d_cyl':
            if (tool.inclination != 0.0 or tool.azimuth != 0.0
                    or tool.tool_face != 0.0):
                raise NotImplementedError(
                    "2D 轴对称路径 (CylindricalMesh) 不支持倾斜仪器。"
                    "轴对称假设要求源必须为 z 方向。"
                    "请使用 geometry='2d_tensor'、'1d_tensor' 或 '3d'。"
                )
        if geometry == '1d' and anisotropic:
            raise NotImplementedError(
                "1D Hankel 路径暂不支持各向异性（底层 geoana "
                "rTE_forward 仅接受标量 sigma）。"
                "各向异性模拟请使用 geometry='1d_tensor'、"
                "'2d_tensor' 或 '3d'。"
            )

        if geometry == "1d":
            if tool.receiver_geometry != "radial":
                raise ValueError(
                    "1D 层状仿真基于径向 Hankel 变换，接收器必须径向偏移"
                    "（receiver_geometry='radial'）；轴向偏移请改用 '2d_cyl'。"
                )
            thicknesses = np.asarray(layer_thicknesses, dtype=float)
            if thicknesses.ndim != 1 or np.any(thicknesses <= 0):
                raise ValueError("geometry='1d' 时 layer_thicknesses 必须为正的 1D 数组 (m)。")
            self.layer_thicknesses = thicknesses
            self.mesh = None
            self.form = "1d"
        elif geometry == "1d_tensor":
            # 1D 层状参数化 + 3D TensorMesh FVM 求解
            thicknesses = np.asarray(layer_thicknesses, dtype=float)
            if thicknesses.ndim != 1 or np.any(thicknesses <= 0):
                raise ValueError(
                    "geometry='1d_tensor' 时 layer_thicknesses "
                    "必须为正的 1D 数组 (m)。"
                )
            self._layer_thicknesses = thicknesses
            self.layer_thicknesses = thicknesses
            if mesh is None:
                from .meshes import build_3d_tensor_mesh
                z_extent = float(np.sum(thicknesses)) * 4.0
                x_extent = z_extent * 0.5
                mesh = build_3d_tensor_mesh(
                    core_extent=(x_extent, x_extent * 0.1, z_extent),
                    frequencies=tool.frequencies,
                    conductivities=[0.001, 1.0],
                )
            self.mesh = mesh
            self.form = form or _DEFAULT_FORM[tool.field_kind]
            validate_string("form", self.form,
                            string_list=("e", "b", "h", "j"))
            # 预计算层界面深度
            self._layer_boundaries = np.concatenate(
                [[0.0], np.cumsum(thicknesses)]
            )
        else:
            if mesh is None:
                raise ValueError(f"geometry={geometry!r} 时必须提供 mesh。")
            self.layer_thicknesses = None
            self.mesh = mesh
            self.form = form or _DEFAULT_FORM[tool.field_kind]
            validate_string("form", self.form, string_list=("e", "b", "h", "j"))

        self.survey = build_survey(tool, self.form)
        self._sigma_map = sigma_map
        self._solver = solver
        self._cole_cole = self._validate_cole_cole(cole_cole, geometry)
        self._simulation = None
        # 新路径使用倾斜仪器 survey 构建器覆盖
        if geometry in ('1d_tensor', '2d_tensor'):
            self.survey = build_survey_tilted(tool, self.form)
        # 2d_tensor 路径需要计算 2D 参数数量
        self._n_2d_params: Optional[int] = None
        if geometry == '2d_tensor':
            core_w = getattr(self.mesh, '_lwd_core_widths', None)
            if core_w is not None:
                nx_core = len(core_w[0])
                nz_core = len(core_w[2])
                self._n_2d_params = nx_core * nz_core
            else:
                self._n_2d_params = (self.mesh.shape_cells[0]
                                     * self.mesh.shape_cells[2])

    # ------------------------------------------------------------------ #
    # Cole-Cole 参数校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_cole_cole(cc: Optional[Dict], geometry: str) -> Optional[Dict]:
        """校验 Cole-Cole 参数字典，返回规范化后的字典或 None。"""
        if cc is None:
            return None
        if geometry != "1d":
            raise ValueError(
                "Cole-Cole 频散目前仅支持 1D 层状几何"
                "（Simulation1DLayered 内置 Pelton 公式）；"
                "2D/3D 仿真尚未实现频率依赖 σ(ω)。"
            )
        required = {"eta", "tau", "c"}
        missing = required - set(cc.keys())
        if missing:
            raise ValueError(f"cole_cole 字典缺少键: {missing}")
        out = {}
        for key in ("eta", "tau", "c"):
            v = np.atleast_1d(np.asarray(cc[key], dtype=float))
            if v.ndim != 1:
                raise ValueError(f"cole_cole['{key}'] 须为标量或 1D 数组")
            out[key] = v
        # 物理范围检查
        if np.any(out["eta"] < 0) or np.any(out["eta"] >= 1):
            raise ValueError("eta (充电子率) 须满足 0 ≤ eta < 1")
        if np.any(out["tau"] <= 0):
            raise ValueError("tau (弛豫时间) 须为正数")
        if np.any(out["c"] <= 0) or np.any(out["c"] >= 1):
            raise ValueError("c (频率指数) 须满足 0 < c < 1")
        return out

    @property
    def cole_cole(self) -> Optional[Dict]:
        """Cole-Cole 频散参数（只读）。"""
        return self._cole_cole

    # ------------------------------------------------------------------ #
    # 仿真构建
    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        """模型参数个数（1D：层数含半空间；2D/3D：单元数）。"""
        if self.geometry == "1d":
            return len(self.layer_thicknesses) + 1
        if self.geometry == "1d_tensor":
            n_layers = len(self._layer_thicknesses) + 1
            return n_layers * 2 if self._anisotropic else n_layers
        if self.geometry == "2d_tensor":
            if self._anisotropic:
                return self._n_2d_params * 2
            return self._n_2d_params
        if self._anisotropic:
            return int(self.mesh.n_cells) * 3
        return int(self.mesh.n_cells)

    def default_sigma_map(self) -> maps.IdentityMap:
        """默认电导率映射。

        各向同性路径返回 ``IdentityMap(nP=n_params)``（模型向量 = 单元电导率）。

        各向异性路径返回 ``_AnisotropicSigmaMap(nP=n_cells*3)``：模型向量保持
        flat 1D（fdemcore 的 ``simulation.model`` 只接受 1D），而映射输出严格为
        ``(n_cells, 3)`` 的张量 sigma，供 discretize 的
        ``get_edge_inner_product`` 正确识别为对角各向异性。

        对于 1d_tensor 路径，sigmaMap 作用于网格级 sigma
        （dpred 先做层→网格映射再调用 simulation）。
        """
        if self.geometry == '1d_tensor':
            # dpred 先调 _build_sigma_from_layers_1d 转为网格级 sigma，
            # 然后传给 simulation.dpred，所以 sigmaMap 用网格维度
            if self._anisotropic:
                return _AnisotropicSigmaMap(nP=int(self.mesh.n_cells) * 3)
            return maps.IdentityMap(nP=int(self.mesh.n_cells))
        if self._anisotropic and self.geometry in ('2d_tensor', '3d'):
            return _AnisotropicSigmaMap(nP=int(self.mesh.n_cells) * 3)
        return maps.IdentityMap(nP=self.n_params)

    def build_simulation(self, sigma_map: Optional[maps.IdentityMap] = None):
        """构建（或重建）底层 fdemcore 仿真对象。

        *物理含义*: 根据当前几何维度和仪器配置，创建对应的 Simulation 实例，
        内部完成网格内积矩阵、源场向量、接收器投影等初始化。
        *使用方式*: 通常无需手动调用（``dpred``/``jvec`` 会自动触发）。
        反演中需要更换 sigma_map（如从 IdentityMap 换为 ExpMap）时可显式调用。
        *应用场景*: 反演初始化、参数扫描时重建仿真。

        Args:
            sigma_map: 覆盖构造时的电导率映射（反演层用 ExpMap 时调用）。

        Returns:
            Simulation1DLayered（1D）或对应 form 的 3D FVM 仿真对象。
        """
        s_map = sigma_map if sigma_map is not None else self._sigma_map
        if s_map is None:
            s_map = self.default_sigma_map()
        if self.geometry == "1d":
            kw_1d = dict(
                survey=self.survey,
                thicknesses=self.layer_thicknesses,
                sigmaMap=s_map,
            )
            if self._cole_cole is not None:
                kw_1d["eta"] = self._cole_cole["eta"]
                kw_1d["tau"] = self._cole_cole["tau"]
                kw_1d["c"] = self._cole_cole["c"]
            sim = fdem.Simulation1DLayered(**kw_1d)
        else:
            cls = _SIM_CLASSES[self.form]
            kwargs = dict(mesh=self.mesh, survey=self.survey, sigmaMap=s_map)
            if self._solver is not None:
                kwargs["solver"] = self._solver
            sim = cls(**kwargs)
        return sim

    @property
    def simulation(self):
        """惰性创建并缓存的仿真对象（默认映射）。"""
        if self._simulation is None:
            self._simulation = self.build_simulation()
        return self._simulation

    # ------------------------------------------------------------------ #
    # 正演与导数
    # ------------------------------------------------------------------ #
    def dpred(self, sigma: np.ndarray) -> np.ndarray:
        """计算模型的正演预测数据。

        *物理含义*: 给定地层电导率分布，求解 Maxwell 方程得到各频率、各收发距、
        各分量的电磁场响应。
        *使用方式*: 传入电导率模型向量 (S/m)，返回预测数据向量。
        *应用场景*: 正演模拟、反演目标函数计算、灵敏度分析。

        Args:
            sigma: 电导率模型向量 (S/m)。
                1D: 形状 (n_layers+1,)，自上而下，末层为半空间。
                1d_tensor: 形状 (n_layers,) 或 (n_layers*2,) 各向异性时。
                2D/3D: 形状 (n_cells,) 或 (n_cells*3,) 各向异性时；
                各向异性也接受 (n_cells, 3) 二维数组（内部自动展平，
                形状还原由 sigmaMap 负责）。

        Returns:
            预测数据向量，形状 (nD,)。
            排列顺序：频率 → 收发距 → 分量（与 ``data_order`` 一致）。
        """
        sigma = np.asarray(sigma, dtype=float)
        # 1d_tensor 路径：层参数 → 网格 sigma
        if self.geometry == '1d_tensor':
            sigma_grid = self._build_sigma_from_layers_1d(sigma)
            return self.simulation.dpred(self._as_flat_model(sigma_grid))
        # 其余路径：模型向量必须保持 flat 1D 交给 simulation，
        # 各向异性的 (n_cells, 3) 形状还原由 sigmaMap 完成。
        return self.simulation.dpred(self._as_flat_model(sigma))

    def _as_flat_model(self, sigma: np.ndarray) -> np.ndarray:
        """把待传入 ``simulation.dpred`` 的模型压平为 1D，并对各向异性做形状断言。

        fdemcore 的 ``simulation.model`` 只接受 1D 数组（``shape=('*',)``）；
        而各向异性时 discretize 要求 sigmaMap 输出严格为 ``(n_cells, 3)``，
        否则扁平数组会被静默接受却给出错误物理结果。故在此统一压平并校验。
        """
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim > 1:
            if not self._anisotropic:
                raise ValueError(
                    "各向同性路径的模型向量必须为 1D，"
                    f"实际得到 shape={sigma.shape}。"
                )
            sigma = sigma.ravel()
        if self._anisotropic and self.mesh is not None:
            self._assert_tensor_sigma_shape(sigma)
        return sigma

    def _assert_tensor_sigma_shape(self, sigma_flat: np.ndarray) -> None:
        """断言各向异性 sigmaMap 输出严格为 (n_cells, 3)。

        Args:
            sigma_flat: flat 模型向量，形状 (n_cells*3,)。

        Raises:
            ValueError: sigmaMap 输出形状不是 (n_cells, 3) 时。
        """
        s_map = getattr(self.simulation, "sigmaMap", None)
        if s_map is None:
            return
        expected = (int(self.mesh.n_cells), 3)
        mapped = s_map * np.asarray(sigma_flat, dtype=float).ravel()
        if getattr(mapped, "shape", None) != expected:
            raise ValueError(
                f"各向异性正演要求 sigmaMap 输出形状为 {expected}，实际得到 "
                f"{getattr(mapped, 'shape', None)}。discretize 的 "
                "get_edge_inner_product 对扁平 (3*n_cells,) 数组不报错，但会按"
                "列主序错配方向权重而静默给出错误结果；请使用 "
                "lwd_fdem.forward._AnisotropicSigmaMap（或以其包裹 ExpMap 等）。"
            )

    def jvec(self, sigma: np.ndarray, v: np.ndarray) -> np.ndarray:
        """计算灵敏度矩阵与向量的乘积 J·v（方向导数）。

        *物理含义*: J 是数据对模型的灵敏度矩阵（Jacobian），J·v 给出
        模型沿方向 v 扰动时数据的线性变化量。
        *使用方式*: 传入当前模型 sigma 和扰动方向 v（形状同 sigma）。
        *应用场景*: Gauss-Newton 反演中计算模型更新步长；
        灵敏度分析中评估特定参数变化对数据的影响。

        Args:
            sigma: 当前电导率模型 (S/m)。
            v: 扰动方向向量，形状同 sigma。

        Returns:
            J·v，形状 (nD,)。
        """
        return self.simulation.Jvec(self._as_flat_model(sigma),
                                    np.asarray(v, dtype=float).ravel())

    def jtvec(self, sigma: np.ndarray, v: np.ndarray) -> np.ndarray:
        """计算灵敏度矩阵转置与向量的乘积 Jᵀ·v（伴随方向导数）。

        *物理含义*: Jᵀ 是灵敏度矩阵的转置，Jᵀ·v 将数据空间残差映射回模型空间，
        得到梯度方向。
        *使用方式*: 传入当前模型 sigma 和数据空间向量 v（形状 nD）。
        *应用场景*: 反演目标函数梯度计算 ∇φ = Jᵀ·Wᵀ·W·(d_pred - d_obs)；
        伴随场法高效计算梯度。

        Args:
            sigma: 当前电导率模型 (S/m)。
            v: 数据空间向量，形状 (nD,)。

        Returns:
            Jᵀ·v，形状同 sigma。
        """
        return self.simulation.Jtvec(self._as_flat_model(sigma),
                                     np.asarray(v, dtype=float))

    # ------------------------------------------------------------------ #
    # 1d_tensor 层状参数 → 网格 sigma 映射
    # ------------------------------------------------------------------ #
    def _assign_layers(self, z_centers: np.ndarray) -> np.ndarray:
        """根据 cell 中心 z 坐标确定所属层号。

        层界面深度由 _layer_boundaries 定义（从地表 z=0 向下累积）。
        网格 z 轴向上为正，因此需要将 z_centers 转换为深度。

        Args:
            z_centers: 网格单元中心 z 坐标数组 (m)。

        Returns:
            层索引数组，范围 [0, n_layers-1]。
        """
        # 网格 z 向上为正，深度 = -z（假定地表在 z=0）
        depths = -z_centers
        # searchsorted: 找到每个 depth 属于哪一层
        # boundaries = [0, h1, h1+h2, ...]  共 n_layers 个边界
        idx = np.searchsorted(self._layer_boundaries, depths, side='right') - 1
        # 裁剪到合法范围
        n_layers = len(self._layer_thicknesses) + 1
        return np.clip(idx, 0, n_layers - 1)

    def _build_sigma_from_layers_1d(self, m: np.ndarray) -> np.ndarray:
        """将 1D 层状模型参数映射到 3D 网格的 sigma 数组。

        Args:
            m: 模型参数向量。各向同性时 shape=(n_layers,)，
                各向异性时 shape=(n_layers*2,)，前半为 σh 后半为 σv。

        Returns:
            各向同性: shape (n_cells,) 的 sigma 数组。
            各向异性: shape (n_cells, 3) 的 sigma 数组 [σh, σh, σv]；
            调用方（``dpred``）会将其 C 序展平为 (n_cells*3,) 后交给
            simulation，形状还原由 ``_AnisotropicSigmaMap`` 完成。
        """
        m = np.asarray(m, dtype=float)
        n_layers = len(self._layer_thicknesses) + 1
        if self._anisotropic:
            sigma_h = m[:n_layers]
            sigma_v = m[n_layers:]
        else:
            sigma_h = m
            sigma_v = m  # 未使用

        z_centers = self.mesh.cell_centers[:, 2]
        layer_idx = self._assign_layers(z_centers)

        if self._anisotropic:
            sigma = np.zeros((self.mesh.n_cells, 3))
            sigma[:, 0] = sigma_h[layer_idx]
            sigma[:, 1] = sigma_h[layer_idx]
            sigma[:, 2] = sigma_v[layer_idx]
        else:
            sigma = sigma_h[layer_idx]
        return sigma

    # ------------------------------------------------------------------ #
    # 数据整形与绘图
    # ------------------------------------------------------------------ #
    def to_matrix(self, d: np.ndarray) -> np.ndarray:
        """将数据向量整形为 (n_freq, n_spacing, n_comp) 数组。"""
        order = data_order(self.tool)
        d = np.asarray(d)
        if d.size != int(np.prod(order)):
            raise ValueError(f"数据长度 {d.size} 与观测结构 {order} 不符。")
        return d.reshape(order)

    def plot_response(self, d: np.ndarray, plot_type: str = "amplitude",
                      ax=None) -> None:
        """按频率绘制响应-收发距曲线。

        Args:
            d: 数据向量 (nD,)。
            plot_type: 'amplitude' 绘制 |value|，'phase' 绘制带符号数值。
            ax: matplotlib 轴对象；None 时新建图形。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        mat = self.to_matrix(d)
        for i, freq in enumerate(self.tool.frequencies):
            for k, comp in enumerate(self.tool.components):
                values = mat[i, :, k]
                label = (f"{freq / 1e3:.0f} kHz {comp}"
                         if len(self.tool.components) > 1 else f"{freq / 1e3:.0f} kHz")
                if plot_type == "amplitude":
                    ax.semilogx(self.tool.spacings, np.abs(values), marker="o",
                                label=label)
                else:
                    ax.semilogx(self.tool.spacings, values, marker="o", label=label)
        ax.set_xlabel("收发距 (m)")
        ax.set_ylabel("幅值" if plot_type == "amplitude" else "分量值")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    def plot_views(self, sigma: np.ndarray, d: Optional[np.ndarray] = None,
                   out_dir=".", prefix: str = "fwd", labels=None,
                   analytic=None) -> dict:
        """一站式绘制并保存模型图 / 网格图 / 响应图（见 plotting.plot_forward_views）。"""
        from .plotting import plot_forward_views

        return plot_forward_views(self, sigma, d=d, out_dir=out_dir,
                                  prefix=prefix, labels=labels, analytic=analytic)

    def plot_figure(self, sigma: np.ndarray, d: Optional[np.ndarray] = None,
                    out_dir=".", prefix: str = "fwd_plate", dpi: int = 300,
                    labels=None, analytic=None) -> str:
        """合成 SCI 多联图版（模型/网格/响应）并保存，返回文件路径。

        见 plotting.plot_forward_figure。
        """
        from .plotting import plot_forward_figure

        return plot_forward_figure(self, sigma, d=d, out_dir=out_dir,
                                   prefix=prefix, dpi=dpi, labels=labels,
                                   analytic=analytic)
