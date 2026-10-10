"""LWD 频域电磁正演统一入口。

按几何维度（1D 层状 / 2D 轴对称 / 3D 直角）自动选择 fdemcore 中的
Simulation1DLayered（Hankel 解析）或 FVM 体元仿真（e/b/h/j-form），
封装 dpred / Jvec / Jtvec 与响应矩阵变换（对应决策清单 A1/B1/B2/F）。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np

from ..fdemcore import maps
from ..fdemcore.electromagnetics import frequency_domain as fdem
from ..fdemcore.utils import validate_string
from .survey import LWDToolConfig, build_survey, data_order

logger = logging.getLogger(__name__)

Geometry = Literal["1d", "2d_cyl", "3d"]

# 场量到 FVM 离散形式的默认映射
_DEFAULT_FORM = {"e": "e", "b": "b", "h": "h", "j": "j"}

_SIM_CLASSES = {
    "e": fdem.Simulation3DElectricField,
    "b": fdem.Simulation3DMagneticFluxDensity,
    "h": fdem.Simulation3DMagneticField,
    "j": fdem.Simulation3DCurrentDensity,
}


class LWDFDEMForward:
    """LWD 频域电磁波正演求解器。

    Attributes:
        geometry: 几何维度，'1d' / '2d_cyl' / '3d'。
        tool: LWD 仪器配置（LWDToolConfig）。
        survey: 构建好的 fdem.Survey。
        mesh: FVM 网格（1D 几何时为 None）。
        form: 实际使用的离散形式，'1d'/'e'/'b'/'h'/'j'。

    Args:
        geometry: '1d' 要求 tool.receiver_geometry='radial'（Hankel 变换以
            径向偏移为自变量）；'2d_cyl'/'3d' 需提供 mesh。
        tool: 仪器与观测系统配置。
        layer_thicknesses: 1D 几何的层厚度 (m)，自上而下，半空间另计。
        mesh: 2D/3D 几何的 discretize 网格。
        form: 覆盖默认离散形式（'e'/'b'/'h'/'j'）；None 时按 tool.field_kind 自动选择。
        sigma_map: 电导率映射；None 时默认恒等映射（模型向量 = 单元电导率 S/m）。
        solver: 线性求解器类；None 时使用 fdemcore 默认直接法。

    Raises:
        ValueError: 几何/网格/参数不匹配时。
    """

    def __init__(self, geometry: str, tool: LWDToolConfig,
                 layer_thicknesses: Optional[np.ndarray] = None,
                 mesh=None, form: Optional[str] = None,
                 sigma_map: Optional[maps.IdentityMap] = None,
                 solver=None):
        validate_string("geometry", geometry, string_list=("1d", "2d_cyl", "3d"))
        self.geometry = geometry
        self.tool = tool

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
        self._simulation = None

    # ------------------------------------------------------------------ #
    # 仿真构建
    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        """模型参数个数（1D：层数含半空间；2D/3D：单元数）。"""
        if self.geometry == "1d":
            return len(self.layer_thicknesses) + 1
        return int(self.mesh.n_cells)

    def default_sigma_map(self) -> maps.IdentityMap:
        """默认电导率映射（恒等，nP=n_params）。"""
        return maps.IdentityMap(nP=self.n_params)

    def build_simulation(self, sigma_map: Optional[maps.IdentityMap] = None):
        """构建（或重建）仿真对象。

        Args:
            sigma_map: 覆盖构造时的电导率映射（反演层用 ExpMap 时调用）。

        Returns:
            Simulation1DLayered 或对应 form 的 3D FVM 仿真对象。
        """
        s_map = sigma_map if sigma_map is not None else self._sigma_map
        if s_map is None:
            s_map = self.default_sigma_map()
        if self.geometry == "1d":
            sim = fdem.Simulation1DLayered(
                survey=self.survey,
                thicknesses=self.layer_thicknesses,
                sigmaMap=s_map,
            )
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
        """计算模型响应。

        Args:
            sigma: 电导率模型向量 (S/m)，形状 (n_params,)。

        Returns:
            预测数据向量，形状 (nD,)，排列顺序 频率->收发距->分量。
        """
        return self.simulation.dpred(np.asarray(sigma, dtype=float))

    def jvec(self, sigma: np.ndarray, v: np.ndarray) -> np.ndarray:
        """计算方向导数 J @ v。"""
        return self.simulation.Jvec(np.asarray(sigma, dtype=float),
                                    np.asarray(v, dtype=float))

    def jtvec(self, sigma: np.ndarray, v: np.ndarray) -> np.ndarray:
        """计算伴随方向导数 J.T @ v。"""
        return self.simulation.Jtvec(np.asarray(sigma, dtype=float),
                                     np.asarray(v, dtype=float))

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
