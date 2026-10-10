"""lwd_fdem —— 随钻电磁波频域正反演业务层。

基于 fdemcore（SimPEG 精简派生）提供 LWD 专用封装：
  meshes    —— 网格生成与趋肤深度校核
  survey    —— 多频多偏移距线圈观测系统
  forward   —— 1D/2D 轴对称/3D 正演统一入口
  plotting  —— 模型图/网格图/响应图可视化
  inversion —— Occam 风格反演管线
  dataio    —— 测井数据 I/O 与不确定度

用法示例见 examples/lwd_fdem_*.py 与 docs/ARCHITECTURE.md。
"""

from .dataio import add_synthetic_noise, load_lwd_csv, make_data_object
from .forward import LWDFDEMForward
from .inversion import LWDFDEMInversion
from .meshes import (
    build_1d_mesh,
    build_2d_cyl_mesh,
    build_3d_tensor_mesh,
    mesh_quality_report,
    recommend_cell_size,
    skin_depth,
)
from .plotting import (
    plot_forward_views,
    plot_mesh,
    plot_model,
    plot_response,
)
from .survey import LWDToolConfig, build_receivers, build_survey, data_order

__all__ = [
    "LWDToolConfig",
    "build_survey",
    "build_receivers",
    "data_order",
    "LWDFDEMForward",
    "LWDFDEMInversion",
    "skin_depth",
    "recommend_cell_size",
    "build_1d_mesh",
    "build_2d_cyl_mesh",
    "build_3d_tensor_mesh",
    "mesh_quality_report",
    "add_synthetic_noise",
    "make_data_object",
    "load_lwd_csv",
    "plot_model",
    "plot_mesh",
    "plot_response",
    "plot_forward_views",
]
