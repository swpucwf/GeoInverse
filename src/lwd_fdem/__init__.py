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
from .apparent import (apparent_resistivity_from_hz, apparent_resistivity_matrix,
                       apparent_resistivity_from_forward)
from .advanced import (compute_sensitivity_matrix, compute_sensitivity_diagonal,
                       forward_parallel_frequencies, export_las2)
from .doi import (doi_1d_sensitivity, doi_skin_depth_estimate,
                  doi_resolution_matrix, doi_summary)
from .window import (sliding_window_inversion_1d, simple_window_builder,
                     simple_inversion_runner)
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
    plot_forward_figure,
    plot_forward_views,
    plot_inversion_figure,
    plot_mesh,
    plot_model,
    plot_response,
    pub_style,
    set_cjk_rcparams,
    fix_log_labels,
    apply_rcparams,
    save_figure,
    case_fig_dir,
    t,
    LBL,
)
from .survey import LWDToolConfig, build_receivers, build_survey, data_order
from .tensor import (build_tensor_receivers, build_tensor_survey,
                     tensor_data_order)

__all__ = [
    "LWDToolConfig",
    "build_survey",
    "build_receivers",
    "data_order",
    "build_tensor_receivers",
    "build_tensor_survey",
    "tensor_data_order",
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
    "apparent_resistivity_from_hz",
    "apparent_resistivity_matrix",
    "apparent_resistivity_from_forward",
    "compute_sensitivity_matrix",
    "compute_sensitivity_diagonal",
    "forward_parallel_frequencies",
    "export_las2",
    "doi_1d_sensitivity",
    "doi_skin_depth_estimate",
    "doi_resolution_matrix",
    "doi_summary",
    "sliding_window_inversion_1d",
    "simple_window_builder",
    "simple_inversion_runner",
    "plot_model",
    "plot_mesh",
    "plot_response",
    "plot_forward_views",
    "plot_forward_figure",
    "plot_inversion_figure",
    "pub_style",
    "set_cjk_rcparams",
    "fix_log_labels",
    "apply_rcparams",
    "save_figure",
    "case_fig_dir",
    "t",
    "LBL",
]
