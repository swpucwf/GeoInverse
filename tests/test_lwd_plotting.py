"""lwd_fdem.plotting 可视化冒烟测试：模型图 / 网格图 / 响应图三件套。

验证 plot_forward_views 对 1D 层状与 2D 轴对称几何均能产出三张 PNG，
且 LWDFDEMForward.plot_views 便捷方法可用。使用非交互 Agg 后端。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import matplotlib
matplotlib.use("Agg")
# 若系统有中文字体则启用，避免示例同款 CJK Glyph 缺失警告
from matplotlib import font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
_cjk = next((n for n in ("SimHei", "Microsoft YaHei") if n in _avail), None)
if _cjk:
    matplotlib.rcParams["font.sans-serif"] = [_cjk]
    matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
import pytest

from src.lwd_fdem import (
    LWDFDEMForward, LWDToolConfig, build_2d_cyl_mesh,
    plot_forward_views, plot_forward_figure,
)


def _cfg(**kw):
    base = dict(frequencies=(1e4, 1e5), spacings=np.array([0.5, 1.0]),
                components=("real", "imag"))
    base.update(kw)
    return LWDToolConfig(**base)


def test_plot_views_1d(tmp_path):
    fwd = LWDFDEMForward("1d", _cfg(field_kind="h", secondary=True,
                                    receiver_geometry="radial"),
                         layer_thicknesses=np.array([2.0, 4.0]))
    sigma = np.array([0.25, 0.01, 0.5])
    out = plot_forward_views(fwd, sigma, out_dir=tmp_path, prefix="t1d")
    for key in ("model", "mesh", "response"):
        assert (tmp_path / f"t1d_{key}.png").exists()
        assert out[key] is not None
    plt_mod = pytest.importorskip("matplotlib.pyplot")
    plt_mod.close("all")


def test_forward_figure_1d(tmp_path):
    fwd = LWDFDEMForward("1d", _cfg(field_kind="h", secondary=True,
                                    receiver_geometry="radial"),
                         layer_thicknesses=np.array([2.0, 4.0]))
    sigma = np.array([0.25, 0.01, 0.5])
    p = plot_forward_figure(fwd, sigma, out_dir=tmp_path, prefix="plate1d")
    import pathlib
    assert pathlib.Path(p).exists()
    pytest.importorskip("matplotlib.pyplot").close("all")


def test_forward_figure_2d_cyl(tmp_path):
    mesh = build_2d_cyl_mesh(r_core=5.0, z_core=12.0, cell_size=1.5,
                             n_pad=3, pad_factor=1.4)
    fwd = LWDFDEMForward("2d_cyl", _cfg(field_kind="b", secondary=True,
                                        receiver_geometry="axial"), mesh=mesh)
    sigma = 0.1 * np.ones(mesh.n_cells)
    p = fwd.plot_figure(sigma, out_dir=tmp_path, prefix="plate2d")
    import pathlib
    assert pathlib.Path(p).exists()
    pytest.importorskip("matplotlib.pyplot").close("all")


def test_plot_views_2d_cyl(tmp_path):
    mesh = build_2d_cyl_mesh(r_core=5.0, z_core=12.0, cell_size=1.5,
                             n_pad=3, pad_factor=1.4)
    fwd = LWDFDEMForward("2d_cyl", _cfg(field_kind="b", secondary=True,
                                        receiver_geometry="axial"), mesh=mesh)
    sigma = 0.1 * np.ones(mesh.n_cells)
    out = fwd.plot_views(sigma, out_dir=tmp_path, prefix="t2d")
    for key in ("model", "mesh", "response"):
        assert (tmp_path / f"t2d_{key}.png").exists()
    plt_mod = pytest.importorskip("matplotlib.pyplot")
    plt_mod.close("all")
