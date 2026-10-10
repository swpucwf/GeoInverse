"""
========================================================
fdemcore —— 频域电磁有限体积法计算核心
========================================================

本包是 SimPEG (https://simpeg.xyz, MIT License) v0.25.1.dev3 的**精简派生**，
仅保留频域电磁（FDEM）正反演所需模块；无关物理（static/flow/seismic/nsem/
vrm/potential_fields/dask/meta 等）已裁剪。派生说明见 THIRD_PARTY_LICENSES.md
与 docs/ARCHITECTURE.md。

主要再导出：
  maps / data / data_misfit / regularization / optimization / directives /
  inverse_problem / inversion / electromagnetics.frequency_domain / utils
"""

# 上游 SimPEG 版权头（本包派生自其代码，MIT License）：
# Copyright (c) 2013 - 2024, SimPEG Team, https://simpeg.xyz

import discretize

from . import typing
from .data import Data, SyntheticData
from . import data_misfit
from . import directives
from . import maps
from . import models
from . import inverse_problem
from . import inversion
from . import regularization
from . import survey
from . import simulation
from .base import BasePDESimulation

from . import utils
from .utils import mkvc
from .utils.solver_utils import (
    SolverWrapD,
    SolverWrapI,
    Solver,
    SolverCG,
    SolverDiag,
    SolverLU,
    SolverBiCG,
)

from . import electromagnetics

__author__ = "SimPEG Team (original), LWD-EM-Inv Team (derived)"
__license__ = "MIT (upstream), Apache-2.0 (project)"
__copyright__ = "2013 - 2024, SimPEG Team, https://simpeg.xyz"

__version__ = "0.25.1.dev3-geoinverse.1"
