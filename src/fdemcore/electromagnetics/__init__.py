"""
=================================================================
Base EM (:mod:`src.fdemcore.electromagnetics`)
=================================================================

fdemcore 中保留的电磁学模块（SimPEG 精简派生）：

  base.py       —— BaseEMSimulation / BaseEMSrc
  base_1d.py    —— BaseEM1DSimulation
  utils/        —— 源项、波数、1D Hankel 工具
  frequency_domain —— 频域电磁正反演（1D 层状解析 + 3D/轴对称 FVM）

已裁剪（相对上游 SimPEG）：time_domain、natural_source、static、vrm、
potential_fields 等与本项目无关的物理模块。
"""

from scipy.constants import mu_0, epsilon_0

from . import utils
from . import frequency_domain
from . import base
from . import base_1d
