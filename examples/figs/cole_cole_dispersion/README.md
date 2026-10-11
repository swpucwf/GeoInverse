# Cole-Cole 频散案例 (cole_cole_dispersion)

## 案例概述

**物理场景**: Cole-Cole 频散模型验证——展示 1D 仿真内置的 Pelton (1978) 频散公式。

**运行命令**: `python examples/cole_cole_dispersion.py`

---

## 图版说明

### fig_01_cole_cole_compare.png — Cole-Cole 对比
**表达内容**: 不同 Cole-Cole 参数 (η, τ, c) 下的频散曲线对比。展示复电阻率随频率的变化。

### fig_02_cole_cole_scan.png — Cole-Cole 参数扫描
**表达内容**: 参数空间扫描图，展示不同参数组合对频散响应的影响。

**物理意义**: 1D base_1d.py 已内置 eta/tau/c 参数（与 SimPEG 字节级一致），本案例验证频散效应。
