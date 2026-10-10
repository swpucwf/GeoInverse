# LWD-EM-Inv 架构说明

> 对应决策清单（`external/SimPEG-src/频域电磁波LWD正反演_定制决策清单.md`）的 H1/H2/H3 项。
> 工程路线：**精简代码库派** —— 将 SimPEG 中频域电磁（FDEM）有限体积法正反演所需模块
> 精简拷贝进 `src/fdemcore/`，项目不再依赖 pip 安装的 SimPEG。

## 1. SimPEG 正反演链路（被派生的核心架构）

```
discretize.Mesh (TensorMesh / CylindricalMesh)      —— 网格与差分算子 (grad, curl, div)
        │
sources + receivers + Survey                         —— 观测系统（频率、线圈、分量、data_type）
        │
BaseSimulation / BasePDESimulation                   —— 模型->物性->A矩阵->场->数据
   ├── fdem.Simulation1DLayered                      —— 1D 层状各向异性解析解 (Hankel/libdlf)
   ├── fdem.Simulation3DElectricField (e-form)       —— FVM 棱边电场，3D/轴对称
   ├── fdem.Simulation3DMagneticFluxDensity (b-form) —— FVM 面通量密度
   ├── fdem.Simulation3DMagneticField (h-form)
   └── fdem.Simulation3DCurrentDensity (j-form)
        │
Fields                                               —— 场对象缓存（二次场插值）
        │
maps (ExpMap/IdentityMap/Wires/Parametric*)          —— 反演参数 -> 网格物性模型
        │
data.Data (dobs + relative error)                    —— 观测数据与不确定度
        │
objective_function 体系:
   data_misfit.L2DataMisfit  +  regularization (Small/Smoothness/Sparse/WLS)
        │
optimization (InexactGaussNewton / ProjectedGNCG / BFGS)
        │
directives (BetaEstimate_ByEig / BetaSchedule / TargetMisfit /
            UpdatePreconditioner / UpdateSensitivityWeights / UpdateIRLS ...)
        │
inverse_problem.BaseInvProblem  +  inversion.BaseInversion   —— 反演外循环
```

FVM 离散（以 e-form 为例）：`C = curl 矩阵`，`M_sigma^-1` 为面质量逆矩阵，
求解 `(C^T M_mu C + i·omega·M_sigma) e = s`；接收器从 Fields 插值。
轴对称（CylindricalMesh）时 SimPEG 自动使用 m=0 模态的柱坐标 curl。

## 2. fdemcore 保留模块清单（决策清单 H2）

| `src/fdemcore/` 路径 | 来源（SimPEG 同名路径） | 作用 |
|---|---|---|
| `props.py`, `models.py`, `typing/` | 根级 | 物性描述符、模型容器、类型别名 |
| `survey.py`, `fields.py`, `simulation.py`, `base/` | 根级 | BaseSurvey/BaseRx/Fields/BaseSimulation/PdeSimulation |
| `data.py`, `data_misfit.py`, `objective_function.py` | 根级 | 数据容器、拟合差、目标函数基类 |
| `optimization.py` | 根级 | 优化器（含 ProjectedGNCG/InexactGaussNewton/BFGS） |
| `inverse_problem.py`, `inversion.py` | 根级 | 反演外循环组装 |
| `maps/` | 全包 | ExpMap/IdentityMap/Wires/Parametric* 等模型映射 |
| `regularization/` | 除 PGI 指令外全保留 | Small/Smoothness/Sparse/WLS/CrossGradient |
| `directives/` | `_directives/_regularization/_sim_directives` | beta 调度、预条件、IRLS、目标停止 |
| `utils/` | code/mat/model/mesh/curv/coord/counter/solver/plot/logger/warnings | 校验工具、mkvc、网格辅助、求解器默认值 |
| `electromagnetics/base.py`, `base_1d.py`, `utils/` | EM 基类与源项/1D Hankel 工具 | BaseEMSimulation、em1d_utils |
| `electromagnetics/frequency_domain/` | 全部 6 文件 | FDEM 源/收/survey/fields/1D/3D 仿真 |

裁剪模块（决策清单 H3）：`static`、`flow`、`seismic`、`nsem`（natural_source）、`vrm`、
`potential_fields`、`dask`、`meta`、`utils/drivers`、`utils/pgi_utils`（若导入链不需要则不复制）。

> 注：为减少对 `_directives.py` 的侵入式改写，`regularization/pgi.py` 若被其顶层导入引用则
> 一并保留（仅作为依赖休眠，不对外提供 PGI 反演配置）。

## 3. lwd_fdem 业务层（本项目原创）

| 模块 | 职责 |
|---|---|
| `meshes.py` | 1D 层厚 / 2D 圆柱（core+padding）/ 3D 直角网格构建；趋肤深度 δ/3 自动校核（决策清单 A3/A4） |
| `survey.py` | LWD 线圈观测系统构建器：多频 × 多收发距、MagDipole/CircularLoop、real/imag/amp/phase、ppm/secondary/total（决策清单 C/D） |
| `forward.py` | `LWDFDEMForward` 统一正演入口（决策清单 A1/B1/B2/F），封装 dpred/Jvec/Jadj |
| `inversion.py` | `LWDFDEMInversion` 反演管线组装（决策清单 E/G），默认 Occam 风格平滑反演 |
| `dataio.py` | 合成数据/观测数据 <-> `data.Data` 转换与不确定度估计（决策清单 H5） |

物理选型默认值：B1 = 全频域 Maxwell（保留位移电流，覆盖 20 kHz–2 MHz）；
B2 = e-form（默认）/ b-form（轴对称交叉验证用）；F1 = 自然边界 + padding；
F3 = 小规模直接法（SuperLU），大规模迭代 BiCGStab + 对角预条件。

## 4. 与既有模块的关系

- `src/forward`（一维解析）、`src/forward_2d`、`src/inversion`（自研 Occam 2D）保持原样，
  作为项目的解析解/快速模型通道；`fdemcore` 是独立的数值（FVM/1D 解析）通道。
- 验收判据（决策清单 I）见 `tests/`：1D 对标解析解、2D-3D 交叉验证、σ→0 一致性、
  `check_derivative` 梯度校验、合成数据反演回收。

## 5. 依赖边界

外部依赖仅：`numpy`、`scipy`、`matplotlib`、`discretize`、`pymatsolver`、`geoana`、`libdlf`
（+ 若 PGI 休眠保留则间接需要 `scikit-learn`，fdemcore 导入链验证后在 requirements 中固化）。
禁止 `fdemcore` 导入任何 `simpeg` 顶层包名（已全部改写为 `src.fdemcore`）。
