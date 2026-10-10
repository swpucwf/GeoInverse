# 项目知识库（Project Memory）

> 本文件是项目长期记忆在仓库内的**可读镜像**，与 IDE 记忆系统保持同步。
> **常态化更新约定**：每当沉淀出新的项目约定、验证判据或踩坑经验，除写入记忆系统外，
> 应同步更新本文件对应小节（见文末「维护约定」）。
> **路径约定**：本文件内涉及目录一律使用**相对仓库根目录**的路径（如 `src/fdemcore`），不使用绝对路径。

## 1. 项目概述

LWD-EM-Inv 是用于随钻电磁波测井（LWD-EM）数据的地球物理正反演代码库。

- 支持多种反演方法（Occam、贝叶斯、深度学习等）。
- 适用于多频率、多偏移距的随钻电磁测井数据。
- 提供可扩展的正演与反演框架（FEM、FDTD、积分方程等）。
- 应用场景：油气勘探、地下结构成像。

## 2. 运行环境与依赖

- Python 3.13（推荐 Anaconda）。
- 依赖以 `requirements.txt` 为准：
  - `numpy>=1.22`（**允许 numpy 2.x**，已解除早期 `<2.0` 限制）
  - `scipy>=1.12`
  - `matplotlib>=3.5.0`
  - `typing-extensions>=4.0.0`（仅 python<3.13）
  - `discretize>=0.12`
  - `pymatsolver>=0.3`
  - `geoana>=0.8.1`
  - `libdlf`
- 验证参考基准：pip 安装的 `simpeg 0.25.1.dev3`（源码开发安装）。**仅用于正演数值对标验证**，本项目实际交付的派生核心 `src/fdemcore` 不依赖 pip `simpeg`。

## 3. 目录与构建约定

- **SimPEG 源码隔离**：上游 SimPEG 完整检出放在 `external/SimPEG-src/`（本地存档/抽取源，**不入库**，已在 `.gitignore` 忽略）。若把同名 `simpeg/` 目录留在仓库根，会因目录遮蔽导致 `import simpeg` 报错。
- **派生核心**：`src/fdemcore/` —— 从 SimPEG 精简派生的频域电磁 FVM 正反演核心（自包含，不依赖 pip simpeg）。
- **业务层**：`src/lwd_fdem/` —— 随钻测井 FDEM 正反演封装（survey/meshes/forward/inversion/dataio/plotting）。
- **产物忽略**：`__pycache__/`、`*.pyc`、`examples/figs/`（示例运行生成的图件）均不入库。

## 4. 代码编写规范

- 命名遵循 PEP8。
- 注释采用中文 Google 风格，含完整类型标注。
- 物理量统一 SI 单位：电导率 S/m、频率 Hz、偏移距 m。
- 日志输出使用 `logger` 替代 `print`。

## 5. 测试与正演验证判据

**验证策略**：用 pip SimPEG（参考基准）与派生代码 `src/fdemcore` 构建完全相同的观测系统与模型，逐道比对 `dpred`。

**验收阈值**：

| 判据 | 阈值 |
|---|---|
| 数值等价性（派生核心 vs pip SimPEG） | 相对差 0（bit-identical） |
| 1D 正演对标准静态解析解 | 误差 ≤ 0.1%（≤1e-3） |
| 2D 轴对称 vs 3D 直角交叉验证 | 误差 ≤ 15% |
| σ→0 一致性 | 误差 ≤ 2% |
| 3D 互易性 | ≤ 5% |
| 梯度校验 | 通过 `check_derivative` |
| 反演回收 | 需定量验证 |

## 6. 正反演可视化规范（SCI 图版）

正演模块（LWD fdem）必须输出**模型 / 网格（mesh）/ 响应**三类面板，反演输出
**真实/回收剖面 + 收敛史**；绘制风格为 **SCI 论文图版**：衬线字体（Times/STIX
数学）、内向刻度含次刻度、无边框图例、去图内大标题改用 (a)(b)(c) 面板标注、
英文坐标 + 物理量 LaTeX 记法与单位、300 dpi、`bbox_inches='tight'`。多面板经
`GridSpec` 合成为**一张多联图版**。内容约定参考
`external/SimPEG-src/fdem_forward_validation`：

- **模型**：1D 层状示意（空气层+地表+按 σ 着色层块+σ/ρ/层名+层厚 h+井眼+Tx/Rx 线圈，深度向下）；2D/3D 为 log10(σ) 剖面 + colorbar。
- **网格**：2D/3D 在 log10(σ) 剖面上叠加网格线（core+padding 疏密）并标注收发线圈（图版中该面板不重复 colorbar）。
- **响应**：1D 为 log-log \|d\|-频率 频散曲线；2D/3D 为 \|d\|-收发距；曲线用 marker+linestyle 组合区分。

实现见 `src/lwd_fdem/plotting.py`：`pub_style`（样式上下文）、`plot_model`/
`plot_mesh`/`plot_response`（单面板）、`plot_forward_figure`/`plot_inversion_figure`
（合成图版，经 `fwd.plot_figure` 调用）、`plot_forward_views`（三张单图，向后兼容）。
README 案例图版存放于 `docs/images/`（入库），示例产物目录 `examples/figs/` 被忽略。

## 7. Git 提交规范

1. 提交前先删除生成的图件（`examples/figs/` 下 PNG），因其可由示例重新生成。
2. 按功能模块划分提交，每个模块独立 commit：环境配置 / 文档 / 核心库 `src/fdemcore` / 业务层 `src/lwd_fdem` / 示例 / 测试。
3. `git add` 前用 `git add -n` 干跑，确认不误入 `__pycache__`、`external/`、图件等。

## 8. 常见陷阱（Pitfalls）

### 8.1 discretize 圆柱网格绘图：用 plot_image/plot_slice，勿手喂 pcolormesh
`CylindricalMesh.nodes_x` 返回的是 r 方向**单元中心**（长度 = `shape_cells[0]`），而 `nodes_z` 返回**边**（长度 = nz+1），二者语义不一致；`pcolormesh(shading='flat')` 要求 C 比 X/Y 各小 1，遂报 `Dimensions of C ... should be one smaller than X`。`TensorMesh` 的 `nodes_x/y/z` 则都是边。
**修复**：填充图优先用 `mesh.plot_image(v, ax=..., cmap=..., grid=False)[0]`（2D）/ `mesh.plot_slice(v, normal='y', ind=ny//2, ..., grid=False)[0]`（3D）；二者返回 `(QuadMesh,)` 元组且不自动加 colorbar，需手动 `fig.colorbar(im, ax=ax)`。适用 `discretize>=0.12`（旧 `vectorNx` 已移除，改用 `nodes_x`）。

### 8.2 SimPEG 派生代码 API 参数顺序与导入路径
- `analytics` 模块可能未列入 `__init__.py`，`from ...frequency_domain import analytics` 会失败，需按实际文件位置/相对导入。
- 构造函数位置参数易错：`MagDipole(...)` 第 4 位是 `orientation` 而非 `moment`；`PointMagneticField` 无第 4 位置参。**统一用关键字参数**（`orientation="z", moment=1.0`、`data_type=..., component=...`）。

### 8.3 SimPEG v0.25 SaveOutputEveryIteration 的 save_txt 废弃
`SaveOutputEveryIteration(save_txt=False)` 抛 `NotImplementedError`；改用 `on_disk=False`。

### 8.4 3D 轴向 Ex 接收器在均匀背景中响应≈0
磁偶极子源在井轴、接收器沿轴偏移测同轴 Ex 分量时，均匀全空间中 Ex 场关于源反对称，正演≈数值噪声底（~1e-18），做相对变化会得到无意义巨值（如 -5.9e15%）。
**修复**：勿对近零背景做相对变化；直接打印背景/异常的 real/imag 绝对幅度，柱状图改四联柱（背景 real/imag、异常 real/imag）并启用对数坐标。

## 9. 维护约定

- 本文件是记忆的仓库内镜像：新增/修改项目约定、判据或踩坑经验时，**同步更新本文件与记忆系统**（常态化）。
- 涉及目录路径**优先使用相对仓库根目录**的写法，避免绝对路径与机器绑定。
- 与记忆系统冲突时，以本文件与代码/配置现状为准，并回填修正记忆。
