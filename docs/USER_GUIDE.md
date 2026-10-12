# 使用手册

本手册面向首次使用 LWD-EM-Inv 的科研人员，从安装到出图完整走一遍工作流。
建议配合 `docs/PARAMETER_REFERENCE.md`（参数速查）和 `docs/FAQ.md`（排障）使用。

---

## 目录

1. [安装与环境](#1-安装与环境)
2. [核心概念](#2-核心概念)
3. [工作流总览](#3-工作流总览)
4. [步骤 1：配置仪器](#4-步骤-1配置仪器)
5. [步骤 2：构建网格（2D/3D）](#5-步骤-2构建网格2d3d)
6. [步骤 3：正演计算](#6-步骤-3正演计算)
7. [步骤 4：合成数据与反演](#7-步骤-4合成数据与反演)
8. [步骤 5：可视化与出图](#8-步骤-5可视化与出图)
9. [进阶用法](#9-进阶用法)
10. [高级分析功能](#10-高级分析功能)
11. [示例脚本索引](#11-示例脚本索引)

---

## 1. 安装与环境

### 1.1 创建 conda 环境

```bash
conda create -n lwd-eminv python=3.13
conda activate lwd-eminv
```

### 1.2 安装依赖

```bash
cd LWD-EM-Inv
pip install -r requirements.txt
python setup.py install
```

核心依赖：`discretize`（网格与内积）、`pymatsolver`（线性求解器）、`geoana`（物理常数）、`libdlf`（数字滤波 Hankel 变换）。

### 1.3 验证安装

```bash
python -c "from src.lwd_fdem import LWDFDEMForward, LWDToolConfig; print('OK')"
pytest -m "not slow"    # 快速回归测试
```

---

## 2. 核心概念

### 2.1 分层架构

```
┌─────────────────────────────────────────────────┐
│  使用者代码（你的脚本）                            │
├─────────────────────────────────────────────────┤
│  lwd_fdem 业务层                                 │
│  ┌──────────┬──────────┬──────────┬───────────┐ │
│  │ survey   │ meshes   │ forward  │ inversion │ │
│  │ 仪器配置  │ 网格生成  │ 正演入口  │ 反演管线   │ │
│  ├──────────┼──────────┼──────────┼───────────┤ │
│  │ tensor   │ advanced │ apparent │ doi       │ │
│  │ 张量观测  │ 灵敏度/并行│ 视电阻率 │ 探测深度   │ │
│  ├──────────┼──────────┴──────────┴───────────┤ │
│  │ window   │ dataio   │ plotting             │ │
│  │ 滑动窗口 │ 数据读写  │ 可视化                 │ │
│  └──────────┴──────────┴──────────────────────┘ │
├─────────────────────────────────────────────────┤
│  fdemcore 计算核心（SimPEG 精简派生，独立运行）     │
│  FDEM simulation / maps / regularization / ...   │
└─────────────────────────────────────────────────┘
```

### 2.2 三种几何

| 几何 | 求解器 | 速度 | 适用场景 |
|------|--------|------|---------|
| **1D 层状** | Hankel 解析解 | 最快（毫秒级） | 垂直井/水平井层状地层近似 |
| **1D 张量** | 1D 层状参数化 + 3D FVM | 中等 | 层状模型但需 VTI 各向异性或倾斜井时 |
| **2D 轴对称** | CylindricalMesh FVM | 中等（秒级） | 含井眼/侵入带的轴对称模型 |
| **2D 张量** | 2D TensorMesh FVM | 中等 | 非轴对称 2D 模型、倾斜仪器 |
| **3D 直角** | TensorMesh FVM | 最慢（分钟级） | 倾斜井/非对称结构/块体异常 |

### 2.3 数据排列顺序

所有数据向量按 **频率 → 收发距 → 分量** 排列：

```
d = [f1_s1_real, f1_s1_imag, f1_s2_real, f1_s2_imag, ..., f2_s1_real, ...]
```

用 `fwd.to_matrix(d)` 可整形为 `(n_freq, n_spacing, n_comp)` 三维数组。

---

## 3. 工作流总览

```
配置仪器 (LWDToolConfig)
    │
    ├─ 1D: 直接传 layer_thicknesses
    ├─ 2D: build_2d_cyl_mesh → CylindricalMesh
    └─ 3D: build_3d_tensor_mesh → TensorMesh
    │
    ▼
构建正演 (LWDFDEMForward)
    │
    ├─ dpred(sigma) → 正演数据
    ├─ jvec(sigma, v) → 灵敏度矩阵乘积
    └─ jtvec(sigma, v) → 伴随梯度
    │
    ▼
合成数据 / 加载实测 (add_synthetic_noise / load_lwd_csv)
    │
    ▼
构建反演 (LWDFDEMInversion)
    │
    └─ run() → (recovered_sigma, history)
    │
    ▼
可视化 (plot_forward_figure / plot_inversion_figure)
```

---

## 4. 步骤 1：配置仪器

`LWDToolConfig` 描述一台 LWD 仪器的全部参数。

### 4.1 水平井典型配置（1D 层状，径向偏移）

```python
import numpy as np
from src.lwd_fdem import LWDToolConfig

tool = LWDToolConfig(
    frequencies=(1e4, 1e5, 5e5, 1e6),    # 4 个频率：10 kHz ~ 1 MHz
    spacings=np.array([0.3, 0.6, 1.0]),   # 3 个收发距：0.3 / 0.6 / 1.0 m
    components=("real", "imag"),           # 记录实部 + 虚部
    field_kind="h",                        # 磁场接收器
    secondary=True,                        # 二次场（去掉一次场背景）
    receiver_geometry="radial",            # 径向偏移（1D 必须）
    orientation="z",                       # 轴向偶极子
    source_type="mag_dipole",              # 磁偶极子源
    moment=1.0,                            # 偶极矩 1 A·m²
)
```

### 4.2 垂直井典型配置（2D 轴对称，轴向偏移）

```python
tool_2d = LWDToolConfig(
    frequencies=(2e4, 2e5),
    spacings=np.array([0.5, 1.0, 1.5, 2.0]),
    components=("real", "imag"),
    field_kind="h",
    secondary=False,                       # 总场
    receiver_geometry="axial",             # 轴向偏移（2D/3D）
    orientation="z",
)
```

### 4.3 参数选择指南

**频率选择：**
- 高频 → 探测浅但灵敏度高；低频 → 探测深但灵敏度低
- 趋肤深度 δ ≈ 503/√(f·σ) m（准静态近似），如 f=10 kHz, σ=0.01 S/m → δ ≈ 159 m
- 建议：频率范围覆盖目标深度的趋肤深度

**收发距选择：**
- 短收发距 → 对井眼/侵入带灵敏
- 长收发距 → 对原状地层灵敏
- 建议：至少 2 个收发距，长短搭配

**二次场 vs 总场：**
- `secondary=True`：去掉一次场背景，对异常体更灵敏，适合金属矿
- `secondary=False`：总场，适合地层电阻率测量（LWD 常用）

---

## 5. 步骤 2：构建网格（2D/3D）

> 1D 层状不需要网格，直接传 `layer_thicknesses`。

### 5.1 趋肤深度检查

```python
from src.lwd_fdem import skin_depth, recommend_cell_size

# 计算单个趋肤深度
delta = skin_depth(frequency=1e5, conductivity=0.1)
print(f"趋肤深度: {delta:.1f} m")

# 自动推荐最大单元尺寸（每个趋肤深度至少 3 个单元）
h_max = recommend_cell_size(
    frequencies=(1e4, 1e5, 1e6),
    conductivities=(0.01, 0.1, 1.0),
)
print(f"推荐最大单元尺寸: {h_max:.2f} m")
```

### 5.2 2D 轴对称网格

```python
from src.lwd_fdem import build_2d_cyl_mesh, mesh_quality_report

mesh = build_2d_cyl_mesh(
    r_core=2.0,                # 径向核心区域 2 m（覆盖井眼+侵入带）
    z_core=4.0,                # 轴向核心区域 4 m（覆盖工具+附近地层）
    frequencies=(1e4, 1e5),    # 用于自动计算单元尺寸
    conductivities=(0.01, 1.0),
    n_pad=8,                   # 每方向 8 层 padding
    pad_factor=1.3,            # padding 放大 30%
)

# 检查网格质量
report = mesh_quality_report(mesh, frequencies=(1e4, 1e5), conductivities=(0.01, 1.0))
print(f"最小趋肤深度: {report['min_skin_depth']:.1f} m")
print(f"每趋肤深度单元数: {report['cells_per_skin_depth']:.1f}")
print(f"网格合格: {'是' if report['ok'] else '否'}")
```

### 5.3 3D 直角网格

```python
from src.lwd_fdem import build_3d_tensor_mesh

mesh_3d = build_3d_tensor_mesh(
    core_extent=(4.0, 4.0, 6.0),    # 核心区域 (dx, dy, dz)
    frequencies=(1e4, 1e5),
    conductivities=(0.01, 0.1),
    n_pad=6,
    pad_factor=1.3,
)
```

---

## 6. 步骤 3：正演计算

### 6.1 1D 层状正演

```python
from src.lwd_fdem import LWDFDEMForward

# 三层模型：0.25 / 0.01 / 0.5 S/m，厚度 2 m + 4 m + 半空间
fwd = LWDFDEMForward(
    geometry="1d",
    tool=tool,
    layer_thicknesses=np.array([2.0, 4.0]),
)

sigma_model = np.array([0.25, 0.01, 0.5])    # 三层电导率 (S/m)
d = fwd.dpred(sigma_model)                    # 正演数据

# 查看数据形状
mat = fwd.to_matrix(d)
print(f"数据形状: {mat.shape}")    # (4, 3, 2) = (4 频率, 3 收发距, 2 分量)
```

### 6.2 2D 轴对称正演

```python
fwd_2d = LWDFDEMForward(
    geometry="2d_cyl",
    tool=tool_2d,
    mesh=mesh,
)

# 2D 模型向量长度 = 网格单元数
sigma_2d = 0.1 * np.ones(mesh.n_cells)
d_2d = fwd_2d.dpred(sigma_2d)
```

### 6.3 Cole-Cole 频散（仅 1D）

```python
fwd_cc = LWDFDEMForward(
    geometry="1d",
    tool=tool,
    layer_thicknesses=np.array([5.0, 10.0]),
    cole_cole={
        "eta": np.array([0.0, 0.3, 0.0]),    # 中间层有频散
        "tau": 0.1,                           # 弛豫时间 0.1 s
        "c": 0.7,                             # 频率指数
    },
)
```

### 6.4 灵敏度矩阵乘积

```python
# J·v：模型扰动方向 v 引起的数据变化
v = np.random.randn(fwd.n_params)
Jv = fwd.jvec(sigma_model, v)

# Jᵀ·v：数据残差映射回模型空间（梯度方向）
residual = np.random.randn(d.size)
JTv = fwd.jtvec(sigma_model, residual)
```

---

## 7. 步骤 4：合成数据与反演

### 7.1 合成数据

```python
from src.lwd_fdem import add_synthetic_noise, make_data_object

# 加 5% 高斯噪声
dobs, std = add_synthetic_noise(d, noise_level=0.05, seed=42)

# 构建反演数据对象
data_obj = make_data_object(fwd.survey, dobs, absolute_std=std)
```

### 7.2 1D Occam 反演

```python
from src.lwd_fdem import LWDFDEMInversion

inv = LWDFDEMInversion(
    forward=fwd,
    data_obj=data_obj,
    m0=np.log(0.1 * np.ones(3)),    # 初始模型：均匀 0.1 S/m
    max_iter=30,                     # 最多 30 次迭代
    beta_cooling=5.0,                # β 冷却因子
    chifact=1.0,                     # 目标 φ_d = nD
)

sigma_recovered, history = inv.run()

print(f"真实模型:  {sigma_model}")
print(f"反演结果:  {sigma_recovered}")
print(f"最终 φ_d:  {history['final_phi_d']:.2f}")
```

### 7.3 稀疏反演（块状模型）

```python
inv_sparse = LWDFDEMInversion(
    forward=fwd,
    data_obj=data_obj,
    m0=np.log(0.1 * np.ones(3)),
    sparse=True,                     # 开启 IRLS 稀疏正则化
    norms=[0, 2],                    # L0 模型 + L2 梯度
    alpha_s=1e-4,
    max_iter=40,
)

sigma_sparse, hist_sparse = inv_sparse.run()
```

### 7.4 2D 反演

```python
inv_2d = LWDFDEMInversion(
    forward=fwd_2d,
    data_obj=data_obj_2d,
    m0=np.log(0.1 * np.ones(mesh.n_cells)),
    alpha_x=1.0,                     # r 方向平滑
    alpha_z=1.0,                     # z 方向平滑
    alpha_s=1e-3,
    max_iter=30,
)

sigma_2d_rec, hist_2d = inv_2d.run()
```

### 7.5 反演参数调优建议

| 问题 | 调整 |
|------|------|
| 反演不收敛 | 增大 `max_iter`；降低 `beta_cooling`（如 2.0） |
| 过度光滑 | 减小 `alpha_s`（如 1e-4）；开启 `sparse=True` |
| 数据拟合不够 | 增大 `chifact`（如 1.5）；检查噪声水平 |
| 模型振荡 | 增大 `alpha_x`/`alpha_z`（如 2.0）；减小 `alpha_s` |

---

## 8. 步骤 5：可视化与出图

### 8.1 正演图版

```python
from src.lwd_fdem import plot_forward_figure, pub_style

pub_style()    # 设置 SCI 图版样式

plot_forward_figure(
    fwd, sigma_model, d,
    out_dir="figures",
    prefix="forward_1d",
)
# 输出: figures/forward_1d_plate.png
```

### 8.2 反演图版

```python
from src.lwd_fdem import plot_inversion_figure

plot_inversion_figure(
    fwd, sigma_recovered, history,
    sigma_true=sigma_model,           # 真实模型（可选，用于对比）
    out_dir="figures",
    prefix="inversion_1d",
)
```

### 8.3 单独绘图

```python
from src.lwd_fdem import plot_model, plot_mesh, plot_response

# 模型图
plot_model(fwd, sigma_model, labels=["覆盖层", "低阻层", "基岩"])

# 网格图（2D/3D）
plot_mesh(mesh)

# 响应曲线
plot_response(fwd, d)
```

### 8.4 中文标注

```python
from src.lwd_fdem import set_cjk_rcparams

set_cjk_rcparams()    # 启用中文字体（SimHei / Microsoft YaHei）
```

---

## 9. 进阶用法

### 9.1 解析解验证

```python
from src.fdemcore.electromagnetics.analytics import hzAnalyticDipoleF

# 半空间解析解
sigma_half = 0.01
freqs = np.logspace(1, 5, 25)
r = 100.0    # 偏移距

# 1D 数值解（用很厚的层近似半空间）
fwd_half = LWDFDEMForward("1d", tool, layer_thicknesses=np.array([1e5]))
hz_num = fwd_half.dpred(np.array([sigma_half, sigma_half]))

# 解析解
hz_ana = hzAnalyticDipoleF(freqs, r, sigma_half).flatten()

# 对比
rel_err = np.max(np.abs(hz_num - hz_ana) / np.abs(hz_ana))
print(f"最大相对误差: {rel_err:.2e}")
```

### 9.2 自定义电导率映射

```python
from src.fdemcore import maps

# 使用 ExpMap：模型向量 = ln(σ)
exp_map = maps.ExpMap(nP=3)
fwd_exp = LWDFDEMForward("1d", tool,
                          layer_thicknesses=np.array([2.0, 4.0]),
                          sigma_map=exp_map)

# 此时 dpred 的输入为 log(sigma)
log_sigma = np.log([0.25, 0.01, 0.5])
d = fwd_exp.dpred(log_sigma)
```

### 9.3 网格质量检查

```python
from src.lwd_fdem import mesh_quality_report

report = mesh_quality_report(mesh, frequencies, conductivities)
if not report['ok']:
    print(f"警告: 每趋肤深度仅 {report['cells_per_skin_depth']:.1f} 个单元")
    print(f"建议: 减小 cell_size 或增大 frequencies/conductivities 范围")
```

---

## 10. 高级分析功能

### 10.1 张量观测系统

三轴源 × 三轴接收器构建 9 分量张量 Survey：

```python
from src.lwd_fdem import build_tensor_survey, build_tensor_receivers, tensor_data_order

survey = build_tensor_survey(
    frequencies=(1e4, 1e5),
    spacings=np.array([10.0, 14.0]),
    field_kind="h",
    secondary=False,
    components=("real", "imag"),
    receiver_geometry="axial",
    rx_orientations=("x", "y", "z"),
    src_orientations=("x", "y", "z"),
)
# 数据排列：频率 → 源取向(3) → 收发距 → 接收取向(3) → 分量(2)
```

### 10.2 各向异性正演（VTI）

```python
fwd_ani = LWDFDEMForward(
    "3d", tool, mesh=mesh_3d,
    anisotropic=True,    # 启用各向异性路径
)

# 模型向量: [σh_all_cells, σv_all_cells] (flat 1D)
# 内部通过 _AnisotropicSigmaMap 转为 (n_cells, 3) = [σh, σh, σv]
sigma_ani = np.concatenate([0.1 * np.ones(mesh_3d.n_cells),
                            0.05 * np.ones(mesh_3d.n_cells)])
d_ani = fwd_ani.dpred(sigma_ani)
```

1d_tensor 路径支持层状参数化 + 各向异性：

```python
fwd_1d_ani = LWDFDEMForward(
    "1d_tensor", tool,
    layer_thicknesses=np.array([5.0, 10.0]),
    anisotropic=True,
)
# 模型: [σh_layer0, σh_layer1, σh_layer2, σv_layer0, σv_layer1, σv_layer2]
d = fwd_1d_ani.dpred(np.array([0.1, 0.5, 0.01, 0.05, 0.25, 0.005]))
```

### 10.3 视电阻率转换

```python
from src.lwd_fdem import apparent_resistivity_from_hz, apparent_resistivity_from_forward

# 基于 Hz 解析解的半空间视电阻率
rho_a = apparent_resistivity_from_hz(
    frequency=1e4, hz_measured=1e-5 + 0.5j * 1e-5,
    offset=1.0, method="amplitude",
)

# 基于正演的视电阻率（适用于任意几何）
rho_a_matrix = apparent_resistivity_from_forward(fwd, sigma_model, d)
```

### 10.4 探测深度（DOI）分析

```python
from src.lwd_fdem import doi_1d_sensitivity, doi_skin_depth_estimate, doi_summary

# 基于灵敏度的 1D DOI
doi_result = doi_1d_sensitivity(fwd, sigma_model, threshold_ratio=0.1)
print(f"探测深度: {doi_result['doi_depth']:.1f} m")
print(f"可靠层: {doi_result['reliable_mask']}")

# 趋肤深度估计
delta = doi_skin_depth_estimate(fwd.tool.frequencies, sigma_model)

# 汇总报告
summary = doi_summary(fwd, sigma_model)
```

### 10.5 滑动窗口反演

沿井深逐窗口 1D 反演，生成连续电阻率剖面：

```python
from src.lwd_fdem import sliding_window_inversion_1d, simple_window_builder, simple_inversion_runner

result = sliding_window_inversion_1d(
    depths=depth_array,
    data_at_depths=data_cube,
    forward_builder=lambda: fwd_template,
    inversion_runner=simple_inversion_runner,
    window_size=5,
    step=1,
)
# result['depths']       —— 深度数组
# result['sigma_mid']    —— 各深度反演的中层电导率
# result['sigma_models'] —— 各深度完整模型
```

### 10.6 灵敏度矩阵

```python
from src.lwd_fdem import compute_sensitivity_matrix, compute_sensitivity_diagonal

# 完整 J 矩阵 (nD × nP)
J = compute_sensitivity_matrix(fwd, sigma_model)

# JᵀJ 对角元（更省内存）
diag = compute_sensitivity_diagonal(fwd, sigma_model)
```

### 10.7 并行频率计算

```python
from src.lwd_fdem import forward_parallel_frequencies

# 各频率独立正演，用 concurrent.futures 并行
results = forward_parallel_frequencies(tool, sigma_model, geometry="1d",
                                        layer_thicknesses=np.array([2.0, 4.0]))
```

### 10.8 LAS 2.0 导出

```python
from src.lwd_fdem import export_las2

export_las2(
    filename="result.las",
    depths=depth_array,
    data={"RHOO": sigma_recovered},
    well_name="Well-01",
    company="LWD-EM-Inv",
)
```

---

## 11. 示例脚本索引

| 脚本 | 几何 | 演示内容 |
|------|------|---------|
| `lwd_fdem_forward_1d.py` | 1D | 基础正演流程 |
| `lwd_fdem_inversion_1d.py` | 1D | Occam 反演 |
| `lwd_fdem_forward_2d.py` | 2D | 轴对称正演 |
| `lwd_fdem_inversion_2d.py` | 2D | 2D 反演 |
| `lwd_fdem_forward_3d.py` | 3D | 3D 正演 |
| `lwd_fdem_inversion_3d_smoke.py` | 3D | 3D 反演（快速烟雾测试） |
| `horizontal_well_3layer.py` | 1D | 水平井完整流程 |
| `vertical_well_3layer.py` | 2D | 垂直井完整流程 |
| `horizontal_well_advanced.py` | 1D | 二次场/总场、Occam/稀疏对比 |
| `vertical_well_advanced.py` | 2D | 偶极子/回线源、Occam/稀疏对比 |
| `cole_cole_dispersion.py` | 1D | Cole-Cole 频散效应 |
| `benchmark_analytic.py` | 1D | 数值解 vs 解析解验证 |
| `test_apparent_resistivity.py` | 1D | 视电阻率转换 |
| `test_doi_analysis.py` | 1D | 探测深度（DOI）分析 |
| `test_sliding_window.py` | 1D | 滑动窗口反演 |
| `test_tensor_survey.py` | 3D | 张量多分量观测系统 |
| `test_advanced.py` | 1D/2D | 高级功能综合演示 |

运行示例：

```bash
python examples/lwd_fdem_forward_1d.py
python examples/lwd_fdem_inversion_1d.py
```
