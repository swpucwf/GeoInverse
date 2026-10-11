# 参数速查表

本文档列出 `lwd_fdem` 业务层所有公开类与函数的参数，按 **物理含义 → 取值范围 → 典型值 → 注意事项** 四列说明，方便查阅。

> 所有物理量使用 SI 单位制，时间因子约定为 exp(+iωt)。

---

## 1. LWDToolConfig — 仪器与观测系统配置

```python
from src.lwd_fdem import LWDToolConfig
```

### 1.1 发射源参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `source_type` | 发射源类型 | `'mag_dipole'` / `'circular_loop'` | `'mag_dipole'` | 偶极子适用于收发距 ≫ 线圈尺寸；回线源在收发距与线圈尺寸可比时更精确 |
| `moment` | 磁偶极矩 (A·m²)，即 I·A·N | 正浮点数 | 1.0 | 正演结果与 moment 线性成正比；实际仪器 1–100 A·m² |
| `loop_radius` | 圆环线源半径 (m) | 正浮点数 | 0.05 | 仅 `source_type='circular_loop'` 时生效 |
| `source_location` | 发射器全局坐标 (x, y, z) (m) | 三元组 | `(0, 0, 0)` | 单源时一般取原点 |
| `orientation` | 偶极子/线圈轴向 | `'x'` / `'y'` / `'z'` | `'z'` | 垂直井用 `'z'`（轴向）；水平井可能用 `'x'` 或 `'y'` |

### 1.2 接收器布局参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `spacings` | 收发距数组 (m)，接收器相对发射器的偏移距离 | 正 1D 数组 | `np.array([0.3, 0.6, 1.0])` | 全部取正值；偏移方向由 `receiver_geometry` 决定 |
| `receiver_geometry` | 接收器偏移方向 | `'axial'` / `'radial'` | 见下 | **1D 层状必须用 `'radial'`**；2D/3D 轴向仪器用 `'axial'` |

**典型配置：**
- 垂直井径向仪器 → `receiver_geometry='radial'`
- 2D/3D 轴向仪器 → `receiver_geometry='axial'`

### 1.3 观测频率与分量参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `frequencies` | 工作频率列表 (Hz) | 正数元组 | `(2e3, 2e4, 2e5)` | 多频探测不同深度（趋肤深度 δ ∝ 1/√f）；典型 LWD 用 2–8 个频率 |
| `field_kind` | 记录的场量类型 | `'h'` / `'b'` / `'e'` / `'j'` | `'h'` | 线圈接收器测 H/B；电极接收器测 E。1D 中 H 与 B 等价 |
| `secondary` | 是否仅记录二次场 | `True` / `False` | `True` | True = 散射场（去一次场）；False = 总场。电场接收器仅支持 False |
| `components` | 记录的数据分量 | `('real','imag')` / `('amp','phase')` 等 | `('real', 'imag')` | 反演时须与实际观测数据一致 |
| `data_type` | 数据归一化方式 | `'field'` / `'ppm'` | `'field'` | 正演输出始终为绝对场值；PPM 仅在反演预处理中生效 |

**场量对照表：**

| field_kind | 物理量 | 单位 | 接收器类型 |
|-----------|--------|------|-----------|
| `'h'` | 磁场 H | A/m | PointMagneticField |
| `'b'` | 磁通密度 B | T | PointMagneticFluxDensity |
| `'e'` | 电场 E | V/m | PointElectricField |
| `'j'` | 电流密度 J | A/m² | PointCurrentDensity |

---

## 2. LWDFDEMForward — 正演求解器

```python
from src.lwd_fdem import LWDFDEMForward
```

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `geometry` | 几何维度，决定底层求解器 | `'1d'` / `'2d_cyl'` / `'3d'` | — | 1D → Hankel 解析；2D → CylindricalMesh FVM；3D → TensorMesh FVM |
| `tool` | 仪器配置 | `LWDToolConfig` 实例 | — | 所有几何均必需 |
| `layer_thicknesses` | 1D 层厚度 (m)，自上而下，末层为半空间 | 正 1D 数组 | `np.array([5.0, 10.0])` | **仅 1D 必需**；如 `array([5, 10])` 表示 3 层（第 3 层半空间） |
| `mesh` | FVM 计算网格 | `CylindricalMesh` / `TensorMesh` | — | **仅 2D/3D 必需**；用 `build_2d_cyl_mesh` / `build_3d_tensor_mesh` 生成 |
| `form` | FVM 离散形式（场量未知量类型） | `'e'` / `'b'` / `'h'` / `'j'` / `None` | `None` | None 时按 `tool.field_kind` 自动匹配；一般无需手动指定 |
| `sigma_map` | 电导率映射（模型→网格电导率） | `IdentityMap` / `ExpMap` 等 | `None`（恒等映射） | 反演中常用 `ExpMap`（模型 = ln(σ)）保证正性 |
| `solver` | 线性方程组求解器 | 求解器类 | `None`（默认直接法） | 大规模 3D 可换迭代求解器节省内存 |
| `cole_cole` | Cole-Cole 频散参数 | 字典或 `None` | `None` | **仅 1D 生效**。格式见下表 |

### 2.1 cole_cole 字典参数

| 键 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|----|---------|---------|--------|---------|
| `eta` | 充电子率 | 0 ≤ η < 1 | 0.0（无色散）/ 0.3 | 标量→各层相同；数组→长度须等于层数 |
| `tau` | 弛豫时间 (s) | τ > 0 | 0.01–1.0 | 控制频散转折频率 |
| `c` | 频率指数 | 0 < c < 1 | 0.5–0.8 | 控制频散过渡带宽度 |

**Cole-Cole 公式**（Pelton, 1978）：

```
σ(ω) = σ∞ · [1 − η · (1 − 1/(1 + (iωτ)^c))]
```

### 2.2 主要方法

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `dpred(sigma)` | 电导率模型 (S/m) | 预测数据 (nD,) | 正演模拟 |
| `jvec(sigma, v)` | 模型 + 扰动方向 | J·v (nD,) | Gauss-Newton 步长 |
| `jtvec(sigma, v)` | 模型 + 数据向量 | Jᵀ·v (nP,) | 梯度计算 |
| `to_matrix(d)` | 数据向量 (nD,) | (n_freq, n_spacing, n_comp) | 数据整形 |
| `build_simulation(sigma_map)` | 可选映射 | Simulation 对象 | 重建仿真 |

---

## 3. LWDFDEMInversion — 反演管线

```python
from src.lwd_fdem import LWDFDEMInversion
```

### 3.1 基本参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `forward` | 正演求解器 | `LWDFDEMForward` 实例 | — | 必需 |
| `data_obj` | 观测数据（含 dobs 和标准差） | `Data` 实例 | — | 用 `make_data_object` 构建 |
| `m0` | 初始模型 (log-σ) | 1D 数组 (nP,) | `None`→log(0.1) | 影响收敛路径，建议根据先验设置 |
| `mref` | 参考模型 (log-σ) | 1D 数组 (nP,) | `None`→m0 | 正则化度量偏离的基准 |

### 3.2 正则化参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `sparse` | 是否使用稀疏正则化 (IRLS) | `True` / `False` | `False` | True→块状模型（边界尖锐）；False→平滑模型 |
| `alpha_s` | 小扰动权重（模型与参考偏离惩罚） | ≥ 0 | 1e-3 | 越小越允许偏离参考，数据拟合优先 |
| `alpha_x` | x 方向一阶平滑权重 | ≥ 0 | 1.0 | 1D 仅用 alpha_z；2D 用 x(r)+z；3D 全用 |
| `alpha_y` | y 方向一阶平滑权重 | ≥ 0 | `None`→1.0 | 仅 3D 生效 |
| `alpha_z` | z 方向一阶平滑权重 | ≥ 0 | `None`→1.0 | 1D/2D/3D 均用 |
| `norms` | 稀疏正则化范数阶数 [p, q] | 列表 | `[0, 2]` | p=模型范数，q=梯度范数；仅 `sparse=True` 时生效 |

**正则化组合建议：**

| 场景 | sparse | alpha_s | alpha_x/z | norms |
|------|--------|---------|-----------|-------|
| 平滑地层（Occam） | False | 1e-3 | 1.0 | — |
| 块状地层（边界尖锐） | True | 1e-4 | 1.0 | [0, 2] |
| 全 TV（全变分） | True | 1e-4 | 1.0 | [1, 1] |

### 3.3 优化与停止参数

| 参数 | 物理含义 | 取值范围 | 典型值 | 注意事项 |
|------|---------|---------|--------|---------|
| `optimizer` | 优化器类型 | `'gn'` / `'pgncg'` | `'gn'` | 中小规模用 GN；大规模/带约束用 PGCNCG |
| `beta0_ratio` | 初始 β 比例（数据/正则化权衡） | 正浮点数 | 1.0 | 调大→初始更重数据拟合 |
| `beta_cooling` | β 冷却因子 | > 1 | 5.0 | 每轮 β 乘以此因子，逐步降低正则化 |
| `chifact` | 目标拟合差系数 | 正浮点数 | 1.0 | 目标 φ_d = chifact × nD；标准差准确时取 1.0 |
| `max_iter` | 最大迭代次数 | 正整数 | 40 | 防止不收敛时无限循环 |
| `sensitivity_weights` | 是否启用灵敏度加权 | `True` / `False` | `False` | 深部地层反演时开启可改善分辨率 |
| `preconditioner` | 是否启用 Jacobi 预条件 | `True` / `False` | `False` | 仅 `optimizer='pgncg'` 时生效 |

---

## 4. 网格工具函数

```python
from src.lwd_fdem import (skin_depth, recommend_cell_size,
                           build_1d_mesh, build_2d_cyl_mesh,
                           build_3d_tensor_mesh, mesh_quality_report)
```

### 4.1 趋肤深度与分辨率

| 函数 | 参数 | 返回 | 用途 |
|------|------|------|------|
| `skin_depth(f, σ, εr)` | f(Hz), σ(S/m), εr(—) | δ(m) | 计算趋肤深度 |
| `recommend_cell_size(freqs, sigmas, εr, divisor)` | 频率列表, 电导率列表, 介电常数, 离散份数 | 最大单元尺寸(m) | 自动推荐网格分辨率 |

**趋肤深度公式**（全频域 Maxwell，含位移电流）：

```
δ = 1 / |Im(k)|,  k = √(ω²με − iωμσ)
```

**网格分辨率判据：** 单元尺寸 ≤ δ_min / divisor（默认 divisor=3，即每个趋肤深度至少 3 个单元）

### 4.2 网格构建

| 函数 | 关键参数 | 返回 | 适用几何 |
|------|---------|------|---------|
| `build_1d_mesh(thicknesses)` | 层厚度 (m) | TensorMesh (1D) | 1D 层状（正则化用） |
| `build_2d_cyl_mesh(r_core, z_core, ...)` | 径向/轴向核心尺寸 (m) | CylindricalMesh | 2D 轴对称 |
| `build_3d_tensor_mesh(core_extent, ...)` | 核心区域 (dx,dy,dz) (m) | TensorMesh | 3D 直角 |

**build_2d_cyl_mesh / build_3d_tensor_mesh 共有参数：**

| 参数 | 物理含义 | 典型值 | 注意事项 |
|------|---------|--------|---------|
| `cell_size` | 核心单元尺寸 (m) | 自动 | None 时按 frequencies/conductivities 自动计算 |
| `frequencies` | 工作频率 (Hz) | — | cell_size=None 时必需 |
| `conductivities` | 电导率范围 (S/m) | — | cell_size=None 时必需 |
| `n_pad` | padding 层数 | 8 (2D) / 6 (3D) | 吸收边界反射 |
| `pad_factor` | padding 放大系数 | 1.3 | 每层放大 30% |

---

## 5. 数据 I/O 工具

```python
from src.lwd_fdem import add_synthetic_noise, make_data_object, load_lwd_csv
```

| 函数 | 参数 | 返回 | 用途 |
|------|------|------|------|
| `add_synthetic_noise(d, noise_level, seed)` | 清洁数据, 噪声水平(如0.05=5%), 随机种子 | (dobs, std) | 合成数据加噪 |
| `make_data_object(survey, dobs, relative_error, noise_floor, absolute_std)` | 观测系统, 数据, 相对误差, 噪声底, 逐点标准差 | `Data` 对象 | 构建反演输入 |
| `load_lwd_csv(path)` | CSV 文件路径 | 字典 | 读取实测数据 |

**make_data_object 不确定度计算：**

```
std = relative_error × |dobs| + noise_floor
```

若提供 `absolute_std` 则直接使用，忽略 `relative_error` 和 `noise_floor`。

**load_lwd_csv CSV 格式要求：**

```csv
frequency,spacing,component,value
20000,0.3,real,1.234e-6
20000,0.3,imag,-5.678e-7
```

---

## 6. 可视化函数

```python
from src.lwd_fdem import (plot_model, plot_mesh, plot_response,
                           plot_forward_figure, plot_inversion_figure,
                           pub_style, set_cjk_rcparams, save_figure)
```

| 函数 | 用途 | 关键参数 |
|------|------|---------|
| `plot_model(fwd, sigma, ...)` | 绘制电导率模型 | fwd, sigma, labels |
| `plot_mesh(mesh, ...)` | 绘制计算网格 | mesh |
| `plot_response(fwd, d, ...)` | 绘制响应曲线 | fwd, d (数据向量) |
| `plot_forward_figure(fwd, sigma, d, ...)` | 正演多联图版（模型+网格+响应） | out_dir, prefix |
| `plot_inversion_figure(fwd, rec_sigma, history, ...)` | 反演结果图版 | sigma_true, out_dir, prefix |
| `pub_style()` | 设置 SCI 图版全局样式 | — |
| `set_cjk_rcparams()` | 设置中文字体 | — |
| `save_figure(fig, path, dpi)` | 保存图片 | fig, path, dpi=300 |
