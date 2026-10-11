# 常见问题与排障 (FAQ)

本文档汇总使用 LWD-EM-Inv 过程中的常见问题、错误信息及解决方案。
遇到问题时请先在此查找，若未找到请提 Issue。

---

## 目录

1. [安装与环境](#1-安装与环境)
2. [正演计算](#2-正演计算)
3. [反演计算](#3-反演计算)
4. [可视化](#4-可视化)
5. [性能与内存](#5-性能与内存)
6. [物理与数值](#6-物理与数值)

---

## 1. 安装与环境

### Q1.1: `ImportError: No module named 'discretize'`

**原因：** 未安装依赖包。

**解决：**
```bash
pip install -r requirements.txt
```

### Q1.2: `ModuleNotFoundError: No module named 'src.lwd_fdem'`

**原因：** 未安装包或工作目录不对。

**解决：**
```bash
cd LWD-EM-Inv
python setup.py install
# 或
pip install -e .
```

### Q1.3: Windows 下 `pytest` 报 GBK 编码错误

**原因：** `pytest.ini` 含中文注释，Windows 默认 GBK 编码与 UTF-8 冲突。

**解决：** 使用 conda 环境的 Python 运行 pytest：
```bash
conda activate lwd-eminv
python -m pytest tests/ -v
```

### Q1.4: `libdlf` 找不到或 Hankel 变换失败

**原因：** `libdlf` 是 1D 层状模型的数字滤波系数库，未正确安装。

**解决：**
```bash
pip install libdlf
```

验证：
```python
from libdlf import han84
print("libdlf OK")
```

---

## 2. 正演计算

### Q2.1: `ValueError: 1D 几何要求 receiver_geometry='radial'`

**原因：** 1D 层状模型使用 Hankel 变换，要求接收器径向偏移。

**解决：** 配置仪器时设置 `receiver_geometry="radial"`：
```python
tool = LWDToolConfig(
    ...
    receiver_geometry="radial",    # 1D 必须
)
```

### Q2.2: `ValueError: 电场接收器仅支持总场（secondary=False）`

**原因：** 电场接收器（`field_kind='e'`）不支持二次场模式。

**解决：** 使用电场接收器时设置 `secondary=False`：
```python
tool = LWDToolConfig(
    field_kind="e",
    secondary=False,    # 电场只能测总场
)
```

### Q2.3: 1D 正演结果与解析解偏差大

**可能原因：**
1. 层数不够（用有限厚度层近似半空间）
2. 频率超出 Hankel 滤波系数适用范围

**排查：**
```python
# 半空间应用很厚的层近似
thicknesses = np.array([1e5])    # 100 km 厚
fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thicknesses)

# 与解析解对比
from src.fdemcore.electromagnetics.analytics import hzAnalyticDipoleF
hz_num = fwd.dpred(sigma).flatten()
hz_ana = hzAnalyticDipoleF(freqs, r, sigma).flatten()
print(f"最大相对误差: {np.max(np.abs(hz_num - hz_ana) / np.abs(hz_ana)):.2e}")
```

若误差 > 1e-5，检查频率范围和 `libdlf` 版本。

### Q2.4: 2D/3D 正演报错 `Matrix is singular`

**原因：** 网格质量差（单元过大或过小）或电导率为零/负。

**排查：**
```python
# 检查网格质量
from src.lwd_fdem import mesh_quality_report
report = mesh_quality_report(mesh, frequencies, conductivities)
print(report)

# 检查电导率模型
assert np.all(sigma > 0), "电导率必须全为正"
```

**解决：**
- 增大 `n_pad`（如 10）或减小 `cell_size`
- 确保模型电导率 > 0（建议 ≥ 1e-6 S/m）

### Q2.5: `ValueError: cole_cole 字典缺少键: ...`

**原因：** Cole-Cole 频散参数不完整。

**解决：** 必须提供 `eta`、`tau`、`c` 三个键：
```python
cole_cole = {
    "eta": 0.3,     # 充电子率
    "tau": 0.1,     # 弛豫时间 (s)
    "c": 0.7,       # 频率指数
}
fwd = LWDFDEMForward("1d", tool, layer_thicknesses=thick, cole_cole=cole_cole)
```

### Q2.6: Cole-Cole 参数报错 `eta 必须在 [0, 1) 范围`

**原因：** 充电子率 η 物理范围为 0 ≤ η < 1。

**解决：** 检查参数：
- `eta`: 0 ≤ η < 1（典型 0.0–0.5）
- `tau`: > 0（典型 0.001–10 s）
- `c`: 0 < c < 1（典型 0.5–0.9）

---

## 3. 反演计算

### Q3.1: `ValueError: 存在非正的数据标准差`

**原因：** `make_data_object` 计算的标准差含零或负值。

**排查：**
```python
std = relative_error * np.abs(dobs) + noise_floor
print(f"最小标准差: {np.min(std)}")
```

**解决：**
- 增大 `relative_error`（如 0.05 = 5%）
- 增大 `noise_floor`（如 1e-12）
- 或直接提供 `absolute_std`

### Q3.2: 反演不收敛（达到 max_iter 但 φ_d 仍很大）

**可能原因：**
1. 初始模型偏离太远
2. 正则化太强（alpha_s 太大）
3. β 冷却太快

**排查：**
```python
print(f"最终 φ_d: {history['final_phi_d']:.2f}")
print(f"目标 φ_d: {chifact * nD:.2f}")
```

**解决：**
```python
inv = LWDFDEMInversion(
    forward=fwd,
    data_obj=data_obj,
    m0=np.log(0.1 * np.ones(n_params)),    # 调整初始模型
    alpha_s=1e-4,                           # 减小正则化
    beta_cooling=2.0,                       # 降低冷却速度
    max_iter=60,                            # 增加迭代
    chifact=1.5,                            # 放宽目标
)
```

### Q3.3: 反演结果过度光滑

**原因：** L2 正则化天然倾向平滑；alpha_s 过大。

**解决：**
```python
# 方法 1: 减小 alpha_s
inv = LWDFDEMInversion(..., alpha_s=1e-4)

# 方法 2: 使用稀疏正则化
inv = LWDFDEMInversion(
    ...,
    sparse=True,
    norms=[0, 2],      # L0 模型 + L2 梯度
    alpha_s=1e-4,
)
```

### Q3.4: 反演结果振荡（棋盘格）

**原因：** 正则化太弱或网格过粗。

**解决：**
```python
# 增大平滑权重
inv = LWDFDEMInversion(
    ...,
    alpha_s=1e-2,          # 增大
    alpha_x=2.0,           # 增大方向平滑
    alpha_z=2.0,
)
```

### Q3.5: `sparse=True` 时报错 `BetaSchedule 与 UpdateIRLS 冲突`

**说明：** 此 bug 已修复（commit 5868c84）。若仍遇到，请更新代码：
```bash
git pull
python setup.py install
```

### Q3.6: 2D 反演内存不足

**原因：** 2D 网格单元数过多，灵敏度矩阵太大。

**解决：**
```python
# 方法 1: 减小网格
mesh = build_2d_cyl_mesh(r_core=1.5, z_core=3.0, cell_size=0.2, ...)

# 方法 2: 使用迭代求解器
from pymatsolver import BicgStabSolver
fwd = LWDFDEMForward("2d_cyl", tool, mesh=mesh, solver=BicgStabSolver)

# 方法 3: 减少频率/收发距数量
```

---

## 4. 可视化

### Q4.1: 中文标注显示为方框

**原因：** Matplotlib 默认字体不支持中文。

**解决：**
```python
from src.lwd_fdem import set_cjk_rcparams
set_cjk_rcparams()    # 启用 SimHei / Microsoft YaHei
```

### Q4.2: 对数坐标图出现 `-inf` 或标签异常

**原因：** 数据含零或负值，`semilogx` 无法处理。

**解决：**
```python
from src.lwd_fdem import fix_log_labels
# 绘图后调用
fix_log_labels(ax)
```

### Q4.3: 图片保存模糊

**原因：** DPI 太低。

**解决：**
```python
from src.lwd_fdem import save_figure
save_figure(fig, "output.png", dpi=300)    # 300 DPI 用于印刷
```

---

## 5. 性能与内存

### Q5.1: 3D 正演非常慢

**原因：** 3D FVM 网格单元数多，线性方程组大。

**优化建议：**
1. **减小网格规模：** 缩小 `core_extent` 或增大 `cell_size`
2. **使用高效求解器：**
   ```python
   from pymatsolver import PardisoSolver    # 需安装 Pardiso
   fwd = LWDFDEMForward("3d", tool, mesh=mesh, solver=PardisoSolver)
   ```
3. **减少频率/源数量：** 先跑少量频率验证，再全量计算
4. **并行计算：** 不同频率可独立计算（目前未实现自动并行）

### Q5.2: 内存占用过高

**原因：** 3D 网格单元数过多，或反演存储灵敏度矩阵。

**解决：**
```python
# 方法 1: 减小网格
mesh = build_3d_tensor_mesh(core_extent=(3, 3, 4), cell_size=0.5, ...)

# 方法 2: 反演时不存储完整灵敏度
# （当前默认不存储，仅 Jvec/Jtvec 算子）

# 方法 3: 使用迭代求解器（内存 < 直接法）
from pymatsolver import BicgStabSolver
fwd = LWDFDEMForward("3d", tool, mesh=mesh, solver=BicgStabSolver)
```

### Q5.3: 如何估算运行时间？

**经验公式：**
- 1D 正演：毫秒级（可忽略）
- 2D 正演：~ 秒级（10³–10⁴ 单元）
- 3D 正演：~ 分钟级（10⁴–10⁵ 单元）
- 反演：正演时间 × 迭代次数 × (1 + 梯度计算开销)

粗略估算：
```python
n_cells = mesh.n_cells
n_freq = len(frequencies)
n_iter = 30

# 正演时间 ~ n_cells * n_freq * 1e-4 秒（3D 经验系数）
est_time = n_cells * n_freq * 1e-4
print(f"预估正演时间: {est_time:.1f} s")
print(f"预估反演时间: {est_time * n_iter * 2:.1f} s")    # ×2 含梯度
```

---

## 6. 物理与数值

### Q6.1: 趋肤深度公式是什么？

**全频域 Maxwell（含位移电流）：**

```
δ = 1 / |Im(k)|
k = √(ω²με − iωμσ)
```

**准静态近似（σ ≫ ωε，低频/高导）：**

```
δ ≈ 503 / √(f · σ)    (m)
```

其中 f 单位 Hz，σ 单位 S/m。

**代码计算：**
```python
from src.lwd_fdem import skin_depth
delta = skin_depth(frequency=1e4, conductivity=0.01)
print(f"趋肤深度: {delta:.1f} m")    # ~159 m
```

### Q6.2: 为什么 1D 必须用 `receiver_geometry='radial'`？

**原因：** 1D 层状模型使用 Hankel 变换求解，数学上要求源和接收器在同一水平面，偏移为径向距离。轴向偏移破坏层状对称性，需要 2D/3D 模拟。

**物理含义：**
- `radial`：接收器在水平方向偏离源（垂直井径向仪器）
- `axial`：接收器沿井轴方向偏离源（垂直井轴向仪器）

### Q6.3: `secondary=True` 和 `secondary=False` 的区别？

**`secondary=True`（二次场）：**
- 接收器只记录散射场（总场 − 一次场）
- 对异常体更灵敏（一次场背景被减掉）
- 适用于金属矿勘探

**`secondary=False`（总场）：**
- 接收器记录完整电磁场
- 适用于地层电阻率测量（LWD 常用）
- 一次场提供几何信息

**代码层面：**
- `secondary=True` → 使用 `PointMagneticFieldSecondary` 接收器
- `secondary=False` → 使用 `PointMagneticField` 接收器

### Q6.4: Cole-Cole 频散的物理意义？

**Cole-Cole 模型描述岩石极化效应引起的电导率随频率变化：**

```
σ(ω) = σ∞ · [1 − η · (1 − 1/(1 + (iωτ)^c))]
```

**参数含义：**
- `η`（充电子率）：极化强度，0 = 无极化，→1 = 强极化
- `τ`（弛豫时间）：极化建立的时间尺度
- `c`（频率指数）：频散过渡带宽度

**应用场景：**
- 页岩（含导电矿物）
- 金属矿（硫化物）
- 激发极化 (IP) 效应建模

### Q6.5: 网格分辨率要求？

**判据：** 每个趋肤深度至少 3 个单元

```
cell_size ≤ δ_min / 3
```

**检查：**
```python
from src.lwd_fdem import mesh_quality_report
report = mesh_quality_report(mesh, frequencies, conductivities)
if not report['ok']:
    print(f"网格不合格: 每趋肤深度仅 {report['cells_per_skin_depth']:.1f} 个单元")
    print("建议: 减小 cell_size 或调整 frequencies/conductivities")
```

### Q6.6: 为什么反演用 `log(sigma)` 而不是 `sigma`？

**原因：**
1. **保证正性：** 电导率必须 > 0，`exp(log_sigma)` 自动满足
2. **改善收敛：** 对数参数化使目标函数更接近二次型
3. **物理意义：** 电导率变化常为倍数关系（如 0.01 → 0.1 → 1.0），对数空间更均匀

**代码层面：**
```python
# 反演输入为 log(sigma)
m0 = np.log(0.1 * np.ones(n_params))    # 初始模型
inv = LWDFDEMInversion(forward, data_obj, m0=m0)
sigma_rec, history = inv.run()           # 输出为 sigma（已 exp）
```

---

## 未解决的问题？

1. 检查 `examples/` 目录中的示例脚本
2. 查阅 `docs/PARAMETER_REFERENCE.md`（参数速查）
3. 查阅 `docs/USER_GUIDE.md`（使用手册）
4. 在 GitHub Issues 提问：https://github.com/swpucwf/LWD-EM-Inv/issues
