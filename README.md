# LWD-EM-Inv

随钻电磁波测井（LWD-EM）数据的正演模拟与反演代码库。

## 概述

LWD-EM-Inv 提供频域电磁（FDEM）随钻测井的正反演工具。计算核心 `src/fdemcore`
由 SimPEG 精简派生，实现 1D 层状（Hankel 解析）、2D 轴对称与 3D 直角的有限体积
正演；业务层 `src/lwd_fdem` 在其上封装仪器配置、网格生成、反演管线、数据读写与
可视化。`fdemcore` 不依赖 pip 安装的 SimPEG。

主要能力：

- 频域电磁正演：1D 层状、2D 轴对称、3D 直角三种几何
- Occam 风格反演：对数电导率参数化、加权最小二乘平滑、β 冷却、拟合差停止
- 观测系统：多频率、多收发距、实部/虚部，磁偶极子源与线圈源
- 合成数据与不确定度处理、模型与响应可视化

## 环境

Python 3.10 及以上（开发验证于 3.13）。`fdemcore` 与 `lwd_fdem` 依赖
`discretize`、`pymatsolver`、`geoana`、`libdlf`，版本约束见 `requirements.txt`。

## 安装

```bash
conda create -n lwd-eminv python=3.13
conda activate lwd-eminv
pip install -r requirements.txt

git clone https://github.com/swpucwf/LWD-EM-Inv.git
cd LWD-EM-Inv
python setup.py install
```

## 快速开始

正演（1D 层状，多频多收发距）：

```python
import numpy as np
from src.lwd_fdem import LWDFDEMForward, LWDToolConfig

tool = LWDToolConfig(
    frequencies=(1e4, 1e5, 5e5, 1e6),
    spacings=np.array([0.3, 0.6, 1.0]),
    components=("real", "imag"),
    field_kind="h", secondary=True, receiver_geometry="radial",
)
fwd = LWDFDEMForward("1d", tool, layer_thicknesses=np.array([2.0, 4.0]))
d = fwd.dpred(np.array([0.25, 0.01, 0.5]))
print(fwd.to_matrix(d).shape)   # (n_freq, n_spacing, n_comp)
```

反演（合成数据回收）：

```python
from src.lwd_fdem import add_synthetic_noise, make_data_object, LWDFDEMInversion

dobs, std = add_synthetic_noise(d, noise_level=0.02, seed=1)
data = make_data_object(fwd.survey, dobs, absolute_std=std)
inv = LWDFDEMInversion(fwd, data, m0=np.log(0.1 * np.ones(3)), max_iter=15)
sigma_recovered, history = inv.run()
```

2D 轴对称与 3D 几何通过 `LWDFDEMForward("2d_cyl", tool, mesh=...)` 与
`LWDFDEMForward("3d", tool, mesh=...)` 使用，网格由 `build_2d_cyl_mesh`、
`build_3d_tensor_mesh` 生成。完整脚本见 `examples/`。

## 运行示例与测试

```bash
python examples/lwd_fdem_forward_1d.py
python examples/lwd_fdem_inversion_1d.py

pytest -m "not slow"    # 快速回归
pytest                  # 含 3D 反演等耗时用例
```

## 示例

以下图版由 `examples/` 中的脚本生成。

图 1：1D 层状正演。(a) 三层地电模型与井轴上的收发线圈；(b) 二次场幅度随频率变化。

![1D 正演](docs/images/lwd_fdem_forward_1d_plate.png)

图 2：2D 轴对称正演（r-z）。(a) 电导率模型；(b) 计算网格；(c) 响应随收发距变化。

![2D 正演](docs/images/lwd_fdem_forward_2d_plate.png)

图 3：3D 直角正演（y=0 切片），井旁方块导体。

![3D 正演](docs/images/lwd_fdem_forward_3d_plate.png)

图 4：Occam 反演。左：1D 剖面回收与收敛史；右：2D 真实模型、回收模型与收敛史。

![1D 反演](docs/images/lwd_fdem_inversion_1d.png)

![2D 反演](docs/images/lwd_fdem_inversion_2d.png)

绘图接口位于 `src/lwd_fdem/plotting.py`：

```python
from src.lwd_fdem import plot_forward_figure, plot_inversion_figure, pub_style

plot_forward_figure(fwd, sigma, d, out_dir="examples/figs",
                    prefix="lwd_fdem_forward_2d")
plot_inversion_figure(fwd, rec_sigma, history, sigma_true=sigma_true,
                      out_dir="examples/figs", prefix="lwd_fdem_inversion_1d")
```

## 目录结构

```
LWD-EM-Inv/
├── src/
│   ├── fdemcore/          频域电磁 FVM 计算核心（SimPEG 精简派生，MIT）
│   └── lwd_fdem/          LWD 正反演业务层（survey/meshes/forward/inversion/plotting）
├── examples/              正反演示例脚本
├── tests/                 测试（解析对标、交叉验证、梯度校验、反演回收）
├── docs/                  架构说明、编码规范、项目知识库
├── external/SimPEG-src/   SimPEG 上游源码存档（不参与打包，不入库）
├── CONTRIBUTING.md
├── THIRD_PARTY_LICENSES.md
├── requirements.txt
└── setup.py
```

## 建模选项与待完善功能（Roadmap / TODO）

当前示例为快速演示，对若干物理与仪器维度做了简化。以下按维度列出「已实现 / 待补」的
建模选项，供后续按需完善（`[x]` 已实现，`[ ]` 待做）。

### 源（发射）

- [x] 竖直磁偶极子 `MagDipole`（点偶极，z 取向）——水平井/垂直井示例已用
- [ ] 有限尺寸圆形回线源 `CircularLoop`（线圈半径/匝数；fdemcore 已具备，示例未接入）
- [ ] 倾斜磁偶极子（工具相对层面倾斜，水平井/定向井姿态）
- [ ] 电偶极子 / 接地线源
- [ ] 多分量源（x/y/z 三轴或阵列天线）

### 接收器

- [x] 点接收（`PointMagneticField/FluxDensity`，实部/虚部分量）——当前默认
- [ ] 有限尺寸线圈接收（有效面积 N·A、沿线圈长的空间平均）
- [ ] 多分量/张量接收（同时测 Hx/Hy/Hz）
- [ ] 方位敏感接收（各向异性/边界探测的 azimuth 响应）

### 井眼几何与仪器姿态

- [x] 垂直井（2D 轴对称，轴向偏移）；水平井（1D 层状，径向偏移近似）
- [ ] 水平井的完整 3D 建模（井轴水平，非径向近似）
- [ ] 倾斜井 / 定向井（井轴与层面任意夹角）
- [ ] 仪器偏心 / 靠壁（工具不在井眼中心）
- [ ] 真实井眼轨迹（旋转导向、非直线）

### 地层模型

- [x] 各向同性水平层状（1D）/ 轴对称 r-z（2D）/ 直角（3D）
- [ ] 电导率各向异性（VTI / HTI，页岩层理）
- [ ] 倾斜地层 / 断层 / 非层状界面
- [ ] 侵入剖面（冲刷带/过渡带/原状地层 radial invasion）
- [ ] 介电色散与频变（Cole-Cole，宽频 ε、σ 频散）
- [ ] 非均匀磁导率

### 物理近似

- [x] 全频域 Maxwell（保留位移电流），`exp(+iωt)` 约定
- [ ] 准静态 vs 全波的显式对比（高阻/高频下位移电流）
- [ ] 源-接收互易性建模与加速

### 反演

- [x] Occam 风格：log-σ 参数化、WLS 平滑、β 冷却、目标拟合差停止
- [ ] 各向异性参数反演（σh、σv 联合）
- [ ] 联合反演（多分量 / 多频率 / 多仪器 / 与常规测井）
- [ ] 结构 / 稀疏 / 参数化正则化（层厚+电导率界面参数化）
- [ ] 不确定度与分辨率分析（贝叶斯 / 灵敏度 / 探测深度 DOI）
- [ ] 快速反演（降阶模型 / 插值字典 / 实时成像）

### 数据处理

- [x] 合成高斯噪声 + 绝对/相对标准差
- [ ] 环境校正（井眼、温度、泥浆）
- [ ] 仪器标定（tool factor、幅度/相位标定到视电导率）
- [ ] 实测数据格式接入（CSV / LAS 深度块 / 厂商格式）

### 数值与网格

- [x] TensorMesh / CylindricalMesh + FVM + SuperLU/BiCGStab
- [ ] 自适应网格加密（界面/源附近）
- [ ] 非结构化 / 四面体网格（复杂井轨迹）
- [ ] 高效求解器（Pardiso/MUMPS/多重网格）与多源多频互易加速

> 待做说明：线圈接收器已单独评估——点接收在 ℓ≪δ 下成立，效应①（N·A·ω）为标量/单位/相位，
> 效应②（有限尺寸空间平均）为二阶小量；推荐实现为业务层「线圈→多点线性组合」，不改 fdemcore。
> 待需要接实测 / 物理电压 / 复现含回线论文时启用。

## 引用

反演算法参考 Occam 反演（Constable et al., 1987, *Geophysics*）与反演问题理论
（Tarantola, 2005, *SIAM*）。计算核心派生自 SimPEG。

## 许可证

本项目以 Apache-2.0 许可发布，详见 [LICENSE](LICENSE)。`src/fdemcore` 派生自
SimPEG（MIT 许可），第三方许可声明见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

## 联系

作者：swpucwf — swpucwf@126.com — https://github.com/swpucwf
