# 论文复现：EM 随钻前视测井（look-ahead LWD）正演物理模型

本目录复现两篇 EM 随钻前视测井深度学习反演论文的**正演物理模型与敏感性分析**部分
（DL 反演不在复现范围内）。两篇论文出自同一课题组（中科院 张文秀团队）。

每篇论文均提供**两个独立版本**并做数值对比：

1. **lwd_fdem 版本**：使用本项目 `src.lwd_fdem` 业务层 API；
2. **SimPEG 版本**：直接使用 `external/simpeg-main`（v0.25.1）底层 API，经 `sys.path` 导入，
   不 pip install、不经过 lwd_fdem 业务层。

由于 `fdemcore` 是 SimPEG FDEM 核心的字节级一致精简版，两版本在相同网格 / 源 /
接收器 / 电导率下给出**比特级一致**的正演结果（实测 `rel_err = 0.0`）。

---

## 目录结构

```
paper/
├── README.md               # 本说明
├── _common.py              # 共享的纯 numpy 模型定义与后处理工具（两版本共用，确保同物理量对比）
├── paper1_lwd_fdem.py      # 论文1 正演复现（lwd_fdem 版本）
├── paper1_simpeg.py        # 论文1 正演复现（SimPEG 版本）
├── paper2_lwd_fdem.py      # 论文2 敏感性复现（lwd_fdem 版本）
├── paper2_simpeg.py        # 论文2 敏感性复现（SimPEG 版本）
├── compare_results.py      # 两版本数据对比脚本
├── paper1_lwd_fdem.npz     # 运行后生成，供 compare_results.py 加载
├── paper1_simpeg.npz
├── paper2_lwd_fdem.npz
├── paper2_simpeg.npz
└── figs/                   # 输出图件（PNG, dpi=150）
```

---

## 环境与运行

需要已安装项目依赖的 Python 环境（见仓库根目录 `requirements.txt`）。每个脚本均可独立运行，支持两档模式：

- `--quick`（默认）：粗网格 + 代表频点 + 少量扫描点，秒级—分钟级完成，用于演示与对比验证；
- `--full`：接近论文精度的细网格与完整频率 / 扫描序列，计算量大（数十分钟至数小时），
  建议在多核大内存机器上运行。

> 注：脚本默认即为 quick 档；仅当显式传入 `--full` 时才使用完整档。

### 推荐执行顺序

```bash
# 论文1（各约 15 s，quick 档）
python paper/paper1_lwd_fdem.py --quick
python paper/paper1_simpeg.py   --quick

# 论文2（各约 17 s，quick 档）
python paper/paper2_lwd_fdem.py --quick
python paper/paper2_simpeg.py   --quick

# 两版本对比
python paper/compare_results.py
```

`compare_results.py` 会加载上述 `.npz`，逐分量计算相对误差并输出 PASS/FAIL 汇总表，
判定阈值 `REL_TOL = 1e-10`。**quick 与 full 档需成对运行**（对比脚本按文件名加载，
两版本档位应一致）。

---

## 论文1：Multi-task U-net inversion of synthetic look-ahead LWD data

- 期刊：Petroleum Science 23 (2026) 1908–1928
- 仪器：单发射 + 双接收，三轴正交线圈磁偶极子
- 收发距：TR1 = 10 m，TR2 = 14 m
- 频率：10 / 20 / 30 / 50 kHz（quick 取 10 / 30 kHz）
- 地层：5 层各向异性 VTI（σh, σv），界面深度 5 / 10 / 15 / 20 m（相对仪器）
- 测量：3×3 磁场张量；衰减比 `Att_ij = -20*log10(|H_R1_ij| / |H_R2_ij|)`

5 层 VTI 电导率模型（S/m）：

| 层 | σh | σv | 说明 |
|----|-----|-----|------|
| 1 | 0.1 | 0.05 | 背景 |
| 2 | 0.5 | 0.25 | |
| 3 | 0.01 | 0.005 | 目标层（高阻） |
| 4 | 0.3 | 0.15 | |
| 5 | 0.1 | 0.05 | 基底 |

**实现要点**

- 采用 `geometry='3d'` 的 `TensorMesh` FVM（h-form）路径，`LWDFDEMForward(anisotropic=True)`；
- 用 `tensor.build_tensor_survey` 构建三轴源 × 三轴接收（9 分量）Survey，注入正演器；
- 各向异性电导率以 `(n_cells, 3)` = `[σh, σh, σv]` 传入，由 `_AnisotropicSigmaMap`
  还原为对角张量（VTI）；
- 前视深度扫描通过平移层界面（`shift`）实现：仪器固定在原点、界面向仪器靠近，
  等价于仪器向下钻进；
- SimPEG 版本手动构建 `TensorMesh`（相同 h 列表、`origin="CCC"`）、`MagDipole` 源
  （orientation 为 'x'/'y'/'z'）、`PointMagneticField` 接收器、`Simulation3DMagneticField`，
  并以本地 `_AniSigmaMap` 复刻各向异性映射。

**关键结论（供分析）**：yy 分量对各向异性最敏感；同轴(xx,zz)+共面(yy)对电导率最敏感；
交叉(xz,zx)+同轴对界面最敏感。脚本输出全部 9 分量的 Att 供进一步分析。

**输出图件**：`figs/paper1_lwd_attenuation.png`、`figs/paper1_lwd_att_vs_depth.png`、
`figs/paper1_simpeg_attenuation.png`。

论文1 两版本衰减比散点对比与差值直方图：`figs/compare_paper1.png`。

---

## 论文2：Knowledge-guided deep learning inversion of EM look-ahead LWD data

- 期刊：Petroleum Science (2026)，DOI 10.1016/j.petsci.2026.04.041
- 仪器：单发射 + 三接收 R1/R2/R3，三轴磁偶极子
- 收发距：R1 = 10 m，R2 = 20 m，R3 = 30 m
- 频率：2 / 6 / 12 / 24 / 48 / 96 kHz（quick 取 6 / 24 kHz）
  - Sub1(R1=10 m)：24 / 48 / 96 kHz；Sub2(R2=20 m)：6 / 12 / 24 / 48 kHz；Sub3(R3=30 m)：2 / 6 / 12 kHz
- 模型：2 层，上层 ρ1 = 100 Ω·m，下层 ρ2 扫描；界面深度 DTB = 5 / 15 / 30 m（quick 取 15 m）
- 测量：主分量 Hxx / Hyy / Hzz（实 / 虚部）；
  `UHRA = 20*log10|2Vzz/(Vxx+Vyy)|`，`UHRP = angle[2Vzz/(Vxx+Vyy)]`

**实现要点**

- 各向同性 2 层模型，`geometry='3d'`、`anisotropic=False`，同样注入三轴 × 三轴张量 Survey，
  取主对角分量 Hxx/Hyy/Hzz；
- 敏感性扫描：固定 DTB，改变下层 ρ2，计算各分量响应；
- 敏感度定量指标：归一化敏感度 `S = (max - min) / rms(|values|)`，比较实部与虚部；
- 因接收器最远 R3 = 30 m，论文2 使用比论文1 更大的 z 向网格（`MESH_*_P2`）。

**关键结论复现**：quick 档实测三个主分量的**虚部**对下层电阻率变化远比实部敏感
（例如 6 kHz Hxx：实部 S≈0.10，虚部 S≈1.28），与论文一致。
（各向同性水平层状模型下 Hxx 与 Hyy 因对称性相等，属预期物理结果。）

**输出图件**：`figs/paper2_lwd_components.png`、`figs/paper2_lwd_sensitivity.png`、
`figs/paper2_simpeg_components.png`、`figs/paper2_simpeg_sensitivity.png`。

论文2 两版本主分量重叠对比：`figs/compare_paper2.png`。

---

## 计算效率与网格说明

3D FVM 正演每个 `(频率, 源取向)` 组合需一次稀疏 LU 分解求解，计算量随网格单元数
超线性增长。参考实测（默认 `SolverLU`，本机）：

| 网格单元数 | 单次求解耗时（量级） |
|-----------|---------------------|
| ~1.3k（quick, cell=6–8 m） | 数秒 |
| ~6.8k（cell=4 m） | ~18 s |
| ~1e5（full, cell=2 m） | 数十分钟起 |

为兼顾可运行性，quick 档统一采用粗网格、代表频点与少量扫描点。**注意**：quick 档
网格较粗，其绝对数值不等于论文精细网格结果，但两版本使用完全相同的离散化，
故 lwd_fdem 与 SimPEG 的对比严格成立（`rel_err = 0.0`）。追求与论文一致的绝对精度
请使用 `--full` 档并配置更强算力。

> 环境中 `pydiso` 未安装，故 Pardiso 直接求解器不可用，两版本均使用默认 `SolverLU`
> 以保证结果一致。若安装 `pydiso`，可在两个脚本中同步传入相同 `solver` 以加速。

---

## 约束遵守

- 不修改 `src/fdemcore/` 与 `src/lwd_fdem/` 下任何文件。
- SimPEG 版本仅通过 `sys.path` 导入 `external/simpeg-main`，不 pip install。
  归档缺少构建期生成的 `simpeg/version.py`，脚本在导入前注入占位模块
  （仅进程内生效，不写盘、不改动归档）。
- 仅复现正演物理模型与敏感性分析，DL 反演不在范围内。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
