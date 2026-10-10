
# **LWD-EM-Inv**  
📌 **随钻电磁波通用地球物理反演（LWD Electromagnetic Inversion）**  

## **📖 项目简介**  
**LWD-EM-Inv** 是一个用于 **随钻电磁波测井（LWD-EM）** 数据的 **地球物理反演** 代码库，支持 **正演模拟、反演求解、数据处理** 及 **可视化**。本项目旨在提供 **高效、通用、可扩展** 的随钻电磁波反演工具，适用于 **油气勘探、地下结构成像** 等应用。  

### **✨ 主要特性**
✅ **支持多种反演方法**（Occam 反演、贝叶斯反演、深度学习等）  
✅ **适用于随钻电磁测井数据**（多频率、多偏移距）  
✅ **可扩展的正演与反演框架**（支持 FEM、FDTD、积分方程等）  
✅ **高效计算**（支持 GPU 加速）  
✅ **Python & C++ 实现**（可与 NumPy、PyTorch 等集成）  

---

## **📦 安装指南**
### **🔹 依赖环境**
本项目基于 **Python 3.10+**（开发验证于 3.13），推荐使用 **Anaconda** 进行环境管理。
频域电磁 FVM 正反演（`src/fdemcore` + `src/lwd_fdem`）依赖 **discretize / pymatsolver / geoana / libdlf**，但**不依赖 pip 安装的 SimPEG**（SimPEG 相关算法已按精简方案内置）。

```bash
# 创建新的 Python 环境
conda create -n lwd-eminv python=3.13
conda activate lwd-eminv

# 安装依赖库
pip install -r requirements.txt
pip install torch  # 如使用深度学习
```

### **🔹 源码安装**
```bash
git clone https://github.com/your_username/LWD-EM-Inv.git
cd LWD-EM-Inv
python setup.py install
```

---

## **🚀 快速开始**
### **🔹 1. 频域电磁正演（1D 层状，多频多偏移距）**
```python
import numpy as np
from src.lwd_fdem import LWDFDEMForward, LWDToolConfig

# 定义仪器：4 频率 × 3 收发距，记录 Hz 二次场实/虚部
tool = LWDToolConfig(
    frequencies=(1e4, 1e5, 5e5, 1e6),
    spacings=np.array([0.3, 0.6, 1.0]),
    components=("real", "imag"),
    field_kind="h", secondary=True, receiver_geometry="radial",
)
# 三层地电模型（2 m / 4 m 层 + 半空间）
fwd = LWDFDEMForward("1d", tool, layer_thicknesses=np.array([2.0, 4.0]))
d = fwd.dpred(np.array([0.25, 0.01, 0.5]))   # (nD,) 响应
print(fwd.to_matrix(d).shape)                 # (n_freq, n_spacing, n_comp)
```

### **🔹 2. Occam 风格反演（合成数据回收）**
```python
from src.lwd_fdem import add_synthetic_noise, make_data_object, LWDFDEMInversion

dobs, std = add_synthetic_noise(d, noise_level=0.02, seed=1)
data = make_data_object(fwd.survey, dobs, absolute_std=std)
inv = LWDFDEMInversion(fwd, data, m0=np.log(0.1 * np.ones(3)), max_iter=15)
sigma_recovered, history = inv.run()
```

2D 轴对称与 3D 几何同样支持：`LWDFDEMForward("2d_cyl", tool, mesh=...)` /
`("3d", tool, mesh=...)`，网格由 `build_2d_cyl_mesh` / `build_3d_tensor_mesh`
生成并自带趋肤深度校核。完整脚本见 `examples/lwd_fdem_*.py`。

### **🔹 3. 运行示例与测试**
```bash
# 端到端示例（生成 figures 到 examples/figs/）
python examples/lwd_fdem_forward_1d.py
python examples/lwd_fdem_inversion_1d.py

# 验证套件（解析对标 / 2D-3D 交叉 / 梯度校验 / 反演回收）
pytest -m "not slow"      # 快速回归
pytest                    # 含 3D 反演等耗时用例
```

---

## **📊 案例与示意图**

> 下列图版均由 `examples/lwd_fdem_*.py` 生成，采用 SCI 论文风格：衬线字体 +
> LaTeX 物理记法、内向刻度、(a)(b)(c) 多联面板、300 dpi。绘制内容约定参考
> `external/SimPEG-src/fdem_forward_validation`。

### **案例 1 · 1D 层状正演（4 频 × 3 收发距，径向 Hz 二次场）**
三层地电模型（2 m / 4 m 层 + 半空间，σ = 0.25 / 0.01 / 0.5 S/m）。
(a) 层状模型示意（含井眼与 Tx/Rx 线圈）；(b) \|H_z^s\|–频率 频散曲线。

![1D 正演图版](docs/images/lwd_fdem_forward_1d_plate.png)

### **案例 2 · 2D 轴对称正演（r-z，环状低阻异常）**
(a) log10(σ) 模型剖面；(b) 计算网格（core+padding 疏密 + 收发线圈）；(c) \|d\|–收发距。

![2D 正演图版](docs/images/lwd_fdem_forward_2d_plate.png)

### **案例 3 · 3D 直角正演（y=0 切片，井旁方块导体）**

![3D 正演图版](docs/images/lwd_fdem_forward_3d_plate.png)

### **案例 4 · Occam 反演结果回收（合成数据）**
1D 剖面回收 + 收敛史；2D 真实/回收模型 + 收敛史。

![1D 反演图版](docs/images/lwd_fdem_inversion_1d.png)
![2D 反演图版](docs/images/lwd_fdem_inversion_2d.png)

### **🔹 图版绘制 API**
```python
from src.lwd_fdem import plot_forward_figure, plot_inversion_figure, pub_style

# 正演：模型/网格/响应合成一张多联图版（1D 为 2 联），返回文件路径
plate = fwd.plot_figure(sigma, d, out_dir="examples/figs",
                        prefix="lwd_fdem_forward_2d_plate",
                        labels=("overburden", "resistive", "baseline"))

# 反演：真实/回收剖面 + 收敛史图版
plot_inversion_figure(fwd, rec_sigma, history, sigma_true=sigma_true,
                      out_dir="examples/figs", prefix="lwd_fdem_inversion_1d")

# 也可单独取用某一面板，或套用统一出版样式
from src.lwd_fdem import plot_model, plot_mesh
with pub_style():
    plot_mesh(fwd, sigma=sigma)   # 网格图（log10 σ + 网格叠加）
```

---

## **🛠 目录结构**
```
LWD-EM-Inv/
│── docs/                     # 文档（编码规范、架构说明、项目知识库）
│   ├── CODING_STANDARDS.md   # 编码规范
│   ├── ARCHITECTURE.md       # SimPEG 架构拆解与模块映射
│   ├── PROJECT_MEMORY.md     # 项目知识库（长期记忆的仓库内镜像）
│   └── images/               # README 案例示意图（由 examples 生成）
│── examples/                 # 示例代码（LWD FDEM 正反演等）
│── src/                      # 核心代码
│   ├── forward/              # 解析法正演模块（1D/2D）
│   ├── inversion/            # 自研反演模块（Occam 2D）
│   ├── model/                # 地层模型与网格工具
│   ├── fdemcore/             # 频域电磁 FVM 计算核心（SimPEG 精简派生，MIT）
│   └── lwd_fdem/             # LWD 频域电磁波正反演业务层
│── tests/                    # 测试代码（解析对标/交叉验证/梯度校验/反演回收）
│── external/SimPEG-src/      # SimPEG 上游源码存档（vendor 来源，不参与打包）
│── CONTRIBUTING.md           # 贡献指南
│── THIRD_PARTY_LICENSES.md   # 第三方许可证声明
│── README.md                 # 项目说明
│── setup.py                  # 安装脚本
│── requirements.txt          # 依赖清单
```

---

## **📄 参考文献**
- Constable, S. C., Parker, R. L., & Constable, C. G. (1987). Occam’s inversion: A practical algorithm for generating smooth models from electromagnetic sounding data. *Geophysics*.  
- Tarantola, A. (2005). Inverse problem theory and methods for model parameter estimation. *SIAM*.  

---

## **🤝 贡献指南**
欢迎贡献代码！请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何提交 PR、报告 Bug 或提出新功能建议。  

---

## **📧 联系方式**
📌 **作者**: swpucwf
📌 **邮箱**: swpucwf@126.com  
📌 **GitHub**: https://github.com/swpucwf

---

## **📜 许可证**
本项目基于 **  Apache License Version 2.0** 开源，详细信息请见 [LICENSE](LICENSE)。

