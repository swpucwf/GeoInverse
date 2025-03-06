
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
本项目基于 **Python 3.8+**，推荐使用 **Anaconda** 进行环境管理。  

```bash
# 创建新的 Python 环境
conda create -n lwd-eminv python=3.8
conda activate lwd-eminv

# 安装依赖库
pip install numpy scipy matplotlib tqdm
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
### **🔹 1. 运行示例**
```python
from lwd_eminv import LWDInversion

# 初始化反演
inv = LWDInversion(method="Occam", max_iter=100)

# 加载测井数据
data = inv.load_data("example_data.csv")

# 运行反演
result = inv.run_inversion(data)

# 可视化结果
inv.plot_result(result)
```

### **🔹 2. 命令行运行**
```bash
python run_inversion.py --method Occam --data example_data.csv
```

---

## **📊 结果示例**

### **🔹 反演结果可视化**
```python
# 绘制反演结果
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 绘制测量数据与拟合数据对比
inv.plot_data_fit(result, ax=ax[0])

# 绘制反演模型
inv.plot_model(result, ax=ax[1])

plt.tight_layout()
plt.show()
```

### **🔹 参数敏感性分析**
```python
# 进行参数敏感性分析
sensitivity = inv.analyze_sensitivity(result)

# 绘制敏感性矩阵
inv.plot_sensitivity(sensitivity)
```

---

## **🛠 目录结构**
```
LWD-EM-Inv/
│── docs/                # 文档与示例
│── examples/            # 示例代码
│── src/                 # 核心代码
│   ├── lwd_eminv/       # 反演模块
│   ├── forward/         # 正演模块
│   ├── utils/           # 工具库
│── tests/               # 测试代码
│── run_inversion.py     # 反演主脚本
│── README.md            # 项目说明
│── setup.py             # 安装脚本
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
📌 **GitHub**: [your_username](https://github.com/swpucwf)  

---

## **📜 许可证**
本项目基于 **  Apache License Version 2.0** 开源，详细信息请见 [LICENSE](LICENSE)。

