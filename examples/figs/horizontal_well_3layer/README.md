# 水平井三层案例 (horizontal_well_3layer)

## 案例概述

**物理场景**: 水平井 LWD-FDEM 正演与反演——仪器在水平井眼中沿轨迹移动。

**地层模型**: 三层水平地层，水平井眼穿越储层。

**运行命令**: `python examples/horizontal_well_3layer.py`

---

## 图版说明

### 01_concept_model.png — 概念模型
**表达内容**: 水平井穿透三层地层的概念模型示意图。

### 02_mesh.png — 计算网格
**表达内容**: 2D 轴对称柱网格的 r-z 剖面。

### 03_survey.png — 观测系统
**表达内容**: 源-接收器排列及观测点分布。

### 04_forward_response.png — 正演响应
**表达内容**: 多频率正演数据随仪器位置的变化曲线。

### 05_synthetic_data.png — 合成数据
**表达内容**: 添加噪声后的合成观测数据。

### 06_inversion.png — 反演结果
**表达内容**: 1D/2D 反演 recovered 模型与真实模型对比。

### 07_convergence.png — 收敛曲线
**表达内容**: 反演迭代过程中目标函数 φ_d 和模型范数 φ_m 的收敛历史。
