# 高级功能测试 (test_advanced)

## 测试概述

**测试功能**: E6+E7+E8 LAS 导出、并行频率计算、灵敏度矩阵。

**运行命令**: `python examples/test_advanced.py`

---

## 图版说明

### sensitivity_matrix.png — 灵敏度矩阵
**表达内容**: 左图为灵敏度矩阵 J (nD×nP)，右图为 JᵀJ 对角元（内存友好版）。

**测试验证**: 验证 `compute_sensitivity_matrix` 和 `compute_sensitivity_diagonal` 正确计算灵敏度信息。
