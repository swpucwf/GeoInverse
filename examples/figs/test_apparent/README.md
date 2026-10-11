# 视电阻率测试 (test_apparent)

## 测试概述

**测试功能**: E5 视电阻率转换。

**运行命令**: `python examples/test_apparent_resistivity.py`

---

## 图版说明

### apparent_resistivity.png — 视电阻率矩阵
**表达内容**: 左图为 Hz 振幅数据矩阵 (log10)，右图为反演的视电阻率矩阵 (log10)。半空间模型 σ=0.1 S/m。

**测试验证**: 验证 `apparent_resistivity_from_hz` 和 `apparent_resistivity_matrix` 正确从电磁数据反演视电阻率。
