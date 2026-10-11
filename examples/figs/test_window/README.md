# 滑动窗口测试 (test_window)

## 测试概述

**测试功能**: E10 滑动窗口反演。

**运行命令**: `python examples/test_sliding_window.py`

---

## 图版说明

### sliding_window_profile.png — 滑动窗口反演剖面
**表达内容**: 
- 左图：反演电阻率随深度变化曲线（三层模型）
- 右图：数据拟合差 φ_d 随深度变化

**测试验证**: 验证 `sliding_window_inversion_1d` 正确实现深度扫描反演，生成连续电阻率剖面。
