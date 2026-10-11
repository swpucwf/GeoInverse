# 张量接收测试 (test_tensor)

## 测试概述

**测试功能**: E1+E2 多分量张量接收与多分量源阵列。

**运行命令**: `python examples/test_tensor_survey.py`

---

## 图版说明

### tensor_survey_layout.png — 张量 Survey 布局
**表达内容**: 三轴源 × 三轴接收的张量观测系统布局图。每个子图展示一个源取向下的接收器位置和取向分布。

**测试验证**: 验证 `build_tensor_receivers` 和 `build_tensor_survey` 正确构建多分量观测系统。
