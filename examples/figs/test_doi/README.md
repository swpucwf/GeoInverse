# DOI 分析测试 (test_doi)

## 测试概述

**测试功能**: E9 DOI 探测深度分析。

**运行命令**: `python examples/test_doi_analysis.py`

---

## 图版说明

### doi_analysis.png — DOI 分析图
**表达内容**: 
- 左图：各层灵敏度对角元（柱状图，log 尺度）
- 中图：分辨率矩阵 R = (JᵀJ + αI)⁻¹JᵀJ
- 右图：DOI 深度示意（绿色=可靠，红色=不可靠）

**测试验证**: 验证 `doi_1d_sensitivity`、`doi_resolution_matrix` 正确评估反演的探测深度和可靠性。
