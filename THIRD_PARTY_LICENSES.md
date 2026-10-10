# 第三方许可证声明

本项目以 Apache-2.0 许可发布，但 `src/fdemcore/` 目录包含源自 **SimPEG** 项目的派生代码，遵循 SimPEG 的 MIT 许可。

## SimPEG 派生代码

- 来源项目：SimPEG（https://simpeg.xyz）
- 来源版本：v0.25.1.dev3（commit 0528d9028，本地检出存档于 `external/SimPEG-src/`）
- 许可类型：MIT License
- 处理方式：按「精简代码库」方案裁剪出频域电磁（FDEM）正反演所需模块，保留原目录结构与文件内版权头；顶层绝对导入由 `simpeg` 改写为 `src.fdemcore`；`__init__.py` 做了导出瘦身。
- 版权原文：

```text
The MIT License (MIT)

Copyright (c) 2013 - 2024, SimPEG Team, https://simpeg.xyz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## 其他运行时依赖（各自遵循其许可证）

| 包 | 用途 | 许可 |
|---|---|---|
| discretize | 有限体积网格与差分算子 | MIT |
| pymatsolver | 稀疏线性求解器封装 | MIT |
| numpy | 数值计算 | BSD-3 |
| scipy | 稀疏矩阵/LU/插值 | BSD-3 |
| matplotlib | 可视化 | PSF-like (BSD兼容) |
| geoana | 解析解（对标验证） | MIT |
| libdlf | Hankel 数字滤波系数 | MIT |
