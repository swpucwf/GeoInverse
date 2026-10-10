# 贡献指南

感谢参与 LWD-EM-Inv 的开发！参与前请先阅读 [编码规范](docs/CODING_STANDARDS.md) 与 [架构说明](docs/ARCHITECTURE.md)。

## 开发环境

```bash
conda create -n lwd-eminv python=3.13
conda activate lwd-eminv
pip install -r requirements.txt
pip install pytest            # 测试
```

## 工作流程

1. 从 `main` 拉取分支，命名 `feature/<模块>-<简述>` 或 `fix/<简述>`。
2. 改动落在对应模块：`src/lwd_fdem/`（业务层）、`src/fdemcore/`（SimPEG 派生核心）、`examples/`、`tests/`、`docs/`。
3. 本地验证：
   ```bash
   pytest tests/               # 常规用例
   pytest -m slow tests/       # 含 3D/反演冒烟（耗时长）
   ```
4. 提交信息格式：`<模块>: <祈使句摘要>`，例如 `lwd_fdem: add CircularLoop source option`。
5. 发起 PR，勾选：测试通过、文档同步、新增公开 API 有中文 Google 风格 docstring 与类型标注。

## 特别约定

- **`src/fdemcore/` 为 SimPEG (MIT) 派生代码**：改动须在 PR 描述中说明与上游的差异与原因；新增文件不得移除上游版权头（见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)）。
- 数值行为改动必须附带对标证据（解析解误差、2D-3D 交叉验证或 `check_derivative` 结果）。
- Bug 报告请附最小复现脚本与 `pip list` 输出；功能建议请说明对应的测井场景（频率、偏移距、几何维度）。

## 许可

提交即表示同意您的贡献以 Apache-2.0 许可发布。
