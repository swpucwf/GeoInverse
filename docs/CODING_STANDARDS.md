# LWD-EM-Inv 编码规范

> 本规范适用于仓库内全部 Python 代码，包括 `src/` 业务模块、`examples/` 与 `tests/`。
> `src/fdemcore/` 为 SimPEG 精简派生代码，允许保留其上游风格，但任何新增/修改行须遵循本规范。

## 1. 语言版本与基本风格

- Python >= 3.10（当前开发环境 3.13），仅使用标准库 + 已声明依赖（见 `requirements.txt`），禁止引入未声明的隐式依赖。
- 遵循 PEP 8：4 空格缩进；行宽 <= 100；导入顺序为标准库 → 第三方 → 本项目，禁止 `import *`（`__init__.py` 的显式再导除外）。
- 命名：模块/函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`，私有成员前缀 `_`。

## 2. 类型标注与文档字符串

- 所有公开函数、方法、类属性必须有完整类型标注（`typing` 或 PEP 604 联合写法）。
- docstring 使用中文 Google 风格（与现有 `src/forward`、`src/inversion` 模块一致）：

```python
def build_survey(frequencies: list[float], spacings: np.ndarray) -> Survey:
    """构建 LWD 多频多偏移距观测系统。

    Args:
        frequencies: 工作频率列表，单位 Hz。
        spacings: 收发距数组，单位 m。

    Returns:
        频域电磁 Survey 对象。

    Raises:
        ValueError: 当频率或收发距非正时抛出。
    """
```

- 类必须在 docstring 的 `Attributes` 段说明每个实例属性的物理含义与单位。

## 3. 物理量与单位约定（SI）

| 量 | 变量名 | 单位 |
|---|---|---|
| 电导率 | `sigma` / `conductivity` | S/m |
| 电阻率 | `resistivity` | Ohm·m |
| 频率 | `frequency` / `frequencies` | Hz |
| 收发距（轴向偏移） | `spacing` / `offset` | m |
| 径向距离 | `radius` / `r` | m |
| 磁场（归一化） | `Hpp` / ppm 数据 | ppm（相对一次场） |
| 磁偶极矩 | `moment` | A·m^2 |

- 电阻率与电导率换算一律 `sigma = 1.0 / rho`，禁止在公式中混用两种参数化。
- 时间因子约定沿用 SimPEG FDEM 的 `exp(+i·omega·t)`；位移电流项按全频域 Maxwell 保留。

## 4. 数组与数值约定

- 模型向量（反演参数）为一维 `np.ndarray`，形状 `(n_params,)`；网格数据由映射（`maps`）负责展开为 `(n_cells,)`。
- 正演输出数据向量形状 `(nD,)`，排列顺序由 `Survey` 的 源×接收器×分量 索引唯一确定；需要按频率重排时，通过 `survey.source_list[i].receiver_list` 推导索引，禁止魔法数。
- 任何构造数组的函数须在 docstring 中标注形状与物理含义（如 `(nFreq, nSpacing)`）。
- 随机数必须接受 `seed` 参数并经 `np.random.default_rng(seed)` 生成，保证可复现。
- 稀疏矩阵使用 `scipy.sparse`；稠密化仅允许发生在规模可控（<= 1e4 阶）的调试路径。

## 5. 参数校验与异常

- 公开接口的入参校验统一使用 `src/fdemcore/utils` 提供的 `validate_float / validate_string / validate_type / validate_ndarray_with_shape` 等工具，保证报错文案风格一致。
- 物理非法输入（负频率、负电导率、空观测系统）抛 `ValueError`；未实现路径抛 `NotImplementedError`；禁止裸 `except:` 与静默吞异常。
- 库代码不使用 `print` 报告过程信息，统一使用 `src.fdemcore.utils.get_logger("...")`；仅 `examples/` 与命令行脚本允许 `print` 汇报进度。

## 6. 测试规范

- 测试置于 `tests/`，文件命名 `test_*.py`，函数命名 `test_<行为>`；使用 pytest，禁止依赖执行顺序。
- 每个物理/数值模块必须有对标测试：解析解对比、交叉验证（2D 圆柱 vs 3D 直角）、`sigma -> 0` 一致性、梯度校验（`check_derivative`）。
- 数值容差须写为常量并注释来源（如 `REL_TOL_XCHECK = 0.15`）；冒烟类测试用 `@pytest.mark.slow` 标记。
- 新增反演配置必须附带一次小规模合成数据回收测试（拟合归一化残差 ~ 1）。

## 7. 提交与评审

- 提交信息格式：`<模块>: <祈使句摘要>`（例：`lwd_fdem: add skin-depth mesh check`）；一次提交只做一件事。
- 涉及 `src/fdemcore/` 的改动必须在 PR 描述中说明与上游 SimPEG 的差异及原因。
- 合入前 requirements：pytest 全绿、示例可运行、文档（README/docstring）同步更新。

## 8. 许可证头

- 本项目原创文件：Apache-2.0 头（见 `LICENSE`），新文件加模块级 docstring 说明版权。
- SimPEG 派生文件（`src/fdemcore/`）：保留上游 MIT 版权头，不做删改；详见 `THIRD_PARTY_LICENSES.md`。
