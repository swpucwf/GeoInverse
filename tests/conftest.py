"""pytest 公共配置：将仓库根加入 sys.path，使 `from src...` 可在未安装时导入。

Copyright 2026 LWD-EM-Inv Team (Apache-2.0)
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
