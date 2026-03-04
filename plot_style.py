"""
兼容层：历史上 `plot_style.py` 位于仓库根目录。

现在的唯一真源是：`src/disaster/plot_style.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from disaster.plot_style import *  # noqa: F401,F403
