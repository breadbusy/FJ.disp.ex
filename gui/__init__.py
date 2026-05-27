"""FJ 交互式 GUI 工具包。

提供两个独立的 GUI 应用:
  - subarray_splitter: 子阵列分割工具 (Step 1)
  - dispersion_extractor: 频散提取工具 (Step 2)

共享组件:
  - common: StatusBar, LabeledSpinbox, LabeledEntry, FileSelector,
    ParamGroup, run_in_thread, EmbeddedPlot
"""

from .common import (
    EmbeddedPlot,
    FileSelector,
    LabeledEntry,
    LabeledSpinbox,
    ParamGroup,
    StatusBar,
    run_in_thread,
)

__all__ = [
    "StatusBar",
    "LabeledSpinbox",
    "LabeledEntry",
    "FileSelector",
    "ParamGroup",
    "run_in_thread",
    "EmbeddedPlot",
]
