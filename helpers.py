\
"""Backward-compatible helpers.

This file keeps the original public surface of `helpers.py` while the actual
implementation lives in `sard_pipeline.utils`.
"""

from sard_pipeline.utils import (
    LogEntry,
    log_debug,
    require,
    time_call,
)

# Old names kept for compatibility with existing imports.
_log_debug = log_debug
_require = require
_timed_function = time_call

__all__ = [
    "LogEntry",
    "log_debug",
    "require",
    "time_call",
    "_log_debug",
    "_require",
    "_timed_function",
]
