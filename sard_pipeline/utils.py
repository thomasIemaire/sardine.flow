"""Small utilities used across the pipeline.

This module intentionally stays dependency-light.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class LogEntry:
    """A structured log entry emitted by :func:`time_call`."""
    function_name: str
    elapsed_time: float
    params: str
    result: str


def log_debug(message: str, debug: bool = False, tags: Optional[Iterable[str]] = None) -> None:
    """Print a debug message when *debug* is True."""
    if not debug:
        return

    tags_str = f"[{']['.join(tags)}]" if tags else ""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} [DEBUG]{tags_str} :\t {message}")


def require(dep: Any, name: str) -> None:
    """Raise a clear error if an optional dependency is missing."""
    if dep is None:
        raise RuntimeError(
            f"Dépendance manquante: {name}. "
            f"Vérifiez que le package est installé (voir requirements.txt)."
        )


def _supports_kwarg(func: Callable[..., Any], kw: str) -> bool:
    """Best-effort check whether *func* accepts a given keyword argument."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # C-extensions / builtins: assume it accepts **kwargs (best effort).
        return True

    if kw in sig.parameters:
        return True

    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _json_dumps_safe(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def resolve_device(device: str, debug: bool = False, log_tag: str = "device") -> str:
    """
    Resolve the device to use, falling back to CPU if CUDA is requested 
    but unavailable or incompatible with the current PyTorch version.
    """
    requested = (device or "cpu").lower()
    if requested == "cpu":
        return "cpu"

    # Lazy import to keep the module dependency-light at top-level
    try:
        import torch
    except ImportError:
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            log_debug(
                "CUDA indisponible; exécution sur CPU.",
                debug,
                tags=[log_tag],
            )
            return "cpu"
        
        # Check architecture compatibility (e.g. prevent RTX 5070 crash on older PyTorch)
        try:
            capability = torch.cuda.get_device_capability()
            arch = f"sm_{capability[0]}{capability[1]}"
            supported = torch.cuda.get_arch_list()
            if supported and arch not in supported:
                log_debug(
                    f"Architecture GPU {arch} non supportée par cette version de PyTorch; "
                    "exécution sur CPU.",
                    debug,
                    tags=[log_tag],
                )
                return "cpu"
        except Exception:
            # If checking capability fails for any reason, let PyTorch handle it 
            # or assume it's okay (fallback logic usually safer if we assume cpu on error, 
            # but here we proceed if check fails to avoid blocking valid cases).
            pass

    return device


def time_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """Run *func* and return (result, log_dict).

    Convention: if ``debug=...`` is present in kwargs but the function does not accept
    it, we remove it (without hiding unrelated TypeError exceptions).
    """
    debug = bool(kwargs.get("debug", False))
    function_name = getattr(func, "__name__", func.__class__.__name__)

    log_debug(f"Starting function '{function_name}'", debug, tags=["time_call", "start"])
    start_time = time.time()

    # Only pass debug if the function supports it. This avoids catching unrelated TypeErrors.
    call_kwargs = dict(kwargs)
    if "debug" in call_kwargs and not _supports_kwarg(func, "debug"):
        call_kwargs.pop("debug", None)

    try:
        raw_result = func(*args, **call_kwargs)
    except TypeError as e:
        # Backward-compat: some functions may fail when we keep debug.
        if "unexpected keyword argument 'debug'" in str(e) and "debug" in call_kwargs:
            call_kwargs.pop("debug", None)
            raw_result = func(*args, **call_kwargs)
        else:
            raise

    elapsed_time = time.time() - start_time
    log_debug(
        f"Function '{function_name}' completed in {elapsed_time:.4f} seconds",
        debug,
        tags=["time_call", "end"],
    )

    params = _json_dumps_safe({**{str(i): v for i, v in enumerate(args)}, **call_kwargs})
    result = _json_dumps_safe(raw_result)

    log = {
        "function_name": function_name,
        "elapsed_time": elapsed_time,
        "params": params,
        "result": result,
    }
    return raw_result, log