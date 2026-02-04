"""
PSTHM package

Historically this package eagerly imported all submodules on `import PSTHM`, which
pulled in heavy dependencies (torch/pyro) even for workflows that only needed
lightweight utilities (e.g., config validation in `--dry-run`).

We now lazily import submodules on attribute access while preserving the
existing public API style: `PSTHM.kernels`, `PSTHM.opti`, etc.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

# Discover submodules without importing them.
__all__ = []
_SUBMODULES: Dict[str, str] = {}
for _loader, _module_name, _is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(_module_name)
    _SUBMODULES[_module_name] = _module_name


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
