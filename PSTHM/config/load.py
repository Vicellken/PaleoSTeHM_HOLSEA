"""
YAML config loading + normalization.

This module is kept lightweight so it can be used without torch/pyro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .schema import normalize_config_dict, ConfigDict


@dataclass(frozen=True)
class LoadedConfig:
    config: ConfigDict
    config_path: Path
    base_dir: Path


def _resolve_path(p: str, base_dir: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((base_dir / pp).resolve())


def load_config_dict(
    cfg: Mapping[str, Any], *, base_dir: Optional[str | Path] = None
) -> LoadedConfig:
    """
    Normalize a config dict (already in-memory). Useful for notebooks.
    """
    bd = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    norm = normalize_config_dict(cfg)
    # Resolve known paths in-place.
    paths = dict(norm.get("paths", {}))
    for k in (
        "paleo_rsl",
        "field_mapping",
        "modern_observed_rate",
        "projected_rate",
        "output_dir",
    ):
        if k in paths and paths[k] is not None:
            paths[k] = _resolve_path(str(paths[k]), bd)
    norm["paths"] = paths
    return LoadedConfig(config=norm, config_path=bd / "<in_memory>", base_dir=bd)


def load_config(config_path: str | Path) -> LoadedConfig:
    """
    Load YAML config and normalize it.

    Returns a LoadedConfig with:
    - config: normalized dict with defaults + legacy key support
    - base_dir: directory of config file (used for resolving relative paths)
    """
    cp = Path(config_path).expanduser().resolve()
    base_dir = cp.parent
    with cp.open("r") as f:
        raw = yaml.safe_load(f) or {}
    norm = normalize_config_dict(raw)

    # Resolve known paths in-place.
    paths = dict(norm.get("paths", {}))
    for k in (
        "paleo_rsl",
        "field_mapping",
        "modern_observed_rate",
        "projected_rate",
        "output_dir",
    ):
        if k in paths and paths[k] is not None:
            paths[k] = _resolve_path(str(paths[k]), base_dir)
    norm["paths"] = paths

    return LoadedConfig(config=norm, config_path=cp, base_dir=base_dir)
