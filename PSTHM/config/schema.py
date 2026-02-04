"""
Config schema and normalization helpers.

Design goals:
- Keep this module lightweight (no torch/pyro).
- Provide a stable, user-facing YAML shape while allowing legacy keys used by
  existing research pipelines (e.g. `paleo_windows_bp`) to keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping


ConfigDict = Dict[str, Any]


@dataclass(frozen=True)
class TimeAxes:
    paleo_unit: str = "cal_yr_BP"  # "cal_yr_BP" or "CE"
    modern_unit: str = "CE"  # currently only "CE" supported for modern/projections
    reference_year: int = 1950


def _ensure_mapping(x: Any, name: str) -> MutableMapping[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"{name} must be a mapping/dict, got {type(x)}")
    return dict(x)


def normalize_config_dict(cfg: Mapping[str, Any]) -> ConfigDict:
    """
    Normalize a raw config dict:
    - fill missing top-level sections with defaults
    - accept legacy keys (e.g. paleo_windows_bp) and normalize to canonical keys
    - does not resolve relative paths (handled by loader)
    """
    out: ConfigDict = dict(cfg)

    out.setdefault("paths", {})
    out.setdefault("fields", {})
    out.setdefault("time_axes", {})
    out.setdefault("gp_model", {})
    out.setdefault("prediction_grid", {})
    out.setdefault("rate_settings", {})
    out.setdefault("mc_settings", {})
    out.setdefault("exceedance", {})
    out.setdefault("projections", {})
    out.setdefault("output", {})

    # Time axes
    ta = _ensure_mapping(out.get("time_axes"), "time_axes")
    time_axes = TimeAxes(
        paleo_unit=str(ta.get("paleo_unit", "cal_yr_BP")),
        modern_unit=str(ta.get("modern_unit", "CE")),
        reference_year=int(ta.get("reference_year", 1950)),
    )
    out["time_axes"] = {
        "paleo_unit": time_axes.paleo_unit,
        "modern_unit": time_axes.modern_unit,
        "reference_year": time_axes.reference_year,
    }

    # Paleo windows: support both keys; canonical in code is `paleo_windows`
    if "paleo_windows" not in out and "paleo_windows_bp" in out:
        out["paleo_windows"] = out["paleo_windows_bp"]
    out.setdefault("paleo_windows", {})
    # Keep legacy alias so older callers don't break.
    out.setdefault("paleo_windows_bp", out["paleo_windows"])

    # Keep gp_model as a mapping; we intentionally do not support "auto_rules".
    out["gp_model"] = _ensure_mapping(out.get("gp_model"), "gp_model")

    return out
