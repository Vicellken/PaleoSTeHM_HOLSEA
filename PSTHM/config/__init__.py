"""
Configuration utilities for PSTHM.

This subpackage is intentionally lightweight (no torch/pyro imports) so that
configuration validation can be used in "dry run" workflows without requiring
heavy numerical dependencies.

Note: We intentionally do not support automatic tuning of spatial kernel priors.
Default kernel priors are chosen to be physically interpretable (local vs regional
spatial correlation scales).
"""

from .load import load_config, load_config_dict

__all__ = ["load_config", "load_config_dict"]
