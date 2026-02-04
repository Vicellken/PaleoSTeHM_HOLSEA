"""
Time axis utilities for PSTHM pipelines.

Lightweight module (no torch/pyro) to avoid heavy imports in dry-run workflows.
"""

from __future__ import annotations

import numpy as np


def bp_to_ce(age_bp, reference_year: int = 1950):
    """Convert cal yr BP (Before Present) to CE."""
    return np.asarray(reference_year) - np.asarray(age_bp)


def ce_to_bp(age_ce, reference_year: int = 1950):
    """Convert CE to cal yr BP (Before Present)."""
    return np.asarray(reference_year) - np.asarray(age_ce)
