"""
Notebook helper utilities for PSTHM workflows.

This module centralizes config-driven setup logic used across notebooks,
including path resolution, output helpers, and kernel builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

import PSTHM
from PSTHM.config.load import load_config


@dataclass
class NotebookContext:
    config_path: Path
    cfg: Dict[str, Any]
    base_dir: Path
    output_dir: Path
    time_unit: str
    reference_year: int
    analysis_cfg: Dict[str, Any]
    max_age_bp: int
    series_start_bp: int
    series_end_bp: int
    series_step_bp: int
    data_csv: Path
    ice7g_npy: Path
    param_store_no_physical: Path
    param_store_single_gia: Path

    @classmethod
    def from_yaml(cls, cfg_path: Path, time_unit: str = "BP") -> "NotebookContext":
        cfg_path = Path(cfg_path)
        loaded = load_config(cfg_path)
        cfg = loaded.config
        base_dir = loaded.base_dir

        def resolve_path(path_value: str) -> Path:
            p = Path(path_value)
            if p.is_absolute():
                return p
            return (base_dir / p).resolve()

        paths = cfg.get("paths", {}) or {}
        output_dir = (
            resolve_path(paths["output_dir"])
            if "output_dir" in paths
            else (base_dir.parent / "Outputs" / cfg_path.stem).resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        data_csv = resolve_path(paths["paleo_rsl"])
        ice7g_npy = resolve_path(paths["gia_ice7g_rsl_npy"])
        param_store_no_physical = resolve_path(paths["trained_param_store_no_physical"])
        param_store_single_gia = resolve_path(paths["trained_param_store_single_gia"])

        for p in (param_store_no_physical, param_store_single_gia):
            p.parent.mkdir(parents=True, exist_ok=True)

        analysis_cfg = cfg.get("analysis", {}) or {}
        series_cfg = analysis_cfg.get("spatial_series_bp", {}) or {}

        return cls(
            config_path=cfg_path,
            cfg=cfg,
            base_dir=base_dir,
            output_dir=output_dir,
            time_unit=time_unit,
            reference_year=int(
                (cfg.get("time_axes", {}) or {}).get("reference_year", 1950)
            ),
            analysis_cfg=analysis_cfg,
            max_age_bp=int(analysis_cfg.get("max_age_bp", 20000)),
            series_start_bp=int(series_cfg.get("start", 0)),
            series_end_bp=int(series_cfg.get("end", 20000)),
            series_step_bp=int(series_cfg.get("step", 1000)),
            data_csv=data_csv,
            ice7g_npy=ice7g_npy,
            param_store_no_physical=param_store_no_physical,
            param_store_single_gia=param_store_single_gia,
        )

    def cfg_path(self, key: str) -> Path:
        """Resolve cfg['paths'][key] relative to the YAML location."""
        p = Path(self.cfg["paths"][key])
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def save_figure(self, fig, name: str, dpi: int = 300) -> Path:
        if fig is None:
            fig = plt.gcf()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def needs_training(self, train_flag: bool, path: Path, label: str) -> bool:
        if train_flag:
            return True
        if path.exists():
            return False
        print(f"{label} ParamStore not found at {path}. Training to create it.")
        return True

    def _run_cfg(self, run_key: str) -> Dict[str, Any]:
        return (self.cfg.get("runs", {}) or {}).get(run_key, {}) or {}

    def hp(self, run_key: str, k: str, default=None):
        run = self._run_cfg(run_key)
        gp = self.cfg.get("gp_model", {}) or {}
        if k in run:
            return run[k]
        if k in gp:
            return gp[k]
        return default

    def pred_time_axis(self, grid_key: str = "paleo_time") -> np.ndarray:
        grids = self.cfg.get("prediction_grid", {}) or {}
        t = grids.get(grid_key, {}) or {}
        start, end, step = t.get("start"), t.get("end"), t.get("step")
        if start is None or end is None or step is None:
            raise ValueError(
                f"prediction_grid.{grid_key}.{{start,end,step}} must be set"
            )
        axis = np.arange(start, end, step)
        if self.time_unit == "BP":
            axis = PSTHM.time.ce_to_bp(axis, reference_year=self.reference_year)
            axis = axis[axis >= 0]
        return axis

    @staticmethod
    def clear_then_load_param_store(path: Path) -> None:
        pyro.clear_param_store()
        pyro.get_param_store().load(str(path))

    @staticmethod
    def save_param_store(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pyro.get_param_store().save(str(path))

    @staticmethod
    def _uniform(lo, hi):
        return dist.Uniform(torch.tensor(float(lo)), torch.tensor(float(hi)))

    def build_no_physical_kernel(self):
        """Kernel stack: global + regional_nonlinear + local_nonlinear + whitenoise."""
        kc = (self.cfg.get("gp_model", {}) or {}).get("kernels", {}) or {}

        global_cfg = kc.get("global_temporal", {}) or {}
        global_kernel = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        global_kernel.set_prior(
            "lengthscale",
            self._uniform(*global_cfg.get("lengthscale_prior", [100.0, 20000.0])),
        )
        global_kernel.set_prior(
            "variance",
            self._uniform(*global_cfg.get("variance_prior", [0.01**2, 1000.0])),
        )

        rnl = kc.get("regional_nonlinear", {}) or {}
        rnl_tem = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        rnl_tem.set_prior(
            "lengthscale",
            self._uniform(*rnl.get("temporal_lengthscale_prior", [500.0, 5000.0])),
        )
        rnl_tem.set_prior(
            "variance",
            self._uniform(*rnl.get("temporal_variance_prior", [0.05, 100.0])),
        )
        rnl_sp = PSTHM.kernels.Matern21(input_dim=1, geo=True)
        rnl_sp.set_prior(
            "s_lengthscale",
            self._uniform(*rnl.get("spatial_lengthscale_prior", [0.05, 0.25])),
        )
        rnl_kernel = PSTHM.kernels.Product(rnl_tem, rnl_sp)

        lnl = kc.get("local_nonlinear", {}) or {}
        lnl_tem = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        lnl_tem.set_prior(
            "lengthscale",
            self._uniform(*lnl.get("temporal_lengthscale_prior", [100.0, 2000.0])),
        )
        lnl_tem.set_prior(
            "variance",
            self._uniform(*lnl.get("temporal_variance_prior", [0.01, 10.0])),
        )
        lnl_sp = PSTHM.kernels.Matern21(input_dim=1, geo=True)
        lnl_sp.set_prior(
            "s_lengthscale",
            self._uniform(*lnl.get("spatial_lengthscale_prior", [0.01, 0.05])),
        )
        lnl_kernel = PSTHM.kernels.Product(lnl_tem, lnl_sp)

        wn = kc.get("whitenoise", {}) or {}
        wn_kernel = PSTHM.kernels.WhiteNoise(input_dim=1)
        wn_kernel.set_prior(
            "variance",
            self._uniform(*wn.get("variance_prior", [0.01**2, 100.0])),
        )

        combined = PSTHM.kernels.Sum(rnl_kernel, lnl_kernel)
        combined = PSTHM.kernels.Sum(combined, wn_kernel)
        combined = PSTHM.kernels.Sum(combined, global_kernel)
        return combined

    def build_residual_kernel(self):
        """Kernel for single-GIA residual modeling: regional_nonlinear + local_nonlinear + whitenoise."""
        kc = (self.cfg.get("gp_model", {}) or {}).get("kernels", {}) or {}

        rnl = kc.get("regional_nonlinear", {}) or {}
        rnl_tem = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        rnl_tem.set_prior(
            "lengthscale",
            self._uniform(*rnl.get("temporal_lengthscale_prior", [500.0, 5000.0])),
        )
        rnl_tem.set_prior(
            "variance",
            self._uniform(*rnl.get("temporal_variance_prior", [0.05, 100.0])),
        )
        rnl_sp = PSTHM.kernels.Matern21(input_dim=1, geo=True)
        rnl_sp.set_prior(
            "s_lengthscale",
            self._uniform(*rnl.get("spatial_lengthscale_prior", [0.05, 0.25])),
        )
        rnl_kernel = PSTHM.kernels.Product(rnl_tem, rnl_sp)

        lnl = kc.get("local_nonlinear", {}) or {}
        lnl_tem = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        lnl_tem.set_prior(
            "lengthscale",
            self._uniform(*lnl.get("temporal_lengthscale_prior", [100.0, 2000.0])),
        )
        lnl_tem.set_prior(
            "variance",
            self._uniform(*lnl.get("temporal_variance_prior", [0.01, 10.0])),
        )
        lnl_sp = PSTHM.kernels.Matern21(input_dim=1, geo=True)
        lnl_sp.set_prior(
            "s_lengthscale",
            self._uniform(*lnl.get("spatial_lengthscale_prior", [0.01, 0.05])),
        )
        lnl_kernel = PSTHM.kernels.Product(lnl_tem, lnl_sp)

        wn = kc.get("whitenoise", {}) or {}
        wn_kernel = PSTHM.kernels.WhiteNoise(input_dim=1)
        wn_kernel.set_prior(
            "variance",
            self._uniform(*wn.get("variance_prior", [0.01**2, 100.0])),
        )

        combined = PSTHM.kernels.Sum(rnl_kernel, lnl_kernel)
        combined = PSTHM.kernels.Sum(combined, wn_kernel)
        return combined

    def build_local_only_kernel(self):
        """Kernel for multi-GIA mean-function modeling: local_nonlinear + whitenoise."""
        kc = (self.cfg.get("gp_model", {}) or {}).get("kernels", {}) or {}

        lnl = (
            kc.get("multi_gia_local_nonlinear", {})
            or kc.get("local_nonlinear", {})
            or {}
        )
        lnl_tem = PSTHM.kernels.Matern32(input_dim=1, geo=False)
        lnl_tem.set_prior(
            "variance",
            self._uniform(*lnl.get("temporal_variance_prior", [0.5**2, 30.0**2])),
        )
        lnl_tem.set_prior(
            "lengthscale",
            self._uniform(*lnl.get("temporal_lengthscale_prior", [100.0, 20000.0])),
        )
        lnl_sp = PSTHM.kernels.Matern21(input_dim=1, geo=True)
        lnl_sp.set_prior(
            "s_lengthscale",
            self._uniform(*lnl.get("spatial_lengthscale_prior", [0.05, 0.3])),
        )
        lnl_kernel = PSTHM.kernels.Product(lnl_tem, lnl_sp)

        wn = kc.get("whitenoise", {}) or {}
        wn_kernel = PSTHM.kernels.WhiteNoise(input_dim=1)
        wn_kernel.set_prior(
            "variance",
            self._uniform(*wn.get("variance_prior", [0.01**2, 100.0])),
        )

        combined = PSTHM.kernels.Sum(lnl_kernel, wn_kernel)
        return combined

    def export_optimized_params(
        self, gpr, name: str, include_global: bool = True
    ) -> Path:
        if include_global:
            regional_nl = gpr.kernel.kern0.kern0.kern0
            local_nl = gpr.kernel.kern0.kern0.kern1
            whitenoise = gpr.kernel.kern0.kern1
            global_kernel = gpr.kernel.kern1
        else:
            regional_nl = gpr.kernel.kern0.kern0
            local_nl = gpr.kernel.kern0.kern1
            whitenoise = gpr.kernel.kern1
            global_kernel = None

        optimized_params = {}
        if include_global and global_kernel is not None:
            optimized_params["global_kernel_lengthscale (yr)"] = (
                global_kernel.lengthscale.item()
            )
            optimized_params["global_kernel_variance (m^2)"] = (
                global_kernel.variance.item()
            )

        optimized_params["regional_nl_kernel_variance (m^2)"] = (
            regional_nl.kern0.variance.item()
        )
        optimized_params["regional_nl_kernel_lengthscale (yr)"] = (
            regional_nl.kern0.lengthscale.item()
        )
        optimized_params["regional_nl_kernel_spatial_lengthscale (km)"] = (
            regional_nl.kern1.s_lengthscale.item() * 6471
        )
        optimized_params["local_nl_kernel_variance (m^2)"] = (
            local_nl.kern0.variance.item()
        )
        optimized_params["local_nl_kernel_lengthscale (yr)"] = (
            local_nl.kern0.lengthscale.item()
        )
        optimized_params["local_nl_kernel_spatial_lengthscale (km)"] = (
            local_nl.kern1.s_lengthscale.item() * 6471
        )
        optimized_params["whitenoise_kernel_variance (m^2)"] = (
            whitenoise.variance.item()
        )

        df_optimized_params = pd.DataFrame(
            list(optimized_params.items()), columns=["Kernel", "Optimized Value"]
        )
        print("Optimized parameters:", df_optimized_params)
        path = self.save_dataframe(df_optimized_params, f"optimized_params_{name}")
        print(f"Optimized parameters have been saved to {path}")
        return path
