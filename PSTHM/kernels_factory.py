"""
Canonical kernel factory for PSTHM spatio-temporal GP models.

This is the single source of truth for constructing the multi-kernel STGP stack
used across:
- Holocene UI scripts
- Research pipelines (e.g. exceedance timing)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import pyro.distributions as dist

from . import kernels


def build_spatiotemporal_kernel(gp_model_cfg: Dict) -> Any:
    """
    Build the spatio-temporal kernel stack following the Holocene UI pattern.

    Expected gp_model_cfg shape (subset):
      gp_model:
        kernels:
          global_temporal: {type, lengthscale_prior: [lo,hi], variance_prior: [lo,hi]}
          regional_linear: {temporal_variance_prior: [lo,hi], spatial_lengthscale_prior: [lo,hi]}
          regional_nonlinear: {temporal_variance_prior, temporal_lengthscale_prior, spatial_lengthscale_prior}
          local_nonlinear: {temporal_variance_prior, temporal_lengthscale_prior, spatial_lengthscale_prior}
          sp_whitenoise: {variance_prior: [lo,hi]}
          whitenoise: {variance_prior: [lo,hi]}
    """
    kernel_cfg = gp_model_cfg.get("kernels", {})

    # Global temporal kernel
    global_cfg = kernel_cfg.get("global_temporal", {})
    global_kernel = kernels.Matern32(input_dim=1, geo=False)
    ls_prior = global_cfg.get("lengthscale_prior", [100.0, 20000.0])
    var_prior = global_cfg.get("variance_prior", [1e-5, 1000.0])
    global_kernel.set_prior(
        "lengthscale",
        dist.Uniform(torch.tensor(ls_prior[0]), torch.tensor(ls_prior[1])),
    )
    global_kernel.set_prior(
        "variance", dist.Uniform(torch.tensor(var_prior[0]), torch.tensor(var_prior[1]))
    )

    # Regional linear
    rlin_cfg = kernel_cfg.get("regional_linear", {})
    rlin_tem = kernels.Linear(input_dim=1)
    rlin_tem.set_prior(
        "variance",
        dist.Uniform(
            torch.tensor(rlin_cfg.get("temporal_variance_prior", [1e-12, 1e-6])[0]),
            torch.tensor(rlin_cfg.get("temporal_variance_prior", [1e-12, 1e-6])[1]),
        ),
    )
    rlin_tem.ref_year = torch.tensor(0.0)
    rlin_sp = kernels.Matern21(input_dim=1, geo=True)
    rlin_sls = rlin_cfg.get("spatial_lengthscale_prior", [0.01, 0.5])
    rlin_sp.set_prior(
        "s_lengthscale",
        dist.Uniform(torch.tensor(rlin_sls[0]), torch.tensor(rlin_sls[1])),
    )
    regional_linear_kernel = kernels.Product(rlin_tem, rlin_sp)

    # Regional nonlinear
    rnl_cfg = kernel_cfg.get("regional_nonlinear", {})
    rnl_tem = kernels.Matern32(
        input_dim=1, lengthscale=global_kernel.lengthscale, geo=False
    )
    rnl_tem.set_prior(
        "variance",
        dist.Uniform(
            torch.tensor(rnl_cfg.get("temporal_variance_prior", [1e-6, 100.0])[0]),
            torch.tensor(rnl_cfg.get("temporal_variance_prior", [1e-6, 100.0])[1]),
        ),
    )
    rnl_tem.set_prior(
        "lengthscale",
        dist.Uniform(
            torch.tensor(
                rnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[0]
            ),
            torch.tensor(
                rnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[1]
            ),
        ),
    )
    rnl_sp = kernels.Matern21(input_dim=1, geo=True)
    rnl_sls = rnl_cfg.get("spatial_lengthscale_prior", [0.01, 0.5])
    rnl_sp.set_prior(
        "s_lengthscale",
        dist.Uniform(torch.tensor(rnl_sls[0]), torch.tensor(rnl_sls[1])),
    )
    regional_nl_kernel = kernels.Product(rnl_tem, rnl_sp)

    # Local nonlinear
    lnl_cfg = kernel_cfg.get("local_nonlinear", {})
    lnl_tem = kernels.Matern32(
        input_dim=1, lengthscale=global_kernel.lengthscale, geo=False
    )
    lnl_tem.set_prior(
        "variance",
        dist.Uniform(
            torch.tensor(lnl_cfg.get("temporal_variance_prior", [1e-6, 1.0])[0]),
            torch.tensor(lnl_cfg.get("temporal_variance_prior", [1e-6, 1.0])[1]),
        ),
    )
    lnl_tem.set_prior(
        "lengthscale",
        dist.Uniform(
            torch.tensor(
                lnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[0]
            ),
            torch.tensor(
                lnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[1]
            ),
        ),
    )
    lnl_sp = kernels.Matern21(input_dim=1, geo=True)
    lnl_sls = lnl_cfg.get("spatial_lengthscale_prior", [0.001, 0.01])
    lnl_sp.set_prior(
        "s_lengthscale",
        dist.Uniform(torch.tensor(lnl_sls[0]), torch.tensor(lnl_sls[1])),
    )
    local_nl_kernel = kernels.Product(lnl_tem, lnl_sp)

    # Site-specific datum correction
    spwn_cfg = kernel_cfg.get("sp_whitenoise", {})
    sp_whitenoise_kernel = kernels.WhiteNoise_SP(input_dim=1, sp=False, geo=True)
    spwn_var = spwn_cfg.get("variance_prior", [1e-7, 1.0])
    sp_whitenoise_kernel.set_prior(
        "variance", dist.Uniform(torch.tensor(spwn_var[0]), torch.tensor(spwn_var[1]))
    )

    # White noise
    wn_cfg = kernel_cfg.get("whitenoise", {})
    whitenoise_kernel = kernels.WhiteNoise(input_dim=1)
    wn_var = wn_cfg.get("variance_prior", [1e-7, 1.0])
    whitenoise_kernel.set_prior(
        "variance", dist.Uniform(torch.tensor(wn_var[0]), torch.tensor(wn_var[1]))
    )

    combined = kernels.Sum(global_kernel, regional_linear_kernel)
    combined = kernels.Sum(combined, regional_nl_kernel)
    combined = kernels.Sum(combined, local_nl_kernel)
    combined = kernels.Sum(combined, sp_whitenoise_kernel)
    combined = kernels.Sum(combined, whitenoise_kernel)
    return combined
