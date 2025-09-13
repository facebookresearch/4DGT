# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch


def cov_t_to_dt(cov_t: torch.Tensor, marginal_th: float = 0.05):
    is_tensor = isinstance(cov_t, torch.Tensor)
    if not is_tensor:
        cov_t = torch.as_tensor(cov_t)
    dt = (
        torch.log(torch.as_tensor(marginal_th).to(cov_t, non_blocking=True))
        / -0.5
        * cov_t
    ).sqrt()
    if not is_tensor:
        dt = dt.item()
    return dt


def dt_to_cov_t(dt: torch.Tensor, marginal_th: float = 0.05):
    is_tensor = isinstance(dt, torch.Tensor)
    if not is_tensor:
        dt = torch.as_tensor(dt)
    cov_t = (dt**2) / (
        torch.log(torch.as_tensor(marginal_th).to(dt, non_blocking=True)) / -0.5
    )
    if not is_tensor:
        cov_t = cov_t.item()
    return cov_t


def radius_to_sigma(radius, cutoff: float = 0.05):
    # Calculate sigma
    is_tensor = isinstance(radius, torch.Tensor)
    if not is_tensor:
        radius = torch.as_tensor(radius)
    sigma = radius / torch.sqrt(
        -2 * torch.log(torch.as_tensor(cutoff).to(radius, non_blocking=True))
    )
    if not is_tensor:
        sigma = sigma.item()
    return sigma


def sigma_to_radius(sigma, cutoff: float = 0.05):
    # Calculate sigma
    is_tensor = isinstance(sigma, torch.Tensor)
    if not is_tensor:
        sigma = torch.as_tensor(sigma)
    radius = sigma * torch.sqrt(
        -2 * torch.log(torch.as_tensor(cutoff).to(sigma, non_blocking=True))
    )
    if not is_tensor:
        radius = radius.item()
    return radius



