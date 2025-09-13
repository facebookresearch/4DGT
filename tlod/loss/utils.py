# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch


def scale_ls(pred, gt, mask=None, scale_min=0.2, scale_max=5):
    full_shape = pred.shape
    batch_size = pred.shape[0]
    dim_len = len(pred.shape)

    if mask is not None:
        pred = pred * mask
        gt = gt * mask

    pred = pred.reshape(batch_size, -1)
    gt = gt.reshape(batch_size, -1)

    scale = torch.mean(gt * pred, dim=-1) / torch.clamp(
        torch.mean(pred * pred, dim=-1), min=1e-6
    )
    pred = pred.reshape(full_shape)

    for _ in range(0, dim_len - 1):
        scale = scale.unsqueeze(-1)
    scale = torch.clamp(scale, scale_min, scale_max)

    scale = scale.detach()

    pred = pred * scale
    if mask is not None:
        pred = pred * mask + (1 - mask)

    return pred, scale
