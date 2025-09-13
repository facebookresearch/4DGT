# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
import torch

from ..acceleration.checkpoint import auto_grad_checkpoint
from ..easyvolcap.utils.console_utils import logger


log10 = np.log(10)


class render_loss:
    def __init__(self):
        super().__init__()

    def supervision(
        self, preds, gts, loss_weights_dict, loss_func_dict, loss_arr, loss
    ):
        keys = list(gts.keys())
        loss_keys = list(loss_weights_dict.keys())

        for key in keys:
            for loss_key in loss_keys:
                if (
                    isinstance(loss_weights_dict[loss_key], dict)
                    and loss_key in loss_weights_dict
                    and key in loss_weights_dict[loss_key]
                    and loss_weights_dict[loss_key][key] != 0
                    and key in preds
                ):
                    func = loss_func_dict[loss_key]
                    B, T, C, H, W = gts[key].shape
                    loss_val = auto_grad_checkpoint(
                        func,
                        gts[key].reshape(B * T, C, H, W),
                        preds[key].reshape(B * T, C, H, W),
                    )
                    # loss_val = torch.mean(loss_val)
                    loss_arr[f"{key}_{loss_key}"] = loss_val
                    loss = loss + loss_val * loss_weights_dict[loss_key][key]
        return loss

    def regularization(
        self, preds, gts, loss_weights_dict, loss_func_dict, loss_arr, loss
    ):
        for loss_key in loss_weights_dict:
            if (
                not isinstance(loss_weights_dict[loss_key], dict)
                and loss_weights_dict[loss_key] != 0
                and loss_key in loss_func_dict
            ):
                loss_val = loss_func_dict[loss_key](preds, gts)
                # loss_val = torch.mean(loss_val)
                loss_arr[loss_key] = loss_val
                loss = loss + loss_val * loss_weights_dict[loss_key]
        return loss

    def forward_and_backward(
        self,
        preds,
        gts,
        loss_weights_dict,
        loss_func_dict,
        scaler=None,
        run_backward: bool = True,
        run_loss: bool = True,
        clip_psnr: float = 0.0,
    ):
        keys = list(gts.keys())
        # loss_keys = list(loss_func_dict.keys())

        extra_logs = {}

        loss = 0
        loss_arr = {}
        if run_loss:
            loss = self.supervision(
                preds, gts, loss_weights_dict, loss_func_dict, loss_arr, loss
            )
            loss = self.regularization(
                preds, gts, loss_weights_dict, loss_func_dict, loss_arr, loss
            )

        preds_arr = {}

        # Store everything that's output from the network for easier visualization
        for key in preds:
            if key != "gs":
                preds_arr[key] = preds[key]

        loss_key = "mse"
        for key in keys:
            if (
                key in loss_weights_dict[loss_key]
                and loss_weights_dict[loss_key][key] != 0
                and f"{key}_{loss_key}" in loss_arr
            ):
                mse: torch.Tensor = loss_arr[f"{key}_{loss_key}"]
                loss_arr[f"{key}_psnr"] = -10 * mse.log10()

        loss_arr["loss"] = loss

        # Now we'd expect all gpus have the same gradient and same model to update and continue training
        # FIXME: MODEL WOULD DIVERGE DUE TO DIFFERENT GRADIENTS ON DIFFERENT MACHINES
        if "rgb_psnr" in loss_arr:
            rgb_psnr = loss_arr["rgb_psnr"]
            if rgb_psnr < clip_psnr:
                logger.warn(
                    f"PSNR {rgb_psnr} is below the threshold {clip_psnr}, skipping this batch"
                )
                # Will result in no gradient for this rank
                loss = loss * 0 + loss.detach()

        if run_backward:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        return preds_arr, loss_arr, extra_logs
