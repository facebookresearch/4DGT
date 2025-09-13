# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from collections.abc import Iterable

import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", True):  # defaulting to use checkpoints
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


def dummy_function(module, *args, **kwargs):
    return module(*args, **kwargs)
