# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def init_process_group(**kwargs):
    dist.init_process_group(**kwargs)


def get_parallel_model(model, device):
    if get_world_size() >= 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            # , find_unused_parameters=True
        )
        model._set_static_graph()
    else:
        raise NotImplementedError
    return model


def synchronize() -> None:
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
