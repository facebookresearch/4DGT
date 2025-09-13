# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import builtins as __builtin__
import os
import random
import socket

import numpy as np

import torch
import torch.distributed as dist

from ..easyvolcap.utils.console_utils import log
from ..easyvolcap.utils.net_utils import setup_deterministic

from .dist_helper import get_rank, get_world_size, init_process_group

print_debug_info = False
host_name = socket.gethostname()


is_master = True
builtin_print = __builtin__.print


def patched_print(*args, **kwargs):
    force = kwargs.pop("force", False)
    if is_master or force:
        builtin_print(*args, **kwargs)


__builtin__.print = patched_print


def setup_for_distributed(is_master_local):
    """
    This function disables printing when not in master process
    """
    global is_master
    is_master = is_master_local


def _parse_slurm_node_list(s: str):

    seps = []
    unclosed = False
    # Just use plain and simple string iteration
    # No fancy regex or even state machine required
    for i, c in enumerate(s):
        if c == "[":
            unclosed = True
        elif c == "]":
            unclosed = False
        elif c == ",":
            if not unclosed:
                seps.append(i)
        else:
            pass
    sections = []
    if not len(seps):
        sections.append(s)
    else:
        for i in range(len(seps)):
            if i == 0:
                sections.append(s[: seps[i]])
            else:
                sections.append(s[seps[i - 1] + 1 : seps[i]])
        sections.append(s[seps[-1] + 1 :])
    # Now sections should contain all the separated non-continuous nodes
    nodes = []
    for section in sections:
        if "[" in section and "]" in section:
            prefix = section[: section.find("[")]
            suffix = section[len(prefix) + 1 : -1]
            subs = suffix.split(",")
            for sub in subs:
                if "-" in sub:
                    start, end = sub.split("-")
                    for i in range(int(start), int(end) + 1):
                        nodes.append(prefix + str(i))
                else:
                    nodes.append(prefix + sub)
        else:
            nodes.append(section)
    return nodes


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


class _TorchDistributedEnvironment:
    def __init__(self, interactive_session: bool = False):
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = os.environ.get("MASTER_PORT", "0")
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

        if interactive_session:
            self._set_from_torch_env()
        else:
            self._set_from_slurm_env()

    def _set_from_torch_env(self):
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_world_size = int(os.environ["WORLD_SIZE"])

    def _set_from_slurm_env(self):
        if "SLURM_JOB_ID" in os.environ:
            # logger.info("Initialization from Slurm environment")
            job_id = int(os.environ["SLURM_JOB_ID"])
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(
                os.environ["SLURM_NTASKS"]
            )  # SLURM_GPUS #int(os.environ["WORLD_SIZE"])
            self.local_world_size = self.world_size
            assert self.rank < self.world_size
            self.local_rank = int(os.environ["SLURM_LOCALID"])

            if "SLURM_JOB_NUM_NODES" in os.environ:
                node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
                try:
                    nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
                except Exception as e:  # noqa: F841
                    nodes = ["localhost"]
                # assert len(nodes) == node_count
                self.master_addr = nodes[0]
                self.master_port = _get_master_port(seed=job_id)
                self.local_world_size = self.world_size // node_count

            assert self.local_rank < self.local_world_size

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        # See the "Environment variable initialization" section from
        # https://pytorch.org/docs/stable/distributed.html for the complete list of
        # environment variables required for the env:// initialization method.
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                assert k in os.environ, "{} is missing in environ"
                assert (
                    v == os.environ[k]
                ), "Environment variables inconsistent {} != {}".format(
                    os.environ[k], v
                )

        os.environ.update(env_vars)
        return self


def init_distributed_mode(args):
    setup_deterministic(False, True, False, True, args.seed)
    torch_env = _TorchDistributedEnvironment(args.interactive_session)
    torch_env.export(overwrite=True)

    args.gpu = torch_env.local_rank
    args.global_rank = torch_env.rank
    args.world_size = torch_env.world_size

    if args.enable_ddp:
        log(
            f"Initialize ddp. rank:{args.global_rank}, world size:{args.world_size}, local rank:{args.gpu}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.global_rank,
        )

        torch.cuda.set_device(args.gpu)
        dist.barrier()
        setup_for_distributed(args.global_rank == 0)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
