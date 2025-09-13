# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

# fmt: off
import builtins
import copy  # noqa: F401
import datetime
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
import re
import sys
import tempfile
import time
from collections import defaultdict, deque, OrderedDict

from typing import List

import cv2

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torchvision
import trimesh
from plyfile import PlyData, PlyElement
from skimage import measure
from torch.utils.data import DataLoader

from ..data_loader.utils import compute_rays

# from ..easyvolcap.utils.console_utils import *
from ..easyvolcap.utils.console_utils import blue, dirname, logger, yellow

from .dist_helper import get_rank, get_world_size, is_dist_avail_and_initialized

from .io_helper import pathmgr

print_debug_info = False
# fmt: on


def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    random.seed(worker_id)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class RepeatedDataLoader(DataLoader):
    def __init__(self, *args, worker_init_fn=worker_init_fn, **kwargs):
        super().__init__(*args, worker_init_fn=worker_init_fn, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def linear_to_srgb(l):  # noqa: E741
    # s = np.zeros_like(l)
    s = torch.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def _suppress_print(gpu=None):
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    if (gpu is not None and gpu != 0) or (gpu is None and not is_main_process()):
        builtins.print = print_pass


def pytorch_mlp_clip_gradients(model, clip):
    grad_norm = []
    for p in model.parameters():
        if p is not None and p.grad is not None:
            grad_norm.append(p.grad.view(-1))

    if len(grad_norm) > 0:
        grad_norm = torch.concat(grad_norm).norm(2).item()
        clip_coef = clip / (grad_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return grad_norm
    return None


def unitwise_norm(x):
    """Computes norms of each output unit separately, assuming (HW)IO weights."""
    if len(x.shape) <= 1:  # Scalars and vectors
        return x.norm(2)
    elif len(x.shape) in [2, 3]:  # Linear layers of shape OI
        return x.norm(2, dim=-1, keepdim=True)
    elif len(x.shape) == 4:  # Conv kernels of shape OIHW
        return x.norm(2, dim=[1, 2, 3], keepdim=True)
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 3, 4]! {x}")


def clip_gradients(model, clip, check_nan_inf=True, file_name=None, adaptive=False):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            if check_nan_inf:
                p.grad.data = torch.nan_to_num(
                    p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )

            if adaptive:
                param_norm = unitwise_norm(p)
                grad_norm = unitwise_norm(p.grad.data)
                max_norm = param_norm * clip
                trigger = grad_norm > max_norm
                clipped_grad = p.grad.data * (max_norm / grad_norm.clamp(min=1e-6))
                p.grad.data = torch.where(trigger, clipped_grad, p.grad.data)
            else:
                grad_norm = p.grad.data.norm(2)
                norms.append(grad_norm.item())
                clip_coef = clip / (grad_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def filter_weights_with_wrong_size(model, weights, reinit=()):
    new_weights = OrderedDict()
    missing_keys = []
    state_dict = model.state_dict()
    for name, value in weights.items():
        if name in state_dict:
            target_value = state_dict[name]

            should_reinit = False
            for init in reinit:
                if init in name:
                    should_reinit = True

            if value.size() != target_value.size() or should_reinit:
                missing_keys.append(name)
            else:
                new_weights[name] = value
        else:
            new_weights[name] = value

    return new_weights, missing_keys


def load_ddp_state_dict(  # noqa: C901
    model,
    weights,
    key=None,
    filter_mismatch=True,
    rewrite_weights=(),
    reinit=(),
    clone=(),
):
    if weights is None:
        msg = "No weights available for module {}".format(key)
        return msg

    if len(clone):
        for pair in clone:
            s, t = pair
            # Only clone those that the source exist and target don't
            all_keys = [
                k for k in weights.keys() if s in k and k.replace(s, t) not in weights
            ]
            for k in all_keys:
                new_k = k.replace(s, t)
                weights[new_k] = weights[k].clone()

    # Not very efficient, but this is not often used
    try:
        if len(rewrite_weights) > 0:
            for key in weights.keys():
                for rewrite_spec in rewrite_weights:
                    if rewrite_spec["pname"] in key:
                        weights[key] = rewrite_spec["rewrite_fn"](weights[key])
                        print(
                            " [OuO] Rewrite weight {} into {}".format(
                                key, weights[key].shape
                            )
                        )
                        assert (
                            model.get_parameter(key).shape == weights[key].shape
                        ), "The parameter shape mismatched after rewrite! {} != {}".format(
                            model.get_parameter(key).shape, weights[key].shape
                        )
    except Exception as e:
        print(e)
        # import pdb

        # pdb.set_trace()
        raise e

    if isinstance(model, nn.parallel.DistributedDataParallel):
        if filter_mismatch:
            weights, missing_keys = filter_weights_with_wrong_size(
                model, weights, reinit
            )
            if len(missing_keys) > 0:
                logger.warn(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch or explicit specification of user.",
                )
        msg = model.load_state_dict(weights, strict=False)
    elif isinstance(model, torch.optim.Optimizer):
        msg = model.load_state_dict(weights)
    else:
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k[:7] == "module.":
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            else:
                name = k
            new_weights[name] = v
        if filter_mismatch:
            new_weights, missing_keys = filter_weights_with_wrong_size(
                model, new_weights, reinit
            )
            if len(missing_keys) > 0:
                logger.warn(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch or explicit specification of user.",
                )
        msg = model.load_state_dict(new_weights, strict=False)
    return msg


def restart_from_checkpoint(
    ckp_path,
    run_variables=None,
    load_weights_only=False,
    rewrite_weights=(),
    reinit=(),
    clone=(),
    **kwargs,
):
    """
    Re-start from checkpoint
    """
    if not pathmgr.exists(ckp_path):
        raise ValueError(f"Checkpoint not found at {ckp_path}")
    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    if get_world_size() == 1:
        ckp_path = pathmgr.get_local_path(ckp_path)
    with pathmgr.open(ckp_path, "rb") as fb:
        # Set weights_only=False to load checkpoints with numpy arrays
        # This is safe since we're loading our own trained checkpoints
        checkpoint = torch.load(fb, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in checkpoint and value is not None:
            try:
                msg = load_ddp_state_dict(
                    value,
                    checkpoint[key],
                    key=key,
                    rewrite_weights=rewrite_weights,
                    reinit=reinit,
                    clone=clone,
                )
                logger.info(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError as e:
                logger.info(
                    "=> failed to load '{}' from checkpoint: '{}', with error '{}'".format(
                        key, ckp_path, e
                    )
                )
        else:
            logger.warn(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if not load_weights_only and run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    logger.info(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            logger.warn(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    if len(schedule) != epochs * niter_per_ep:
        logger.warn(
            f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
        )
    return schedule


def linear_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    logger.info(
        f"cosin scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.linspace(base_value, final_value, len(iters))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert (
        len(schedule) == epochs * niter_per_ep
    ), f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    return schedule


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.8f} ({avg:.8f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None, tb_writer=None, epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger
        self.tb_writer = tb_writer
        self.epoch = epoch
        assert self.logger is not None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def log_distr(self, distrs, global_step):
        assert self.tb_writer, "Cannot log distr without Tensorboard"
        for group_key, distrs_group in distrs.items():
            for inst_key, inst_values in distrs_group.items():
                if isinstance(inst_values, torch.Tensor) and (inst_values.numel() > 1):
                    self.tb_writer.add_histogram(
                        f"distr-{group_key}/{inst_key}",
                        inst_values,
                        global_step=global_step,
                        bins="tensorflow",
                    )
                else:  # Welp, some stats of distribution is a scalar
                    self.tb_writer.add_scalar(
                        f"stats-{group_key}/{inst_key}",
                        inst_values,
                        global_step=global_step,
                        new_style=True,
                    )

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, start=0):
        if hasattr(iterable, "dataset") and hasattr(
            iterable.dataset, "batch_image_num"
        ):
            iterable_len = int(
                len(iterable) / iterable.dataset.batch_image_num + 0.5
            )  # normalize data loader length for better recording

        i = start
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}", window_size=1)
        data_time = SmoothedValue(fmt="{avg:.6f}", window_size=1)
        space_fmt = ":" + str(len(str(iterable_len))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                    "cpu mem: {cpu_memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == iterable_len - 1:
                eta_seconds = iter_time.global_avg * (iterable_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                global_step = self.epoch * iterable_len + i
                gpu_memory = torch.cuda.max_memory_allocated() / MB
                cpu_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
                if self.tb_writer:
                    for k in self.meters:
                        self.tb_writer.add_scalar(
                            f"train/{k}",
                            self.meters[k].value,
                            global_step=global_step,
                            new_style=True,
                        )
                    for k in ["iter_time", "data_time", "gpu_memory", "cpu_memory"]:
                        self.tb_writer.add_scalar(
                            f"train/{k}",
                            locals()[k].avg
                            if isinstance(locals()[k], SmoothedValue)
                            else locals()[k],
                            global_step=global_step,
                            new_style=True,
                        )
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            iterable_len,
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                            memory=gpu_memory,
                            cpu_memory=cpu_memory,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            iterable_len,
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                        )
                    )
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / iterable_len
            ),
        )


def is_main_process():
    return get_rank() == 0


def save_on_master(ckpt, model_path, backup_ckp_epoch=-1, topk=-1, max_to_backup=2):
    if not is_main_process():
        return

    logger.info(yellow(f"Saving checkpoint to: {blue(model_path)}"))

    basedir = dirname(model_path)
    if not pathmgr.isdir(basedir):
        pathmgr.mkdirs(basedir)
    with pathmgr.open(model_path, "wb") as fp:
        torch.save(ckpt, fp)

    if backup_ckp_epoch > 0:
        target_path = os.path.join(basedir, f"ckpt_{backup_ckp_epoch}.pth")
        pathmgr.copy(model_path, target_path, overwrite=True)

    if max_to_backup > 0:
        backup_ckps = pathmgr.ls(basedir)
        backup_ckps = [p for p in backup_ckps if "ckpt_" in p and ".pth" in p]
        if len(backup_ckps) > max_to_backup:
            backup_ckps = sorted(
                backup_ckps, key=lambda p: int(p.split("ckpt_")[1].split(".pth")[0])
            )
            ckps_to_remove = backup_ckps[:-max_to_backup]
            for ckp_name in ckps_to_remove:
                ckp_path = os.path.join(basedir, ckp_name)
                pathmgr.rm(ckp_path)

    return


def save_image(image, name, is_gamma=False):
    if len(image.shape) == 5:
        batch_size, im_num, _, h, w = image.shape
        image = image.reshape((batch_size * im_num, -1, h, w))
        nrow = im_num
    else:
        batch_size = image.shape[0]
        nrow = batch_size

    if "mask" not in name.split("/")[-1] and "env" not in name.split("/")[-1]:
        image = 0.5 * (image + 1)

    if is_gamma:
        image = linear_to_srgb(image)

    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(
            image[:, :3], fp, padding=0, format=name.split(".")[-1], nrow=nrow
        )  # only supports 3 channels at most when saving images


def save_single_png(images, name, image_ids=None, is_gamma=False):
    if "mask" not in name:
        images = 0.5 * (images + 1)
    if is_gamma:
        images = linear_to_srgb(images)
    images = images.detach().cpu().numpy()
    images = (255 * np.clip(images, 0, 1)).astype(np.uint8)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)

        buffer = cv2.imencode(".png", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)


def save_single_exr(images, name, image_ids=None, is_gamma=False):
    images = images.detach().cpu().float().numpy()
    images = images.astype(np.float32)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)
        buffer = cv2.imencode(".exr", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)


def save_depth(depth, name, depth_min=1.5, depth_max=2.5):
    batch_size, im_num, _, h, w = depth.shape
    depth = np.clip(depth, depth_min, depth_max)

    cmap = cm.get_cmap("jet")
    depth = (depth.reshape(-1) - depth_min) / (depth_max - depth_min)
    depth = depth.detach().cpu().numpy()
    colors = cmap(depth.flatten())[:, :3]
    colors = colors.reshape(batch_size * im_num, h, w, 3)
    colors = colors.transpose(0, 3, 1, 2)
    colors = torch.from_numpy(colors)
    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(colors, fp, nrow=im_num)


def save_ply(  # noqa: C901
    path, xyz, rgb=None, opacity=None, scale=None, rotation=None, mode="splat"
):
    assert mode in {"splat", "meshlab"}

    def construct_list_of_attributes():
        l = ["x", "y", "z"]  # noqa: E741
        # All channels except the 3 DC
        if rgb is not None:
            if mode == "meshlab":
                l = l + ["red", "green", "blue"]  # noqa: E741
            else:
                l = l + ["r", "g", "b"]  # noqa: E741
        if opacity is not None:
            l.append("opacity")
        if scale is not None:
            for i in range(3):
                l.append("scale_{}".format(i))
        if rotation is not None:
            for i in range(4):
                l.append("rot_{}".format(i))
        return l

    data = []
    xyz = xyz.to(dtype=torch.float32)
    xyz = xyz.detach().cpu().numpy().astype(np.float32)
    data.append(xyz)

    if rgb is not None:
        rgb = rgb.to(dtype=torch.float32)
        if mode == "meshlab":
            rgb = (rgb * 255).byte().detach().cpu().numpy().astype(np.uint8)
        else:
            rgb = rgb.detach().cpu().numpy().astype(np.float32)
        data.append(rgb)

    if opacity is not None:
        opacity = opacity.to(dtype=torch.float32)
        opacity = opacity.detach().cpu().numpy().astype(np.float32)
        data.append(opacity)

    if scale is not None:
        scale = scale.to(dtype=torch.float32)
        scale = scale.detach().cpu().numpy().astype(np.float32)
        data.append(scale)

    if rotation is not None:
        rotation = rotation.to(dtype=torch.float32)
        rotation = rotation.detach().cpu().numpy().astype(np.float32)
        data.append(rotation)

    if mode == "meshlab":
        rgb_keys = {"red", "green", "blue"}
        dtype_full = [
            (attribute, "f4") if attribute not in rgb_keys else (attribute, "u1")
            for attribute in construct_list_of_attributes()
        ]
    else:
        dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(data, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    with tempfile.TemporaryDirectory() as temp_dir:
        local_point_path = os.path.join(temp_dir, path.split("/")[-1])
        PlyData([el]).write(local_point_path)
        pathmgr.copy_from_local(local_point_path, path, overwrite=True)


def save_mesh(path, values, N, threshold, radius, save_volume=False):
    values = values.detach().cpu().numpy()
    values = values.reshape(N, N, N).astype(np.float32)

    if save_volume:
        with pathmgr.open(path + ".npy", "wb") as fp:
            np.save(fp, values)

    vertices, triangles, normals, _ = measure.marching_cubes(values, threshold)
    print(
        "vertices num %d triangles num %d threshold %.3f"
        % (vertices.shape[0], triangles.shape[0], threshold)
    )

    vertices = vertices / (N - 1.0) * 2 * radius - radius
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_normals=normals)
    with tempfile.TemporaryDirectory() as temp_dir:
        local_mesh_path = os.path.join(temp_dir, os.path.basename(path))
        mesh.export(local_mesh_path)
        pathmgr.copy_from_local(local_mesh_path, path, overwrite=True)


def save_o3d_mesh(path, mesh):
    with tempfile.TemporaryDirectory() as temp_dir:
        local_mesh_path = os.path.join(temp_dir, os.path.basename(path))
        print(local_mesh_path, path)
        o3d.io.write_triangle_mesh(local_mesh_path, mesh)
        pathmgr.copy_from_local(local_mesh_path, path, overwrite=True)


def save_o3d_pcd(path, pcd):
    with tempfile.TemporaryDirectory() as temp_dir:
        local_pcd_path = os.path.join(temp_dir, os.path.basename(path))
        print(local_pcd_path, path)
        o3d.io.write_point_cloud(local_pcd_path, pcd)
        pathmgr.copy_from_local(local_pcd_path, path, overwrite=True)


def sample_uniform_cameras(
    fov, res, dist=3, theta_range=(45, 135), theta_num=3, phi_num=4
):
    theta_arr = np.linspace(theta_range[0], theta_range[1], theta_num)
    theta_arr = theta_arr / 180.0 * np.pi
    phi_arr = np.linspace(0.0, phi_num - 1.0, phi_num) / phi_num
    phi_arr = np.pi * 2 * phi_arr
    phi_gap = phi_arr[1] - phi_arr[0]

    x_axis = np.array([1, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1], dtype=np.float32)

    fov = fov / 180.0 * np.pi
    pixel_x = (np.linspace(0, res - 1, res) + 0.5) / res
    pixel_x = (2 * pixel_x - 1) * np.tan(fov / 2.0)
    pixel_y = (np.linspace(0, res - 1, res) + 0.5) / res
    pixel_y = -(2 * pixel_y - 1) * np.tan(fov / 2.0)
    pixel_x, pixel_y = np.meshgrid(pixel_x, pixel_y)
    pixel_z = -np.ones((res, res), dtype=np.float32)

    k_arr = np.array([fov, fov, 0.5, 0.5], dtype=np.float32)

    cams_arr, rays_o_arr, rays_d_arr = [], [], []
    for n in range(0, theta_num):
        phi_arr += phi_gap / theta_num * n
        for m in range(0, phi_num):
            theta = theta_arr[n]
            phi = phi_arr[m]

            origin = (
                np.sin(theta) * np.cos(phi) * x_axis
                + np.sin(theta) * np.sin(phi) * y_axis
                + np.cos(theta) * z_axis
            )
            origin = origin * dist

            cam_z_axis = origin / np.linalg.norm(origin)
            cam_y_axis = z_axis - np.sum(z_axis * cam_z_axis) * cam_z_axis
            cam_y_axis = cam_y_axis / np.linalg.norm(cam_z_axis)
            cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

            cam = np.eye(4)
            cam[0:3, 0] = cam_x_axis
            cam[0:3, 1] = cam_y_axis
            cam[0:3, 2] = cam_z_axis
            cam[0:3, 3] = origin
            cam = cam.reshape(16)
            cam_line = np.concatenate([cam, k_arr])
            cams_arr.append(cam_line)

            rays_o = np.ones([res, res, 1]) * origin.reshape(1, 1, 3)
            rays_d = (
                cam_x_axis.reshape(1, 1, 3) * pixel_x[:, :, None]
                + cam_y_axis.reshape(1, 1, 3) * pixel_y[:, :, None]
                + cam_z_axis.reshape(1, 1, 3) * pixel_z[:, :, None]
            )
            rays_d = rays_d / np.sqrt(np.sum(rays_d * rays_d, axis=-1, keepdims=True))
            rays_o_arr.append(rays_o)
            rays_d_arr.append(rays_d)

    cams_arr = np.stack(cams_arr, axis=0)
    rays_o_arr = np.stack(rays_o_arr, axis=0)
    rays_d_arr = np.stack(rays_d_arr, axis=0)

    return cams_arr, rays_o_arr, rays_d_arr


def sample_oriented_points(preds, threshold=0.4):
    points = preds["points"]
    normals = preds["normal"]
    opacity = preds["mask"]

    opacity = opacity.reshape(-1)
    points = points.reshape(-1, 3)[opacity > threshold, :]
    normals = normals.reshape(-1, 3)[opacity > threshold, :]

    return points, normals


def filter_points_using_input_mask(points, cams, masks, fov):
    batch_size, _, height, width = masks.shape
    point_num = points.shape[0]

    fov = fov / 180.0 * np.pi

    # Enlarge mask a bit
    masks = nn.functional.adaptive_avg_pool2d(masks, (height // 8, width // 8))
    masks = nn.functional.interpolate(masks, (height, width), mode="bilinear")

    index_prod = torch.ones(point_num, dtype=torch.uint8, device=points.device).bool()
    for n in range(0, batch_size):
        cam = cams[n, :][:16].reshape(4, 4)
        diff = points - cam[:3, 3].reshape(1, 3)
        z = torch.sum(diff * cam[:3, 2].reshape(1, 3), dim=-1)
        x = torch.sum(diff * cam[:3, 0].reshape(1, 3), dim=-1)
        y = torch.sum(diff * cam[:3, 1].reshape(1, 3), dim=-1)
        x_ = (x / -z) / np.tan(fov / 2.0)
        y_ = (y / z) / np.tan(fov / 2.0)

        grid = torch.stack([x_, y_], dim=-1).reshape(1, 1, point_num, 2)
        index = nn.functional.grid_sample(masks, grid)
        index = index.reshape(-1) > 0
        index_prod = torch.logical_and(index, index_prod)

    return index_prod


def load_ply(path, device: str = "cpu"):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    rgb = np.stack(
        (
            np.asarray(plydata.elements[0]["r"]),
            np.asarray(plydata.elements[0]["g"]),
            np.asarray(plydata.elements[0]["b"]),
        ),
        axis=1,
    )

    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scale = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rotation = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.from_numpy(xyz.astype(np.float32)).to(device)
    rgb = torch.from_numpy(rgb.astype(np.float32)).to(device)
    opacity = torch.from_numpy(opacity.astype(np.float32)).to(device)
    scale = torch.from_numpy(scale.astype(np.float32)).to(device)
    rotation = torch.from_numpy(rotation.astype(np.float32)).to(device)

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _parse_slurm_node_list(s: str) -> List[str]:
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


def get_params_group_single_model(
    args, model, freeze_backbone=False, freeze_transformer=False
):
    regularized = []
    not_regularized = []
    deformation_group = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # We do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1 or "norm" in name:
            not_regularized.append(param)
        elif "ms3" in name or "omega" in name or "dxyzt" in name or "cov_t" in name:
            deformation_group.append(param)
        else:
            regularized.append(param)

    wd = args.weight_decay
    if freeze_backbone:
        lr_t = 0
        lr_d = 0
    else:
        if freeze_transformer:
            lr_t = 0
            lr_d = args.lr_d
        else:
            lr_t = args.lr
            lr_d = args.lr_d

    return [
        {"params": regularized, "weight_decay": wd, "lr": lr_t, "lr_init": lr_t},
        {"params": not_regularized, "weight_decay": 0.0, "lr": lr_t, "lr_init": lr_t},
        {"params": deformation_group, "weight_decay": 0.0, "lr": lr_d, "lr_init": lr_d},
    ]


def get_params_groups(args, **kwargs):
    params_groups = []
    for key, value in kwargs.items():  # noqa: B007
        if value is None:
            continue
        params_groups += get_params_group_single_model(args, value)
    return params_groups


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def create_video_cameras(radius, frame_num, res, elevation=20, fov=60, init_cam=None):
    fov = fov / 180.0 * np.pi
    dist = radius / np.sin(fov / 2.0) * 1.2
    theta = elevation / 180.0 * np.pi
    x_axis = np.array([1.0, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1.0, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1.0], dtype=np.float32)

    if init_cam is not None:
        init_cam = init_cam.detach().cpu().numpy()
        init_cam = init_cam.transpose(1, 0)
        inv = np.eye(4, dtype=np.float32)
        inv[0:3, 0:3] = init_cam

    camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr = [], [], [], []
    for n in range(0, frame_num):
        phi = float(n) / frame_num * np.pi * 2
        origin = (
            np.cos(theta) * np.cos(phi) * x_axis
            + np.cos(theta) * np.sin(phi) * y_axis
            + np.sin(theta) * z_axis
        )
        origin = origin * dist

        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        cam_z_axis = (origin - target) / np.linalg.norm(origin - target)
        cam_y_axis = up - np.sum(cam_z_axis * up) * cam_z_axis
        cam_y_axis = cam_y_axis / np.linalg.norm(cam_y_axis)
        cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, 0] = cam_x_axis
        extrinsic[:3, 1] = cam_y_axis
        extrinsic[:3, 2] = cam_z_axis
        extrinsic[:3, 3] = origin
        if init_cam is not None:
            extrinsic = np.matmul(inv, extrinsic)

        camera = np.zeros(20, dtype=np.float32)
        camera[0:16] = extrinsic.reshape(-1)
        camera[16:20] = np.array([fov, fov, 0.5, 0.5])

        rays_o, rays_d, rays_d_un = compute_rays(fov, extrinsic, res)

        camera_arr.append(camera)
        rays_o_arr.append(rays_o)
        rays_d_arr.append(rays_d)
        rays_d_un_arr.append(rays_d_un)

    camera_arr = np.stack(camera_arr, axis=0)[None, :, :].astype(np.float32)
    rays_o_arr = np.stack(rays_o_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_arr = np.stack(rays_d_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_un_arr = np.stack(rays_d_un_arr, axis=0)[None, :, :, :].astype(np.float32)

    return camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr


def parse_tuple_args(s):
    if s is None:
        return None
    toks = s.split(",")
    if len(toks) == 1:
        return float(s) if "." in s else int(s)
    else:
        return tuple(float(tok) if "." in tok else int(tok) for tok in toks)
