# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Main
import io
import os
import random
from collections import defaultdict

from itertools import accumulate

# Typing
from types import MethodType
from typing import Callable, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# from typing import TYPE_CHECKING

from .console_utils import (
    blue,
    dirname,
    dotdict,
    exists,
    getsize,
    isdir,
    join,
    log,
    red,
    run,
    splitext,
    tqdm,
    yellow,
)

# Utils
# from .math_utils import affine_inverse, normalize, torch_inverse_3x3

# if TYPE_CHECKING:
#     from ..models.networks.multilevel_network import MultilevelNetwork
#     from ..models.networks.volumetric_video_network import (
#         VolumetricVideoNetwork,
#     )
#     from ..runners.volumetric_video_viewer import VolumetricVideoViewer


def get_bounds(xyz, padding=0.05):
    min_xyz = torch.min(xyz, dim=-2)[0]
    max_xyz = torch.max(xyz, dim=-2)[0]
    min_xyz -= padding
    max_xyz += padding
    bounds = torch.stack([min_xyz, max_xyz], dim=-2)
    return bounds


def print_shape(batch: dotdict):
    if isinstance(batch, dict):
        for k, v in batch.items():
            print(k)
            print_shape(v)
    elif isinstance(batch, list):
        for v in batch:
            print_shape(v)
    elif isinstance(batch, torch.Tensor):
        print(f"{batch.shape}")
    else:
        print(batch)


type_mapping = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.bool: np.bool_,
}


def torch_dtype_to_numpy_dtype(torch_dtype):
    return type_mapping.get(torch_dtype, None)


def reduce_record_stats(record_stats: dotdict):
    reduced_stats = dotdict()
    for k, v in record_stats.items():
        if isinstance(v, torch.Tensor):
            reduced_stats[k] = v.item()  # MARK: will cause sync
        else:
            reduced_stats[k] = v
    return reduced_stats


def typed(input_to: torch.dtype = torch.float, output_to: torch.dtype = torch.float):
    from easyvolcap.utils.data_utils import to_x

    def wrapper(func: Callable):
        def inner(*args, **kwargs):
            args = to_x(args, input_to)
            kwargs = to_x(kwargs, input_to)
            ret = func(*args, **kwargs)
            ret = to_x(ret, output_to)
            return ret

        return inner

    return wrapper


class VolumetricVideoModule(nn.Module):
    # This module does not register 'network' as submodule
    def __init__(self, network: nn.Module = None, **kwargs) -> None:
        if hasattr(self, "unregistered"):
            # Already initializedf
            return

        super().__init__()
        self.unregistered = [network]

        # Prepare fake forward sample function
        # Hacky forward function definition
        def sample(self, *args, **kwargs):
            if not len(kwargs):
                batch = args[-1]
            else:
                batch = kwargs.pop("batch", dotdict())
            self.forward(batch)
            return None, None, None, None

        def render(self, *args, **kwargs):
            sample(self, *args, **kwargs)
            return None

        def compute(self, *args, **kwargs):
            sample(self, *args, **kwargs)
            return None, None

        def supervise(self, *args, **kwargs):
            sample(self, *args, **kwargs)
            return None, None, None

        if not hasattr(self, "sample"):
            self.sample = MethodType(sample, self)
        if not hasattr(self, "render"):
            self.render = MethodType(render, self)
        if not hasattr(self, "compute"):
            self.compute = MethodType(compute, self)
        if not hasattr(self, "supervise"):
            self.supervise = MethodType(supervise, self)

    def render_imgui(self, viewer, batch: dotdict):
        if hasattr(super(), "render_imgui"):
            super().render_imgui(viewer, batch)

    @property
    def network(self):
        network = self.unregistered[0]
        return network


class GradientModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradientModule, self).__init__()

    def take_gradient(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        d_out: torch.Tensor = None,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        return take_gradient(
            output,
            input,
            d_out,
            self.training or create_graph,
            self.training or retain_graph,
        )

    def take_jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [
            self.take_gradient(o, input, retain_graph=(i < len(outputs)))
            for i, o in enumerate(outputs)
        ]
        jac = torch.stack(grads, dim=-2)
        return jac


def get_function(f: Union[Callable, nn.Module, str]):
    if isinstance(f, str):
        try:
            return getattr(F, f)  # 'softplus'
        except AttributeError:
            pass
        try:
            return getattr(nn, f)()  # 'Identity'
        except AttributeError:
            pass
        # Using eval is dangerous, will never support that
    elif isinstance(f, nn.Module):
        return f  # nn.Identity()
    else:
        return f()  # nn.Identity


def modulize(f: Callable):
    return Modulized(f)


class Modulized(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor):
        return self.f(x)


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(torch.as_tensor(params), requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(torch.as_tensor(params), requires_grad=False)


def take_jacobian(
    func: Callable,
    input: torch.Tensor,
    create_graph=False,
    vectorize=True,
    strategy="reverse-mode",
):
    return torch.autograd.functional.jacobian(
        func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy
    )


def take_gradient(
    output: torch.Tensor,
    input: torch.Tensor,
    d_out: torch.Tensor = None,
    create_graph: bool = True,
    retain_graph: bool = True,
    is_grads_batched: bool = False,
):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(
        inputs=input,
        outputs=output,
        grad_outputs=d_output,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
        is_grads_batched=is_grads_batched,
    )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class NoopModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(f"{prefix}"):
                del state_dict[key]


class MLP(GradientModule):
    def __init__(
        self,
        input_ch=32,
        W=256,
        D=8,
        out_ch=257,
        skips=(4),
        actvn=nn.ReLU(),  # noqa: B008
        out_actvn=nn.Identity(),  # noqa: B008
        init_weight=nn.Identity(),  # noqa: B008
        init_bias=nn.Identity(),  # noqa: B008
        init_out_weight=nn.Identity(),  # noqa: B008
        init_out_bias=nn.Identity(),  # noqa: B008
        dtype=torch.float,
    ):
        super(MLP, self).__init__()
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W  # noqa: E741
            if i == 0:
                I = input_ch  # noqa: E741
            if i in skips:
                I = input_ch + W  # noqa: E741
            if i == D:
                O = out_ch  # noqa: E741
            self.linears.append(nn.Linear(I, O, dtype=dtype))
        self.linears: nn.ModuleList[nn.Linear] = nn.ModuleList(self.linears)
        self.actvn = get_function(actvn) if isinstance(actvn, str) else actvn
        self.out_actvn = (
            get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn
        )

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1:
                init_out_weight(l.weight.data)
            else:
                init_weight(l.weight.data)

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1:
                init_out_bias(l.bias.data)
            else:
                init_bias(l.bias.data)

    def forward_with_previous(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            p = x  # store output of previous layer
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x, p

    def forward(self, input: torch.Tensor):
        return self.forward_with_previous(input)[0]


def setup_deterministic(
    fix_random=True,  # all deterministic, same seed, no benchmarking
    allow_tf32=False,  # use tf32 support if possible
    deterministic=True,  # use deterministic algorithm for CNN
    benchmark=False,
    seed=0,  # only used when fix random is set to true
):
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    # https://huggingface.co/docs/transformers/v4.18.0/en/performance#tf32
    torch.backends.cuda.matmul.allow_tf32 = (
        allow_tf32  # by default, tf32 support of CNNs is enabled
    )
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if fix_random:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)


class deterministic_block:
    def __enter__(self):
        self.allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        self.benchmark = torch.backends.cudnn.benchmark
        setup_deterministic(True)

    def __exit__(self, type, value, traceback):
        setup_deterministic(
            False,
            allow_tf32=self.allow_tf32,
            deterministic=self.deterministic,
            benchmark=self.benchmark,
        )


def get_state_dict(state_dict: dotdict, prefix: str = ""):
    if len(prefix) and not prefix.endswith("."):
        prefix = prefix + "."
    d = dotdict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            d[k[len(prefix) :]] = v
    return d


def float_to_half_state_dict(state_dict: dict, key: str = ""):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float and key in k:
            state_dict[k] = v.half()


def half_to_float_state_dict(state_dict: dict, key: str = ""):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.half and key in k:
            state_dict[k] = v.float()


def load_pretrained(  # noqa: C901
    model_dir: str,
    resume: bool = True,
    epoch: int = -1,
    ext: str = ".npz",
    remove_if_not_resuming: bool = False,
    warn_if_not_exist: bool = False,
    only_check_existence: bool = False,
):
    if not resume:  # remove nothing here
        if remove_if_not_resuming:
            if isdir(model_dir) and len(
                os.listdir(model_dir)
            ):  # only inform the use if there are files
                # log(red(f"Removing trained weights: {blue(model_dir)}"))
                try:
                    run(f"rm -r {model_dir}")
                except:  # noqa: B001
                    pass
        return None, None

    if not exists(model_dir):
        if warn_if_not_exist:
            log(red(f"Pretrained network: {blue(model_dir)} does not exist"))
        return None, None

    if isdir(model_dir):
        pts = sorted(os.listdir(model_dir))
        int_pts = [
            int(pt.split(".")[0])
            for pt in pts
            if pt != f"latest{ext}"
            and pt.endswith(ext)
            and pt.split(".")[0].isnumeric()
        ]
        if len(pts) == 0:
            log(red(f"Pretrained network: {blue(model_dir)} does not exist"))
            return None, None
        if epoch == -1:
            if f"latest{ext}" in os.listdir(model_dir):
                pt = "latest"
            elif len(int_pts):
                pt = max(int_pts)
            else:
                pt = pts[0]  # at least one of them exist
        else:
            pt = epoch
        model_path = join(model_dir, f"{pt}{ext}")
    else:
        model_path = model_dir
        ext = splitext(model_path)[-1]

    if not exists(model_path):
        if warn_if_not_exist:
            log(red(f"Pretrained network: {blue(model_path)} does not exist"))
        return None, None

    if only_check_existence:
        return None, model_path

    log(
        f"Loading model from {blue(model_path)}, size: {getsize(model_path) / 1024 / 1024:.3f} MB..."
    )
    if ext == ".pt" or ext == ".pth":
        # We use this method to load large models in chunks to avoid memory issues
        # and to provide a progress bar for better user experience
        file_size = getsize(model_path)
        with open(model_path, "rb") as f:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Loading model",
            ) as pbar:
                buffer = io.BytesIO()
                for chunk in iter(lambda: f.read(8192), b""):
                    buffer.write(chunk)
                    pbar.update(len(chunk))
                buffer.seek(0)
                pretrained = dotdict(torch.load(buffer, "cpu", weights_only=False))
    else:
        # For .npz files, we load data in a similar chunked manner
        # This allows us to show progress and handle potentially large files
        from easyvolcap.utils.data_utils import to_tensor

        file_size = getsize(model_path)
        with open(model_path, "rb") as f:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Loading model",
            ) as pbar:
                buffer = io.BytesIO()
                for chunk in iter(lambda: f.read(8192), b""):
                    buffer.write(chunk)
                    pbar.update(len(chunk))
                buffer.seek(0)
                with np.load(buffer) as data:
                    model = {key: to_tensor(value) for key, value in data.items()}
        epoch = model.pop("epoch", -1)
        pretrained = dotdict(
            model=model, epoch=epoch
        )  # the npz files do not contain training parameters

    return pretrained, model_path


def load_model(  # noqa: C901
    model: nn.Module,
    optimizer: Union[nn.Module, None] = None,
    scheduler: Union[nn.Module, None] = None,
    moderator: Union[nn.Module, None] = None,
    model_dir: str = "",
    resume: bool = True,
    epoch: int = -1,
    strict: bool = True,  # report errors when loading "model" instead of network
    skips: List[str] = (),
    only: List[str] = (),
    allow_mismatch: List[str] = (),
    ext=".pt",
):
    pretrained, model_path = load_pretrained(
        model_dir,
        resume,
        epoch,
        ext,
        remove_if_not_resuming=True,
        warn_if_not_exist=False,
    )
    if pretrained is None:
        return 0

    pretrained_model = pretrained["model"]
    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(
            pretrained_model.keys()
        )  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split(".")
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(
                model_parent,
                last_name,
                nn.Parameter(
                    pretrained_parent[last_name],
                    requires_grad=getattr(model_parent, last_name).requires_grad,
                ),
            )  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(
        pretrained_model, strict=strict
    )
    if optimizer is not None and "optimizer" in pretrained.keys():
        optimizer.load_state_dict(pretrained["optimizer"])
    if scheduler is not None and "scheduler" in pretrained.keys():
        scheduler.load_state_dict(pretrained["scheduler"])
    if moderator is not None and "moderator" in pretrained.keys():
        moderator.load_state_dict(pretrained["moderator"])

    epoch = pretrained["epoch"]
    if isinstance(epoch, torch.Tensor) or isinstance(epoch, np.ndarray):
        epoch = epoch.item()
    log(f"Loaded model {blue(model_path)} at epoch {blue(epoch)}")
    return epoch + 1


def load_network(  # noqa: C901
    model: nn.Module,
    model_dir: str = "",
    resume: bool = True,  # when resume is False, will try as a fresh restart
    epoch: int = -1,
    strict: bool = True,  # report errors if something is wrong
    skips: List[str] = (),
    only: List[str] = (),
    prefix: str = "",  # will match and remove these prefix
    allow_mismatch: List[str] = (),
):
    pretrained, model_path = load_pretrained(
        model_dir, resume, epoch, remove_if_not_resuming=False, warn_if_not_exist=False
    )
    if pretrained is None:
        pretrained, model_path = load_pretrained(
            model_dir,
            resume,
            epoch,
            ".pth",
            remove_if_not_resuming=False,
            warn_if_not_exist=False,
        )
    if pretrained is None:
        pretrained, model_path = load_pretrained(
            model_dir,
            resume,
            epoch,
            ".pt",
            remove_if_not_resuming=False,
            warn_if_not_exist=resume,
        )
    if pretrained is None:
        return 0

    # log(f'Loading network: {blue(model_path)}')
    # ordered dict cannot be mutated while iterating
    # vanilla dict cannot change size while iterating
    pretrained_model = pretrained["model"]

    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(
            pretrained_model.keys()
        )  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    if prefix:
        keys = list(
            pretrained_model.keys()
        )  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if k.startswith(prefix):
                pretrained_model[k[len(prefix) :]] = pretrained_model[k]
            del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model and not strict:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split(".")
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(
                model_parent,
                last_name,
                nn.Parameter(
                    pretrained_parent[last_name],
                    requires_grad=getattr(model_parent, last_name).requires_grad,
                ),
            )  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(
        pretrained_model, strict=strict
    )

    epoch = pretrained["epoch"]
    if isinstance(epoch, torch.Tensor) or isinstance(epoch, np.ndarray):
        epoch = epoch.item()
    log(f"Loaded network {blue(model_path)} at epoch {blue(epoch)}")
    return epoch + 1


def save_npz(
    model: nn.Module,
    model_dir: str = "",
    epoch: int = -1,
    latest: int = True,
    path: str = None,
):
    from easyvolcap.utils.data_utils import to_numpy

    model_path = (
        path if path else join(model_dir, "latest.npz" if latest else f"{epoch}.npz")
    )
    state_dict = (
        model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
    )
    param_dict = to_numpy(state_dict)  # a shallow dict (dotdict)
    param_dict.epoch = epoch  # just 1 scalar
    os.makedirs(dirname(model_path), exist_ok=True)
    log(yellow(f"Saving model to {blue(model_path)}..."))
    # Save the model with a progress bar
    # total_size = sum(param.nbytes if hasattr(param, 'nbytes') else param.size * param.itemsize for param in param_dict.values() if isinstance(param, np.ndarray))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **param_dict)
    buffer.seek(0)
    buffer_size = buffer.getbuffer().nbytes
    with tqdm(
        total=buffer_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Saving npz",
    ) as pbar:
        with open(model_path, "wb") as f:
            for chunk in iter(lambda: buffer.read(8192), b""):
                f.write(chunk)
                pbar.update(len(chunk))
    log(
        yellow(
            f"Saved model {blue(model_path)} at epoch {blue(epoch)}, size: {getsize(model_path) / 1024 / 1024:.3f} MB"
        )
    )


def save_model(
    model: nn.Module,
    optimizer: Union[nn.Module, None] = None,
    scheduler: Union[nn.Module, None] = None,
    moderator: Union[nn.Module, None] = None,
    model_dir: str = "",
    epoch: int = -1,
    latest: int = False,
    save_lim: int = 5,
    path: str = None,
):
    model = {
        # Special handling for ddp modules (incorrect naming)
        "model": model.state_dict()
        if not isinstance(model, DDP)
        else model.module.state_dict(),
        "epoch": epoch,
    }

    if optimizer is not None:
        model["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        model["scheduler"] = scheduler.state_dict()

    if moderator is not None:
        model["moderator"] = moderator.state_dict()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = (
        path if path else join(model_dir, "latest.pt" if latest else f"{epoch}.pt")
    )
    log(yellow(f"Saving model to {blue(model_path)}..."))
    # Save the model with a progress bar
    # total_size = sum(
    #     (param.numel() * param.element_size() if hasattr(param, 'numel') else param.size * param.itemsize)
    #     for param in model['model'].values()
    # )
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    buffer_size = buffer.getbuffer().nbytes
    with tqdm(
        total=buffer_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Saving model",
    ) as pbar:
        with open(model_path, "wb") as f:
            for chunk in iter(lambda: buffer.read(8192), b""):
                f.write(chunk)
                pbar.update(len(chunk))
    log(
        yellow(
            f"Saved model {blue(model_path)} at epoch {blue(epoch)}, size: {getsize(model_path) / 1024 / 1024:.3f} MB"
        )
    )

    ext = ".pt"
    pts = [
        int(pt.split(".")[0])
        for pt in os.listdir(model_dir)
        if pt != f"latest{ext}" and pt.endswith(ext) and pt.split(".")[0].isnumeric()
    ]
    if len(pts) <= save_lim:
        return
    else:
        removing = join(model_dir, f"{min(pts)}.pt")
        # log(red(f"Removing trained weights: {blue(removing)}"))
        os.remove(removing)


def root_of_any(k, l):  # noqa: E741
    for s in l:
        a = accumulate(k.split("."), lambda x, y: x + "." + y)
        for r in a:
            if s == r:
                return True
    return False


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def reset_optimizer_state(optimizer):
    optimizer.__setstate__({"state": defaultdict(dict)})


def update_optimizer_state(optimizer, optimizer_state):
    for k, v in optimizer_state.items():
        if v.new_params is None:
            continue
        val = optimizer.state[k].copy()
        exp_avg = torch.zeros_like(v.new_params)
        exp_avg[v.new_keep] = val["exp_avg"][v.old_keep]
        val["exp_avg"] = exp_avg
        exp_avg_sq = torch.zeros_like(v.new_params)
        exp_avg_sq[v.new_keep] = val["exp_avg_sq"][v.old_keep]
        val["exp_avg_sq"] = exp_avg_sq
        del optimizer.state[k]
        optimizer.state[v.new_params] = val


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2**20
