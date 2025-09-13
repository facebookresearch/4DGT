# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from ..easyvolcap.utils.console_utils import green, logger

try:
    from flash_attn_interface import (  # noqa: F401 # type: ignore
        flash_attn_func,
        flash_attn_qkvpacked_func,
    )
    logger.info(green("Successfully imported flash_attn_func for hopper architecture"))
except ImportError:
    try:
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func  # noqa: F401
        logger.warn(
            "Unable to import flash attention v3 for hopper architecture, falling back to v2"
        )
    except ImportError:
        flash_attn_func = None
        flash_attn_qkvpacked_func = None
        logger.warn("Flash attention not available")

from timm.models.layers import DropPath

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(
    hidden_size: torch.Tensor,
    eps: float,
    affine: bool,
    use_kernel: bool,
    bias: bool = False,
):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm

            class FusedLayerNorm(ApexFusedLayerNorm):
                def __init__(
                    self,
                    normalized_shape,
                    eps: float = 1e-5,
                    elementwise_affine: bool = True,
                    bias: bool = True,
                    **kwargs,
                ) -> None:
                    super().__init__(
                        normalized_shape, eps, elementwise_affine, **kwargs
                    )
                    if not bias:
                        data = self.bias.data
                        del self.bias
                        # just the tensor data, no parameters & optimization
                        # self.register_buffer("bias", torch.zeros_like(bias))
                        self.bias = data

                def forward(self, *args, **kwargs):
                    self.bias = self.bias.to(self.weight, non_blocking=True)
                    return (
                        super().forward(*args, **kwargs).float()
                    )  # force it to be float for better precision

            norm = FusedLayerNorm(
                hidden_size, 1e-5, elementwise_affine=affine, bias=bias
            )
            return norm
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, 1e-5, elementwise_affine=affine, bias=bias)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x



# ===============================================
# General-purpose Layers
# ===============================================

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # Frequency bands for sinusoidal rotation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = None
        self._device_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if (
            self._cos_cached is not None
            and self._seq_len_cached == seq_len
            and self._device_cached == device
        ):
            return  # cache hit

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        # Store for broadcasting: (1, seq_len, 1, dim)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        self._seq_len_cached = seq_len
        self._device_cached = device

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (..., seq_len, dim) where dim must match self.dim
        Returns:
            Rotated x with same shape
        """
        *prefix, seq_len, dim = x.shape
        assert dim == self.dim

        self._build_cache(seq_len, x.device)
        cos = self._cos_cached  # (1, seq_len, 1, dim)
        sin = self._sin_cached

        # Split x
        x_even = x[..., ::2]  # (..., seq_len, dim/2)
        x_odd = x[..., 1::2]  # (..., seq_len, dim/2)

        # Slice cos/sin into even/odd channels
        cos_even = cos[..., ::2]  # (seq_len, dim/2)
        cos_odd = cos[..., 1::2]
        sin_even = sin[..., ::2]
        sin_odd = sin[..., 1::2]

        # Now shapes align:
        #   x_even: (..., seq_len, dim/2)
        #   cos_even: (1, seq_len, 1, dim/2)
        x_rot_even = x_even * cos_even - x_odd * sin_even
        x_rot_odd = x_even * sin_odd + x_odd * cos_odd

        # Re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, dim: int, D: int, H: int, W: int, base: float = 100.0):
        """
        Args:
          dim: total embedding dimension (must be divisible by 6)
          D, H, W: sizes along depth, height, width
          base: rotary frequency base
        """
        super().__init__()
        assert dim % 3 == 0 and (dim // 3) % 2 == 0, "dim must be divisible by 6"
        self.dim = dim
        self.base = base
        self.D, self.H, self.W = D, H, W

        # each axis gets dim//3 dims; within that we use pairs
        self.axis_dim = dim // 3
        half_pairs = self.axis_dim // 2

        # inv freqs for each axis
        inv_freq = 1.0 / (base ** (torch.arange(0, half_pairs).float() * 2 / dim))
        self.register_buffer("inv_freq_d", inv_freq)
        self.register_buffer("inv_freq_h", inv_freq.clone())
        self.register_buffer("inv_freq_w", inv_freq.clone())

        # caches
        self._cache = {}

    def _build_cache(self, device: torch.device):
        key = (self.D, self.H, self.W, device)
        if key in self._cache:
            return

        # positions
        d_pos = torch.arange(self.D, device=device).type_as(self.inv_freq_d)
        h_pos = torch.arange(self.H, device=device).type_as(self.inv_freq_h)
        w_pos = torch.arange(self.W, device=device).type_as(self.inv_freq_w)

        # freqs: (D, half_pairs), (H, half_pairs), (W, half_pairs)
        freq_d = torch.einsum("i,j->ij", d_pos, self.inv_freq_d)
        freq_h = torch.einsum("i,j->ij", h_pos, self.inv_freq_h)
        freq_w = torch.einsum("i,j->ij", w_pos, self.inv_freq_w)

        # duplicate to full axis_dim
        emb_d = torch.cat([freq_d, freq_d], dim=-1)  # (D, axis_dim)
        emb_h = torch.cat([freq_h, freq_h], dim=-1)  # (H, axis_dim)
        emb_w = torch.cat([freq_w, freq_w], dim=-1)  # (W, axis_dim)

        # sin/cos and reshape to broadcast over other dims
        cos_d = emb_d.cos().view(1, self.D, 1, 1, self.axis_dim)
        sin_d = emb_d.sin().view(1, self.D, 1, 1, self.axis_dim)
        cos_h = emb_h.cos().view(1, 1, self.H, 1, self.axis_dim)
        sin_h = emb_h.sin().view(1, 1, self.H, 1, self.axis_dim)
        cos_w = emb_w.cos().view(1, 1, 1, self.W, self.axis_dim)
        sin_w = emb_w.sin().view(1, 1, 1, self.W, self.axis_dim)

        self._cache[key] = (cos_d, sin_d, cos_h, sin_h, cos_w, sin_w)

    def forward(self, x: torch.Tensor, D=None, H=None, W=None):
        """
        Args:
          x: (B, D, H, W, C) where C == dim
        Returns:
          rotated x, same shape
        """
        # B, D, H, W, C = x.shape
        # assert C == self.dim
        # assert (D, H, W) == (self.D, self.H, self.W)
        x, c = x[..., :-1, :], x[..., -1:, :]
        *prefix, seq_len, dim = x.shape
        D, H, W, C = self.D, self.H, self.W, self.dim  # noqa: F841
        x = x.reshape(-1, D, H, W, C)

        self._build_cache(x.device)
        cos_d, sin_d, cos_h, sin_h, cos_w, sin_w = self._cache[(D, H, W, x.device)]

        # split channels into three axis parts
        xd, xh, xw = x.split(self.axis_dim, dim=-1)

        def rotate(x_sub, cos, sin):
            # x_sub: (B, *, C_sub)
            x_even = x_sub[..., ::2]
            x_odd = x_sub[..., 1::2]
            cos_e = cos[..., ::2]
            sin_e = sin[..., ::2]
            cos_o = cos[..., 1::2]
            sin_o = sin[..., 1::2]
            r_even = x_even * cos_e - x_odd * sin_e
            r_odd = x_even * sin_o + x_odd * cos_o
            out = torch.empty_like(x_sub)
            out[..., ::2] = r_even
            out[..., 1::2] = r_odd
            return out

        xd = rotate(xd, cos_d, sin_d)
        xh = rotate(xh, cos_h, sin_h)
        xw = rotate(xw, cos_w, sin_w)

        # concat back
        x = torch.cat([xd, xh, xw], dim=-1).reshape(prefix + [seq_len, C])
        x = torch.cat([x, c], dim=-2)
        return x




class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = True,
        enable_torch_attn: bool = False,
        causal: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_torch_attn = enable_torch_attn
        self.causal = causal
        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        # enable_torch_attn = self.enable_torch_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)  # 3, B, HEAD, N, HEAD_DIM
        q, k, v = qkv.unbind(0)
        # WARNING: this may be a bug
        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        q, k = self.q_norm(q), self.k_norm(k)

        if enable_flash_attn:
            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            x = flash_attn_func(
                q,
                k,
                v,
                # dropout_p=self.attn_drop.p if self.training else 0.0,
                causal=self.causal,
                softmax_scale=self.scale,
            )
            if not isinstance(x, torch.Tensor):
                x = x[0]  # v3, first element of a list
        elif self.enable_torch_attn:

            # q = q.permute(0, 2, 1, 3)
            # k = k.permute(0, 2, 1, 3)
            # v = v.permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MlpB(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=True,
        enable_layernorm_kernel=True,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
        bias=True,
        norm_bias=False,
        rmsnorm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self._enable_sequence_parallelism = enable_sequence_parallelism

        # spatial branch
        if rmsnorm:
            self.norm1 = LlamaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        else:
            self.norm1 = get_layernorm(
                hidden_size,
                eps=1e-6,
                affine=True,
                use_kernel=enable_layernorm_kernel,
                bias=norm_bias,
            )
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=bias,
            enable_flash_attn=enable_flash_attn,
            rope=rope,
            qk_norm=qk_norm,
        )

        # mlp branch
        if rmsnorm:
            self.norm2 = LlamaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        else:
            self.norm2 = get_layernorm(
                hidden_size,
                eps=1e-6,
                affine=True,
                use_kernel=enable_layernorm_kernel,
                bias=bias,
            )
        self.mlp = MlpB(
            # TODO CHEKC
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=0,
            bias=bias,
            # in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=QuickGELU, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x):
        x = x.permute(1, 0, 2)
        x = self.attn(x, x, x, need_weights=False, attn_mask=None)[0]
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x, T=None, S=None):
        B, N, C = x.shape
        x_a = self.norm1(x)
        x_a = self.attn(x_a)
        x = x + self.drop_path(x_a)

        x_m = self.norm2(x)
        x_m = self.mlp(x_m)
        x = x + self.drop_path(x_m)

        return x


def build_pytorch_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    act_layer=nn.GELU,
    depth=10,
    bias=False,
    use_weight_norm=False,
) -> nn.Sequential:
    """Build a PyTorch MLP with configurable depth and options.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        act_layer: Activation layer class (default: nn.GELU)
        depth: Number of hidden layers (0 for direct linear projection)
        bias: Whether to use bias in linear layers
        use_weight_norm: Whether to apply weight normalization
    
    Returns:
        nn.Sequential: The constructed MLP
    """
    if depth == 0:
        mlp = [nn.Linear(input_dim, output_dim, bias=bias)]
    else:
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        mlp.append(act_layer())
        for _ in range(depth - 1):
            mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            mlp.append(act_layer())
        mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    if use_weight_norm:
        mlp = [
            torch.nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
            for layer in mlp
        ]
    mlp = nn.Sequential(*mlp)
    return mlp
