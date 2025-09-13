# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from ...easyvolcap.utils.math_utils import normalize
from ..blocks import get_layernorm, SelfAttentionBlock
from ...registry import MODELS


class BaseImageEncoderConfig(PretrainedConfig):
    model_type = "BaseImageEncoder"

    def __init__(
        self,
        # encoder args
        input_size=(None, None),
        input_sq_size=32,
        in_channels=3,
        out_channels=12,
        patch_size=(8, 8),
        hidden_size=1152,
        depth=28,
        drop_path=0.0,
        num_heads=16,
        mlp_ratio=4.0,
        qk_norm=False,
        enable_flash_attn=True,
        enable_layernorm_kernel=True,
        # upsampler
        up_factor=0,
        # gaussian heads
        t_norm=False,
        pos_bnorm=False,
        use_bias=True,
        use_pe=True,
        use_deconv=False,
        use_rmsnorm=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.drop_path = drop_path
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel

        self.up_factor = up_factor
        self.t_norm = t_norm
        self.pos_bnorm = pos_bnorm
        self.use_pe = use_pe
        self.use_bias = use_bias
        self.use_deconv = use_deconv
        self.use_rmsnorm = use_rmsnorm
        super().__init__(**kwargs)


@MODELS.register_module()
class BaseImageEncoder(PreTrainedModel):
    config_class = BaseImageEncoderConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.H = self.input_size[0] // self.patch_size[0]
        self.W = (
            self.input_size[1] // self.patch_size[1]
        )  # H, W: num_patch_h, num_patch_w
        self.num_patches = self.H * self.W

        self.t_norm = config.t_norm
        self.use_bias = config.use_bias
        self.use_pe = config.use_pe
        self.use_deconv = config.use_deconv
        self.qk_norm = config.qk_norm
        self.use_rmsnorm = config.use_rmsnorm

        self.input_linear = nn.Linear(
            10 * np.prod(self.patch_size), config.hidden_size, bias=self.use_bias
        )
        drop_path = [
            x.item() for x in torch.linspace(0, config.drop_path, config.depth)
        ]

        self.concat_norm = get_layernorm(
            hidden_size=self.hidden_size,
            eps=1e-6,
            affine=True,
            use_kernel=self.enable_layernorm_kernel,
            bias=self.use_bias,
        )

        self.rope = None

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope,
                    qk_norm=self.qk_norm,
                    bias=self.use_bias,
                    rmsnorm=self.use_rmsnorm,
                )
                for i in range(self.depth)
            ]
        )
        # init model
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for i, block in enumerate(self.blocks):
            block: SelfAttentionBlock
            std = 0.02 / (2 * (i + 1)) ** 0.5
            nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=std)

        if self.use_pe:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=1e-6)

    def plucker(self, H: int, W: int, K: torch.Tensor, RT: torch.Tensor):
        B, T, _, _ = K.size()
        RT = RT.reshape(B * T, 4, 4)
        K = K.reshape(B * T, 3, 3)

        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)

        y, x = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=RT.device, dtype=RT.dtype),
            torch.linspace(0, W - 1, W, device=RT.device, dtype=RT.dtype),
            indexing="ij",
        )
        x = x.reshape([-1, H * W]).expand([B * T, H * W]) + 0.5  # [B*V, HxW]
        y = y.reshape([-1, H * W]).expand([B * T, H * W]) + 0.5  # [B*V, HxW]
        x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, RT[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_o = RT[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = ray_o.reshape(B, T, H, W, 3)
        ray_d = ray_d.reshape(B, T, H, W, 3)
        ray_o = ray_o.permute(0, 4, 1, 2, 3)  # B, 3, T, H, W
        ray_d = ray_d.permute(0, 4, 1, 2, 3)  # B, 3, T, H, W
        normd = normalize(ray_d)
        plucker = torch.cat([torch.cross(ray_o, normd, dim=1), normd], dim=1)
        return plucker, ray_o, ray_d

    @abstractmethod
    def forward(
        self,
        num_input: int,
        num_sup: int,
        x: torch.Tensor,
        ts: torch.Tensor,
        K: torch.Tensor,
        RT: torch.Tensor,
    ):
        """
        Abstract forward pass to be implemented by subclasses.
        
        Args:
            num_input: Number of input frames
            num_sup: Number of support frames
            x: Input images [B, T, 3, H, W]
            ts: Timestamps [B, T, 1]
            K: Camera intrinsics [B, T, 3, 3]
            RT: Camera extrinsics [B, T, 4, 4]
        
        Returns:
            Implementation-specific output
        """
        raise NotImplementedError("Subclasses must implement the forward method")



@MODELS.register_module("BaseImageEncoder-B")
def BaseImageEncoder_B(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        raise NotImplementedError("Pretrained model loading not yet implemented")
    config = BaseImageEncoderConfig(**kwargs)
    return BaseImageEncoder(config)
