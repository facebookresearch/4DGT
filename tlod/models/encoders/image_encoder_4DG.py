# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Basic pixel-time aligned 4D gaussian regression
Outputs a FreeTimeGS: xyzt(4), scale(4), rotation(4->3), opacity(1), velocity(3)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from ...acceleration.checkpoint import auto_grad_checkpoint
from ..geometry import dt_to_cov_t, radius_to_sigma
from .image_encoder import BaseImageEncoder
from ..blocks import get_layernorm, SelfAttentionBlock, build_pytorch_mlp
from ...registry import MODELS


class ImageEncoder4DGConfig(PretrainedConfig):
    model_type = "BaseImageEncoder"

    def __init__(
        self,
        # encoder args
        input_size=(None, None),
        patch_size=(8, 8),
        hidden_size=768,
        depth=24,
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
        use_bias=False,
        use_pe=False,
        use_deconv=False,
        use_rmsnorm=False,
        **kwargs,
    ):
        self.input_size = input_size
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
class ImageEncoder4DG(BaseImageEncoder):
    def __init__(
        self,
        config,
        token_dim=768,
        mlp_dim=256,
        mlp_depth=1,
        cp=True,
        norm_layer=nn.LayerNorm,
        depth_near=0.0,
        depth_far=50,
        depth_bias=-4.0,
        scale_bias=-2.3,
        opacity_bias=-2.3,
        dxyzt_bias=0.0,
        ms3_bias=0.0,
        cov_t_bias=-2.3,
        norm_use_bias=True,
        norm_use_affine=True,
        use_weight_norm=False,
        input_image_num=None,
        output_image_num=None,
        color_space="rgbm",
        **kwargs,
    ):
        super(BaseImageEncoder, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.H = self.input_size[0] // self.patch_size[0]
        self.W = (
            self.input_size[1] // self.patch_size[1]
        )  # H, W: num_patch_h, num_patch_w
        self.num_patches = self.H * self.W

        self.t_norm = config.t_norm
        self.use_bias = config.use_bias
        self.use_pe = config.use_pe
        self.use_deconv = config.use_deconv
        self.use_rmsnorm = config.use_rmsnorm
        self.pos_bnorm = config.pos_bnorm

        self.concat_norm = get_layernorm(
            hidden_size=self.hidden_size,
            eps=1e-6,
            affine=True,
            use_kernel=self.enable_layernorm_kernel,
            bias=self.use_bias,
        )

        self.input_linear = nn.Linear(
            10 * np.prod(self.patch_size), config.hidden_size, bias=self.use_bias
        )
        drop_path = [
            x.item() for x in torch.linspace(0, config.drop_path, config.depth)
        ]

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
                    qk_norm=config.qk_norm,
                    bias=self.use_bias,
                    rmsnorm=self.use_rmsnorm,
                )
                for i in range(self.depth)
            ]
        )

        # init model
        self.initialize_weights()

        self.token_dim = token_dim
        self.cp = cp  # Unused
        self.color_space = color_space

        self.input_image_num = input_image_num
        self.output_image_num = output_image_num
        self.depth_near = depth_near
        self.depth_far = depth_far
        self.depth_bias = depth_bias

        self.scale_bias = scale_bias
        self.opacity_bias = opacity_bias
        self.dxyzt_bias = dxyzt_bias
        self.ms3_bias = ms3_bias
        self.cov_t_bias = cov_t_bias

        if color_space == "rgb":
            self.color_dim = 3
        elif color_space == "rgbm":
            self.color_dim = 4
        else:
            raise NotImplementedError(color_space)

        self.norm = norm_layer(
            token_dim, elementwise_affine=norm_use_affine, bias=norm_use_bias
        )

        self.mlp_depth = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 1,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_rgb = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * self.color_dim,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_opacity = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 1,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_scale = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 3,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_rotation = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 4,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_dxyzt = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 4,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_ms3 = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 3,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )
        self.mlp_cov_t = build_pytorch_mlp(
            token_dim,
            mlp_dim,
            np.prod(self.patch_size) * 1,
            depth=mlp_depth,
            bias=False,
            use_weight_norm=use_weight_norm,
        )

    @property
    def dtype(self):
        return self.input_linear.weight.dtype

    @property
    def device(self):
        return self.input_linear.weight.device

    def rescale_predictions(
        self,
        depth=None,
        opacity=None,
        scale=None,
        rotation=None,
        dxyzt=None,
        ms3=None,
        cov_t=None,
        rgb=None,
    ):
        if depth is not None:
            depth = torch.sigmoid(depth + self.depth_bias)
            depth = (1 - depth) * self.depth_near + depth * self.depth_far
        if opacity is not None:
            opacity = torch.sigmoid(opacity + self.opacity_bias)
        if scale is not None:
            l = 0.001  # noqa: E741
            u = 0.050  # should not be too large due to per-frame pred
            scale = (scale + self.scale_bias).sigmoid() * (u - l) + l  # controls mono
            scale = radius_to_sigma(scale)
        if rotation is not None:
            rotation = F.normalize(rotation, dim=-1, eps=1.0e-8)
        if dxyzt is not None:
            dxyzt = (dxyzt + self.dxyzt_bias).tanh() * 0.050  # 0.05m movement diff
        if ms3 is not None:
            l = 0.0  # noqa: E741
            u = 2.0  # should not be too fast
            speed = F.gelu(ms3 + self.ms3_bias).norm(dim=-1, keepdim=True).clamp(l, u)
            ms3 = speed * F.normalize(ms3, dim=-1, eps=1.0e-8)
        if cov_t is not None:
            l = 0.1  # 100 ms minimum # noqa: E741
            u = 100.0  # can live very long
            cov_t = (cov_t + self.cov_t_bias).sigmoid() * (u - l) + l
            cov_t = dt_to_cov_t(cov_t)
        if rgb is not None:
            rgb = rgb.sigmoid()
        return depth, opacity, scale, rotation, dxyzt, ms3, cov_t, rgb

    def decode2(
        self,
        geo_token: torch.Tensor,
        app_token: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ts: torch.Tensor,
    ):
        """
        Decode geometry and appearance tokens into 4D Gaussian parameters.
        
        Args:
            geo_token: Geometry tokens [B, (N_t*N_h*N_w), (C_out*H_p*W_p)]
            app_token: Appearance tokens [B, (N_t*N_h*N_w), (C_out*H_p*W_p)]
            ray_o: Ray origins [B, 3, T, H, W]
            ray_d: Ray directions [B, 3, T, H, W]
            ts: Timestamps [B, 1, T, H, W]
        
        Returns:
            dict: Gaussian parameters with keys:
                - xyz: 3D positions [B, T*H*W, 3]
                - depth: Depth values [B, T*H*W, 1]
                - feature: RGB/RGBM features [B, T*H*W, 3/4]
                - opacity: Opacity values [B, T*H*W, 1]
                - scaling: Scale parameters [B, T*H*W, 3]
                - rotation: Rotation quaternions [B, T*H*W, 4]
                - ms3: Motion scale 3D [B, T*H*W, 3]
                - cov_t: Temporal covariance [B, T*H*W, 1]
                - dxyzt: Position and time perturbations [B, T*H*W, 4]
                - t: Time values [B, T*H*W, 1]
        """

        # Preserve precision
        geo_token = geo_token.float()
        app_token = app_token.float()
        ray_o = ray_o.float()
        ray_d = ray_d.float()
        ts = ts.float()

        B, _, T, H, W = ray_d.shape
        token = self.norm(geo_token)
        app_token = self.norm(app_token)
        depth = self.unpatchify(self.mlp_depth(token), 1, H, W).permute(0, 2, 3, 4, 1)
        rgb = self.unpatchify(self.mlp_rgb(app_token), self.color_dim, H, W).permute(
            0, 2, 3, 4, 1
        )
        opacity = self.unpatchify(self.mlp_opacity(token), 1, H, W).permute(
            0, 2, 3, 4, 1
        )
        scale = self.unpatchify(self.mlp_scale(token), 3, H, W).permute(0, 2, 3, 4, 1)
        rotation = self.unpatchify(self.mlp_rotation(token), 4, H, W).permute(
            0, 2, 3, 4, 1
        )
        dxyzt = self.unpatchify(self.mlp_dxyzt(token), 4, H, W).permute(0, 2, 3, 4, 1)
        ms3 = self.unpatchify(self.mlp_ms3(token), 3, H, W).permute(0, 2, 3, 4, 1)
        cov_t = self.unpatchify(self.mlp_cov_t(token), 1, H, W).permute(0, 2, 3, 4, 1)

        """
        Post-process the output range and format
        """
        depth, opacity, scale, rotation, dxyzt, ms3, cov_t, rgb = (
            self.rescale_predictions(
                depth=depth,
                opacity=opacity,
                scale=scale,
                rotation=rotation,
                dxyzt=dxyzt,
                ms3=ms3,
                cov_t=cov_t,
                rgb=rgb,
            )
        )

        """
        Return
        """
        B, T, H, W, _ = depth.shape
        gs_params = {}
        gs_params["depth"] = depth.view(B, T * H * W, -1)
        gs_params["feature"] = rgb.view(B, T * H * W, -1)
        gs_params["opacity"] = opacity.view(B, T * H * W, -1)
        gs_params["scaling"] = scale.view(B, T * H * W, -1)
        gs_params["rotation"] = rotation.view(B, T * H * W, -1)

        ray_o = ray_o.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, 3)  # B, T, H, W, 3
        ray_d = ray_d.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, 3)  # B, T, H, W, 3
        gs_params["xyz"] = gs_params["depth"] * ray_d + ray_o

        """
        4DGS related
        """
        gs_params["ms3"] = ms3.view(B, T * H * W, -1)
        gs_params["cov_t"] = cov_t.view(B, T * H * W, -1)

        """
        Compute perturbed xyzt
        """
        ts = ts.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, 1)
        gs_params["dxyzt"] = dxyzt.view(B, T * H * W, -1)
        gs_params["xyz"] = gs_params["xyz"] + gs_params["dxyzt"][..., :3]
        gs_params["t"] = ts + gs_params["dxyzt"][..., 3:]

        for k in gs_params:
            gs_params[k] = gs_params[k].float()  # preserve precision
        return gs_params

    def decode(
        self,
        token: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ts: torch.Tensor,
    ):
        return self.decode2(token, token, ray_o, ray_d, ts)

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
        Forward pass for 4D Gaussian parameter prediction.
        
        Args:
            num_input: Number of input frames
            num_sup: Number of support frames (unused)
            x: Input images [B, T, 3, H, W]
            ts: Timestamps [B, T, 1]
            K: Camera intrinsics [B, T, 3, 3]
            RT: Camera extrinsics [B, T, 4, 4]
        
        Returns:
            dict: Gaussian parameters including xyz, depth, features, opacity,
                  scaling, rotation, ms3, cov_t, dxyzt, and t
        """
        ts = ts.float()
        K = K.float()
        RT = RT.float()

        x = torch.transpose(x, 1, 2).contiguous()  # output: [B, 3, T, Hx, Wx]
        ts = torch.transpose(ts, 1, 2)  # [B, 1, T, Hx, Wx]
        dtype = self.input_linear.weight.dtype
        x = x.to(dtype)
        B, _, Tx, Hx, Wx = x.shape  # Tx: num_frame, Hx, Wx: image_h, image_w

        plucker, ray_o, ray_d = self.plucker(Hx, Wx, K, RT)  # b, 6, T, Hx, Wx
        plucker = plucker.to(dtype=dtype, device=x.device)

        ts = ts[:, :, :, None, None].expand(B, 1, Tx, Hx, Wx)
        x = torch.cat([x, ts, plucker], dim=1)  # [B, 10, T, Hx, Wx]

        x = self.patchify(x)
        x = self.input_linear(x)
        x = self.concat_norm(x)

        # blocks
        for _, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
            )

        x = x.float()
        gs_params = self.decode(x, ray_o, ray_d, ts)
        return gs_params

    def patchify(self, x: torch.Tensor, Hp: int = None, Wp: int = None):
        """Convert input tensor to patches.
        
        Args:
            x: Input tensor [B, C, N, H, W] where N is temporal dimension
            Hp: Patch height (default: self.patch_size[0])
            Wp: Patch width (default: self.patch_size[1])
        
        Returns:
            Patched tensor [B, (N*H*W)/(Hp*Wp), C*Hp*Wp]
        """
        from einops import rearrange
        
        Hp = Hp if Hp is not None else self.patch_size[0]
        Wp = Wp if Wp is not None else self.patch_size[1]
        B, C, N, H, W = x.shape
        x = rearrange(
            x,
            "B C_out N_t (N_h H_p) (N_w W_p) -> B (N_t N_h N_w) (C_out H_p W_p)",
            N_t=N,
            N_h=H // Hp,
            N_w=W // Wp,
            H_p=Hp,
            W_p=Wp,
            C_out=C,
        )
        return x  # B, NHW/PP, CPP

    def unpatchify(
        self,
        x: torch.Tensor,
        C: int = None,
        H: int = None,
        W: int = None,
        Hp: int = None,
        Wp: int = None,
    ):
        """Convert patches back to tensor format.
        
        Args:
            x: Patched tensor [B, num_patches, patch_features]
            C: Number of channels (default: inferred from patch features)
            H: Output height (default: self.H * Hp)
            W: Output width (default: self.W * Wp)
            Hp: Patch height (default: self.patch_size[0])
            Wp: Patch width (default: self.patch_size[1])
        
        Returns:
            Unpatched tensor [B, C, N, H, W]
        """
        from einops import rearrange
        
        Hp = Hp if Hp is not None else self.patch_size[0]
        Wp = Wp if Wp is not None else self.patch_size[1]
        H = H if H is not None else self.H * Hp
        W = W if W is not None else self.W * Wp
        C = C if C is not None else int(x.shape[-1] / Hp / Wp)
        B, NHW_PP, CPP = x.shape
        N = NHW_PP // (H * W // Hp // Wp)
        x = rearrange(
            x,
            "B (N_t N_h N_w) (C_out H_p W_p) -> B C_out N_t (N_h H_p) (N_w W_p)",
            N_t=N,
            N_h=H // Hp,
            N_w=W // Wp,
            H_p=Hp,
            W_p=Wp,
            C_out=C,
        )
        return x  # B, C, T, H, W


@MODELS.register_module("ImageEncoder4DG-B")
def ImageEncoder4DG_B(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        raise NotImplementedError("Pretrained model loading not yet implemented")
    config = ImageEncoder4DGConfig(**kwargs)
    return ImageEncoder4DG(config)
