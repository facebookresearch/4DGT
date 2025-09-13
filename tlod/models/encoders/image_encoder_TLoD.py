# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Key-frame based 4D gaussian regression, using plucker rays of some other normalized frames, in a self-supervised manner
Outputs a FreeTimeGS: xyzt(4), scale(4), rotation(4->3), opacity(1), velocity(3)
"""

from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import PretrainedConfig

from ...acceleration.checkpoint import auto_grad_checkpoint
from ...easyvolcap.utils.chunk_utils import multi_gather
from ...easyvolcap.utils.console_utils import (
    dotdict,
    logger,
    tqdm
)
from ...easyvolcap.utils.math_utils import normalize
from ...easyvolcap.utils.net_utils import (
    freeze_module,
    make_buffer,
)
from ..geometry import (
    dt_to_cov_t,
    radius_to_sigma,
    sigma_to_radius,
)

from .image_encoder_4DG import ImageEncoder4DG
from .image_encoder import BaseImageEncoder
from ..blocks import (
    build_pytorch_mlp,
    get_layernorm,
    RotaryPositionEmbedding,
    RotaryPositionEmbedding3D,
    SelfAttentionBlock,
)
from ...registry import MODELS


class TLoDConfig(PretrainedConfig):  # Temporal Level of Detail Config
    model_type = "BaseImageEncoder"

    def __init__(
        self,
        # Encoder args
        input_size=(252, 252, 16),
        patch_size=(14, 14, 1),
        hidden_size=1536,
        depth=12,
        detail_depth=12,
        global_depth=12,
        drop_path=0.0,
        num_heads=16,
        mlp_ratio=4.0,
        qk_norm=False,
        enable_flash_attn=True,
        enable_layernorm_kernel=False,
        t_norm=False,
        pos_bnorm=False,
        use_bias=False,
        use_pe=False,
        use_deconv=False,
        use_rmsnorm=False,
        # Gaussian heads
        mlp_dim=256,
        mlp_depth=2,
        norm_use_bias=True,
        norm_use_affine=True,
        use_weight_norm=False,
        input_image_num=None,
        output_image_num=None,
        color_space="rgbm",
        # DINOv2 encoder args
        dinov2_version="dinov2_vitb14",
        use_prenorm_dinov2=False,
        use_rope3d=False,
        use_rope=False,
        # Decoding related
        depth_near=0.0,
        depth_far=50.0,
        depth_bias=-4.0,
        scale_min=sigma_to_radius(0.0001),  # noqa: B008
        scale_max=sigma_to_radius(0.5),  # noqa: B008
        scale_bias=-2.3,
        opacity_min=0.0,  # default to 0.5
        opacity_max=1.0,  # default to 0.5
        opacity_bias=0.0,  # default to 0.5
        dxyzt_min=-0.5,
        dxyzt_max=0.5,
        dxyzt_bias=0.0,  # default to zero
        cov_t_min=0.1,
        cov_t_max=100.0,
        cov_t_bias=-2.0,  # default to 50s lifespan
        rgb_min=0.0,
        rgb_max=1.0,
        rgb_bias=0.0,
        sigmoid_ms3_min=0.0,
        sigmoid_ms3_max=10.0,
        sigmoid_ms3_bias=-6.9068,  # default to 0.01 m/s
        sigmoid_omega_min=0.0,
        sigmoid_omega_max=10.0,
        sigmoid_omega_bias=-6.9068,  # default to 0.01 2pi rad/s
        omega_clamp: float = 0.01,  # default to 0.01 2pi rad/s
        ms3_clamp: float = 0.01,  # default to 0.01 2pi rad/s
        # Motion degree
        ms3_deg: int = 1,
        # Rotation degree
        omega_deg: int = 1,
        # Better motion regularization
        ms3_deg_downmax_mult: float = 1.0,
        omega_deg_downmax_mult: float = 1.0,
        dxyzt_mult: float = 1.0,
        # Level of detail
        n_levels: int = 1,  # implies a 2x jump and 1x starting point for sampling initial input
        # global_downsample=(4, 4, 1),
        global_downsample=(1, 1, 1),
        # magic_pattern=(
        #     [2, 5, 5, 6, 6, 7, 7, 12, 12, 13],
        #     [8, 6, 7, 0, 4, 1, 6, 1, 8, 2],
        # ),
        magic_pattern=([], []),
        # patch_random_pattern=(177, 64, 187, 47, 80, 85, 106, 67, 105, 50),
        local_split=4,
        magic_num=10,
        recalc_magic_pattern=False,
        recalc_method="patch_sorting",
        upsample_ratio=1.0,
        **kwargs,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.detail_depth = detail_depth
        self.global_depth = global_depth
        self.drop_path = drop_path
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel

        self.t_norm = t_norm
        self.pos_bnorm = pos_bnorm
        self.use_pe = use_pe
        self.use_bias = use_bias
        self.use_deconv = use_deconv
        self.use_rmsnorm = use_rmsnorm

        self.mlp_dim = mlp_dim
        self.mlp_depth = mlp_depth
        self.norm_use_bias = norm_use_bias
        self.norm_use_affine = norm_use_affine
        self.use_weight_norm = use_weight_norm
        self.input_image_num = input_image_num
        self.output_image_num = output_image_num
        self.color_space = color_space

        self.depth_near = depth_near
        self.depth_far = depth_far
        self.depth_bias = depth_bias
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_bias = scale_bias
        self.opacity_min = opacity_min
        self.opacity_max = opacity_max
        self.opacity_bias = opacity_bias
        self.dxyzt_min = dxyzt_min
        self.dxyzt_max = dxyzt_max
        self.dxyzt_bias = dxyzt_bias
        self.cov_t_min = cov_t_min
        self.cov_t_max = cov_t_max
        self.cov_t_bias = cov_t_bias
        self.rgb_min = rgb_min
        self.rgb_max = rgb_max
        self.rgb_bias = rgb_bias

        self.sigmoid_ms3_min = sigmoid_ms3_min
        self.sigmoid_ms3_max = sigmoid_ms3_max
        self.sigmoid_ms3_bias = sigmoid_ms3_bias
        self.sigmoid_omega_min = sigmoid_omega_min
        self.sigmoid_omega_max = sigmoid_omega_max
        self.sigmoid_omega_bias = sigmoid_omega_bias

        self.dinov2_version = dinov2_version
        self.use_prenorm_dinov2 = use_prenorm_dinov2
        self.use_rope3d = use_rope3d  # BUG: what...
        self.use_rope = use_rope

        self.ms3_deg = ms3_deg
        self.omega_deg = omega_deg

        self.omega_clamp = omega_clamp
        self.ms3_clamp = ms3_clamp

        self.ms3_deg_downmax_mult = ms3_deg_downmax_mult
        self.omega_deg_downmax_mult = omega_deg_downmax_mult

        self.dxyzt_mult = dxyzt_mult

        self.n_levels = n_levels

        self.global_downsample = global_downsample
        self.magic_pattern = magic_pattern
        self.local_split = local_split
        self.magic_num = magic_num
        self.recalc_magic_pattern = recalc_magic_pattern
        self.recalc_method = recalc_method
        # self.patch_random_pattern = patch_random_pattern
        self.upsample_ratio = upsample_ratio
        super().__init__(**kwargs)


@MODELS.register_module()
class ImageEncoderTLoD(ImageEncoder4DG):
    def __init__(  # noqa: C901
        self,
        config,
        **kwargs,
    ):
        super(BaseImageEncoder, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.detail_depth = config.detail_depth
        self.global_depth = config.global_depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.global_downsample = config.global_downsample
        self.magic_pattern = config.magic_pattern
        self.recalc_magic_pattern = config.recalc_magic_pattern
        self.recalc_method = config.recalc_method
        if "patch" in self.recalc_method:
            random_shape = np.prod(self.patch_size[:2])
        else:
            random_shape = np.prod(self.input_size[:2])
        self.random_pattern = make_buffer(
            torch.arange(random_shape)[
                torch.randperm(random_shape)[: int(random_shape / 19.6 + 0.5)]
            ],
        )

        self.H = self.input_size[0] // self.patch_size[0]
        self.W = self.input_size[1] // self.patch_size[1]
        self.T = self.input_size[2] // self.patch_size[2]
        self.local_split = config.local_split
        self.magic_num = config.magic_num

        self.t_norm = config.t_norm
        self.qk_norm = config.qk_norm
        self.mlp_dim = config.mlp_dim
        self.mlp_depth = config.mlp_depth
        self.use_bias = config.use_bias
        self.norm_use_bias = config.norm_use_bias
        self.norm_use_affine = config.norm_use_affine
        self.use_pe = config.use_pe
        self.use_deconv = config.use_deconv
        self.use_rmsnorm = config.use_rmsnorm
        self.pos_bnorm = config.pos_bnorm
        self.use_weight_norm = config.use_weight_norm

        self.concat_norm = get_layernorm(
            hidden_size=self.hidden_size,
            eps=1e-6,
            affine=self.norm_use_affine,
            use_kernel=self.enable_layernorm_kernel,
            # bias=self.norm_use_bias,
        )

        # Construct dinov2 encoder for feature extraction
        self._unregistered_dinov2 = [
            torch.hub.load('facebookresearch/dinov2', config.dinov2_version)
        ]
            
        self.dinov2.to("cuda", non_blocking=True)
        self.dinov2.eval()
        freeze_module(self.dinov2)
        feature_dim = self.dinov2.embed_dim
        self.use_prenorm_dinov2 = config.use_prenorm_dinov2
        self.use_rope3d = config.use_rope3d
        self.use_rope = config.use_rope

        # Input image encoder
        channel_dim = 10

        self.input_linear = nn.Linear(
            channel_dim * np.prod(self.patch_size) + feature_dim * self.patch_size[-1],
            self.hidden_size,
            bias=self.use_bias,
        )  # rgb + plucker + t

        # Main fusion transformer
        drop_path = [
            x.item() for x in torch.linspace(0, config.drop_path, config.depth)
        ]
        if self.use_rope3d:
            self.rope = RotaryPositionEmbedding3D(
                self.hidden_size // self.num_heads, self.T, self.H, self.W
            )
        elif self.use_rope:
            self.rope = RotaryPositionEmbedding(self.hidden_size // self.num_heads)
        else:
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
                    # norm_bias=self.norm_use_bias,
                    rmsnorm=self.use_rmsnorm,
                )
                for i in tqdm(
                    range(self.depth), desc="Building fusion transformer blocks"
                )
            ]
        )

        # Used for classification & multi-level decoding
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.plucker = MethodType(BaseImageEncoder.plucker, self)

        self.norm = get_layernorm(
            self.hidden_size,
            eps=1e-6,
            affine=self.norm_use_affine,
            bias=self.norm_use_bias,
            use_kernel=self.enable_layernorm_kernel,
        )

        self.color_space = config.color_space
        self.input_image_num = config.input_image_num
        self.output_image_num = config.output_image_num

        self.depth_near = config.depth_near
        self.depth_far = config.depth_far
        self.depth_bias = config.depth_bias
        self.scale_min = config.scale_min
        self.scale_max = config.scale_max
        self.scale_bias = config.scale_bias
        self.opacity_min = config.opacity_min
        self.opacity_max = config.opacity_max
        self.opacity_bias = config.opacity_bias
        self.dxyzt_min = config.dxyzt_min
        self.dxyzt_max = config.dxyzt_max
        self.dxyzt_bias = config.dxyzt_bias
        self.cov_t_min = config.cov_t_min
        self.cov_t_max = config.cov_t_max
        self.cov_t_bias = config.cov_t_bias
        self.rgb_min = config.rgb_min
        self.rgb_max = config.rgb_max
        self.rgb_bias = config.rgb_bias

        self.sigmoid_ms3_min = config.sigmoid_ms3_min
        self.sigmoid_ms3_max = config.sigmoid_ms3_max
        self.sigmoid_ms3_bias = config.sigmoid_ms3_bias
        self.sigmoid_omega_min = config.sigmoid_omega_min
        self.sigmoid_omega_max = config.sigmoid_omega_max
        self.sigmoid_omega_bias = config.sigmoid_omega_bias

        self.ms3_deg = config.ms3_deg
        self.omega_deg = config.omega_deg

        self.omega_clamp = config.omega_clamp
        self.ms3_clamp = config.ms3_clamp

        self.ms3_deg_downmax_mult = config.ms3_deg_downmax_mult
        self.omega_deg_downmax_mult = config.omega_deg_downmax_mult
        self.dxyzt_mult = config.dxyzt_mult

        self.n_levels = config.n_levels
        self.upsample_ratio = config.upsample_ratio

        if config.color_space == "rgb":
            self.color_dim = 3
        elif config.color_space == "rgbm":
            self.color_dim = 4
        else:
            raise NotImplementedError(config.color_space)

        # Define Gaussian parameter names and their channel dimensions
        self.gs_params = [
            "depth",     # Distance along ray from camera origin
            "opacity",   # Alpha/transparency value for blending
            "scale",     # 3D scale factors (sx, sy, sz) for Gaussian size
            "rotation",  # Quaternion (qw, qx, qy, qz) for 3D orientation
            "dxyzt",     # Position residuals (dx, dy, dz, dt) for fine adjustments
            "ms3",       # Marginal scale for 4D motion (velocity components)
            "cov_t",     # Temporal covariance for time extent
            "omega",     # Angular velocity for rotation over time
            "rgb",       # Color features (RGB or RGBM depending on color_space)
        ]
        # Number of channels for each parameter
        self.gs_params_split = [
            1,                    # depth: 1 channel
            1,                    # opacity: 1 channel  
            3,                    # scale: 3 channels (x, y, z)
            4,                    # rotation: 4 channels (quaternion)
            4,                    # dxyzt: 4 channels (3 spatial + 1 temporal)
            4 * self.ms3_deg,     # ms3: 4 channels per degree (3 direction + 1 magnitude)
            1,                    # cov_t: 1 channel
            4 * self.omega_deg,   # omega: 4 channels per degree (3 axis + 1 magnitude)
            self.color_dim,       # rgb: 3 for RGB or 4 for RGBM
        ]
        self.out_channels = sum(self.gs_params_split)  # Total output channels

        Hp, Wp, Tp = self.patch_size
        heads = {}
        for key, channel in zip(self.gs_params, self.gs_params_split):
            head = build_pytorch_mlp(
                self.hidden_size,
                self.mlp_dim * channel,
                Hp * Wp * Tp * channel,
                depth=self.mlp_depth,
                bias=self.use_bias,
                use_weight_norm=self.use_weight_norm,
            )
            heads[key] = head
        self.linear_heads = nn.ModuleDict(heads)

        def output_head(x: torch.Tensor):
            return torch.cat([head(x) for head in self.linear_heads.values()], dim=-1)

        self.output_head = output_head

        if self.n_levels > 1:
            self.detail_linear = nn.Linear(
                channel_dim * np.prod(self.patch_size)
                + feature_dim * self.patch_size[-1],
                # + self.hidden_size,
                # self.hidden_size * 2,  # skip connection + orginal feature
                # channel_dim * np.prod(self.patch_size) + feature_dim * self.patch_size[-1],
                self.hidden_size,
                bias=self.use_bias,
            )  # rgb + plucker + t + all previous gs parameters

            self.detail_extra_linear = nn.Linear(
                # channel_dim * np.prod(self.patch_size)
                # + feature_dim * self.patch_size[-1],
                # + self.hidden_size,
                self.hidden_size * 2,  # skip connection + orginal feature
                # channel_dim * np.prod(self.patch_size) + feature_dim * self.patch_size[-1],
                self.hidden_size,
                bias=self.use_bias,
            )  # rgb + plucker + t + all previous gs parameters

            # Main reinitialization transformer
            self.details = nn.ModuleList(
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
                        # norm_bias=self.norm_use_bias,
                        rmsnorm=self.use_rmsnorm,
                    )
                    for i in tqdm(
                        range(self.detail_depth),
                        desc="Building detail transformer blocks",
                    )
                ]
            )

            self.detail_concat_norm = get_layernorm(
                hidden_size=self.hidden_size,
                eps=1e-6,
                affine=self.norm_use_affine,
                # bias=self.norm_use_bias,
                use_kernel=self.enable_layernorm_kernel,
            )

            self.detail_norm = get_layernorm(
                self.hidden_size,
                eps=1e-6,
                affine=self.norm_use_affine,
                bias=self.norm_use_bias,
                use_kernel=self.enable_layernorm_kernel,
            )

            heads = {}
            for key, channel in zip(self.gs_params, self.gs_params_split):
                head = build_pytorch_mlp(
                    self.hidden_size,
                    self.mlp_dim * channel,
                    Hp * Wp * Tp * channel,
                    depth=self.mlp_depth,
                    bias=self.use_bias,
                    use_weight_norm=self.use_weight_norm,
                )
                heads[key] = head
            self.detail_heads = nn.ModuleDict(heads)

            def detail_head(x: torch.Tensor):
                return torch.cat(
                    [head(x) for head in self.detail_heads.values()], dim=-1
                )

            self.detail_head = detail_head

        if self.n_levels > 2:
            self.global_linear = nn.Linear(
                channel_dim * np.prod(self.patch_size)
                + feature_dim * self.patch_size[-1],
                # + self.hidden_size,
                # self.hidden_size * 2,  # skip connection + orginal feature
                # channel_dim * np.prod(self.patch_size) + feature_dim * self.patch_size[-1],
                self.hidden_size,
                bias=self.use_bias,
            )  # rgb + plucker + t + all previous gs parameters

            self.global_extra_linear = nn.Linear(
                # channel_dim * np.prod(self.patch_size)
                # + feature_dim * self.patch_size[-1],
                # + self.hidden_size,
                self.hidden_size * 2,  # skip connection + orginal feature
                # channel_dim * np.prod(self.patch_size) + feature_dim * self.patch_size[-1],
                self.hidden_size,
                bias=self.use_bias,
            )  # rgb + plucker + t + all previous gs parameters

            # Main reinitialization transformer
            self.globals = nn.ModuleList(
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
                        # norm_bias=self.norm_use_bias,
                        rmsnorm=self.use_rmsnorm,
                    )
                    for i in tqdm(
                        range(self.global_depth),
                        desc="Building global transformer blocks",
                    )
                ]
            )

            self.global_concat_norm = get_layernorm(
                hidden_size=self.hidden_size,
                eps=1e-6,
                affine=self.norm_use_affine,
                # bias=self.norm_use_bias,
                use_kernel=self.enable_layernorm_kernel,
            )

            self.global_norm = get_layernorm(
                self.hidden_size,
                eps=1e-6,
                affine=self.norm_use_affine,
                bias=self.norm_use_bias,
                use_kernel=self.enable_layernorm_kernel,
            )

            heads = {}
            for key, channel in zip(self.gs_params, self.gs_params_split):
                head = build_pytorch_mlp(
                    self.hidden_size,
                    self.mlp_dim * channel,
                    Hp * Wp * Tp * channel,
                    depth=self.mlp_depth,
                    bias=self.use_bias,
                    use_weight_norm=self.use_weight_norm,
                )
                heads[key] = head
            self.global_heads = nn.ModuleDict(heads)

            def global_head(x: torch.Tensor):
                return torch.cat(
                    [head(x) for head in self.global_heads.values()], dim=-1
                )

            self.global_head = global_head

        # init model
        self.initialize_weights()

    def initialize_weights(self):  # noqa: C901
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

        if hasattr(self, "output_linear"):
            for i, block in enumerate(self.details):
                block: SelfAttentionBlock
                std = 0.02 / (2 * (i + 1)) ** 0.5
                nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=std)
                nn.init.normal_(block.attn.proj.weight, mean=0.0, std=std)

        if self.use_pe:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "output_linear"):
            nn.init.normal_(self.output_linear.linear.weight, mean=0.0, std=0.02)
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=1e-6)
        if hasattr(self, "linear_heads"):
            for key, head in self.linear_heads.items():  # noqa: B007
                nn.init.normal_(head[0].weight, mean=0.0, std=0.02)
        if hasattr(self, "detail_heads"):
            for key, head in self.detail_heads.items():  # noqa: B007
                nn.init.normal_(head[0].weight, mean=0.0, std=0.02)

        if not self.norm_use_bias:
            for k, v in self.named_parameters():
                if "bias" in k and "norm" in k:
                    logger.info(f"Disabling gradient for: {k}")
                    v.requires_grad_(False)

    @property
    def dinov2(self):
        return self._unregistered_dinov2[0]

    @property
    def dtype(self):
        return self.input_linear.weight.dtype

    @property
    def device(self):
        return self.input_linear.weight.device

    def patchify(self, x: torch.Tensor, Hp: int = None, Wp: int = None, Tp: int = None):
        # x: B, 9, N, H, W
        Hp = Hp if Hp is not None else self.patch_size[0]
        Wp = Wp if Wp is not None else self.patch_size[1]
        Tp = Tp if Tp is not None else self.patch_size[2]
        B, C, T, H, W = x.shape
        # B, N * H * W // (self.patch_size[0] * self.patch_size[1]), C, self.patch_size[0], self.patch_size[1]
        x = rearrange(
            x,
            "B C_out (N_t T_p) (N_h H_p) (N_w W_p) -> B (N_t N_h N_w) (C_out H_p W_p T_p)",
            N_t=T // Tp,
            N_h=H // Hp,
            N_w=W // Wp,
            T_p=Tp,
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
        T: int = None,
        Hp: int = None,
        Wp: int = None,
        Tp: int = None,
    ):
        # x: B, 9, N, H, W
        Hp = Hp if Hp is not None else self.patch_size[0]
        Wp = Wp if Wp is not None else self.patch_size[1]
        Tp = Tp if Tp is not None else self.patch_size[2]
        H = H if H is not None else self.input_size[0]
        W = W if W is not None else self.input_size[1]
        T = T if T is not None else self.input_size[2]
        C = C if C is not None else int(x.shape[-1] // Hp // Wp // Tp)
        B, NHW_PP, CPP = x.shape
        T = NHW_PP // (H * W // Hp // Wp) * Tp
        x = rearrange(
            x,
            "B (N_t N_h N_w) (C_out H_p W_p T_p) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=T // Tp,
            N_h=H // Hp,
            N_w=W // Wp,
            H_p=Hp,
            W_p=Wp,
            T_p=Tp,
            C_out=C,
        )
        return x  # B, C, T, H, W

    def rescale_predictions(  # noqa: C901
        self,
        depth=None,
        opacity=None,
        scale=None,
        rotation=None,
        dxyzt=None,
        ms3=None,
        cov_t=None,
        omega=None,
        rgb=None,
    ):
        """
        Apply activation functions and rescale raw predictions to valid ranges.
        
        This function transforms raw network outputs into physically meaningful
        parameters for 4D Gaussian splatting. Each parameter is rescaled using
        sigmoid activation and learned biases to ensure valid ranges.
        
        Args:
            depth: Raw depth predictions along ray direction
                Shape: [B, T, H, W, 1] or similar
            opacity: Raw opacity/alpha predictions
                Shape: [B, T, H, W, 1]
            scale: Raw 3D scale factors for Gaussians
                Shape: [B, T, H, W, 3]
            rotation: Raw rotation quaternions
                Shape: [B, T, H, W, 4]
            dxyzt: Raw spatial-temporal position residuals
                Shape: [B, T, H, W, 4] - (dx, dy, dz, dt)
            ms3: Raw marginal scale parameters for multi-scale modeling
                Shape: [B, T, H, W, 4*ms3_deg] - includes speed components
            cov_t: Raw temporal covariance
                Shape: [B, T, H, W, 1]
            omega: Raw angular velocity
                Shape: [B, T, H, W, 4*omega_deg]
            rgb: Raw RGB color values (optional)
                Shape: [B, T, H, W, 3]
                
        Returns:
            Tuple of rescaled parameters in same order as inputs,
            with physically valid ranges applied
        """
        # Rescale depth to valid range [near, far] using sigmoid activation
        if depth is not None:
            depth = (depth + self.depth_bias).sigmoid() * (
                self.depth_far - self.depth_near
            ) + self.depth_near  # Map to [depth_near, depth_far] range
            
        # Rescale opacity to valid range [min, max] for transparency control
        if opacity is not None:
            opacity = (opacity + self.opacity_bias).sigmoid() * (
                self.opacity_max - self.opacity_min
            ) + self.opacity_min  # Map to [opacity_min, opacity_max] range
            
        # Rescale 3D scale factors and convert from radius to standard deviation
        if scale is not None:
            scale = (scale + self.scale_bias).sigmoid() * (
                self.scale_max - self.scale_min
            ) + self.scale_min  # Map to [scale_min, scale_max] range
            scale = radius_to_sigma(scale)  # Convert radius to Gaussian sigma
            
        # Normalize rotation quaternions to unit length
        if rotation is not None:
            rotation = normalize(rotation)  # Ensure valid unit quaternion
            
        # Rescale spatial-temporal position residuals
        if dxyzt is not None:
            dxyzt = ((dxyzt + self.dxyzt_bias) * self.dxyzt_mult).sigmoid() * (
                self.dxyzt_max - self.dxyzt_min
            ) + self.dxyzt_min  # Map to small displacement range (e.g., 0.05m)
            
        # Process marginal scale parameters for multi-scale representation
        if ms3 is not None:
            # Extract degree of marginal scale (number of scale components)
            ms3_deg = ms3.shape[-1] // 4
            # Extract speed components (every 4th element starting from index 3)
            speed = ms3[..., 3::4, None]  # [B, T, H, W, ms3_deg, 1]
            # Reshape spatial components (first 3 of every 4 elements)
            ms3 = torch.cat(
                [ms3[..., None, i * 4 : i * 4 + 3] for i in range(ms3_deg)], dim=-2
            )  # [B, T, H, W, ms3_deg, 3]

            # Rescale speed with sigmoid and apply clamping threshold
            speed = (speed + self.sigmoid_ms3_bias).sigmoid() * (
                self.sigmoid_ms3_max - self.sigmoid_ms3_min
            ) + self.sigmoid_ms3_min
            speed = (speed - self.ms3_clamp).clamp(0)  # Zero out speeds below threshold

            # Apply decay factor to speed based on scale level  
            # Higher scale levels get progressively smaller speeds
            speed = torch.cat(
                [speed[..., i : i + 1, :] / self.ms3_deg_downmax_mult**i 
                 for i in range(ms3_deg)], dim=-2
            )  # [B, T, H, W, ms3_deg, 1]

            # Apply speed-modulated normalized marginal scales
            ms3 = speed * normalize(ms3[..., :3])  # Normalize and modulate by speed
            ms3 = ms3.reshape(ms3.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, ms3_deg*3]
            
        # Process temporal covariance for time-varying Gaussians
        if cov_t is not None:
            # Rescale temporal covariance using sigmoid activation
            cov_t = (cov_t + self.cov_t_bias).sigmoid() * (
                self.cov_t_max - self.cov_t_min
            ) + self.cov_t_min  # Map to [cov_t_min, cov_t_max] range
            cov_t = dt_to_cov_t(cov_t)  # Convert temporal difference to covariance
            
        # Process angular velocity for rotating Gaussians
        if omega is not None:
            # Extract degree of angular velocity (number of rotation components)
            omega_deg = omega.shape[-1] // 4
            # Extract angular speed components (every 4th element)
            speed = omega[..., 3::4, None]  # [B, T, H, W, omega_deg, 1]
            # Reshape angular direction components (first 3 of every 4)
            omega = torch.cat(
                [omega[..., None, i * 4 : i * 4 + 3] for i in range(omega_deg)], dim=-2
            )  # [B, T, H, W, omega_deg, 3]

            # Rescale angular speed with sigmoid activation
            speed = (speed + self.sigmoid_omega_bias).sigmoid() * (
                self.sigmoid_omega_max - self.sigmoid_omega_min
            ) + self.sigmoid_omega_min
            speed = (speed - self.omega_clamp).clamp(0)  # Zero out small angular velocities

            # Apply decay factor to angular speed based on degree
            # Higher degrees get progressively smaller angular velocities
            speed = torch.cat(
                [speed[..., i : i + 1, :] / self.omega_deg_downmax_mult**i
                 for i in range(omega_deg)], dim=-2
            )  # [B, T, H, W, omega_deg, 1]

            # Apply speed-modulated normalized angular velocity
            omega = speed * normalize(omega[..., :3])  # Normalize axis and modulate by speed
            omega = omega.reshape(omega.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, omega_deg*3]
            
        # Process RGB color values with learned bias and range
        if rgb is not None:
            # Rescale RGB values to valid color range
            rgb = (rgb + self.rgb_bias).sigmoid() * (
                self.rgb_max - self.rgb_min
            ) + self.rgb_min  # Map to [rgb_min, rgb_max] range (typically [0, 1])
            
        return depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, rgb

    def decode_wo_act(
        self,
        token: torch.Tensor,
        ray_d: torch.Tensor,
        output_head=None,
    ):
        """Decode tokens to raw Gaussian parameters without activation functions.
        
        This function converts encoded feature tokens into raw 4D Gaussian parameters
        without applying activation functions or rescaling. It serves as the first stage
        of the decoding pipeline, producing intermediate representations that will be
        processed by activation functions in the main decode() method.
        
        Pipeline:
        1. Apply output head to generate raw parameters
        2. Unpatchify from token space back to image space
        3. Split concatenated parameters into individual components
        4. Return as dictionary of raw parameters
        
        Args:
            token: Encoded feature tokens from transformer.
                Shape: [B, N_tokens, C] where N_tokens = (T*H*W)/(Tp*Hp*Wp)
                B: Batch size, N_tokens: Number of patches, C: Channel dimension
            ray_d: Ray directions for extracting dimensions.
                Shape: [B, 3, T, H, W] - normalized ray directions
            output_head: Optional output projection head.
                Default: None (uses self.output_head)
                
        Returns:
            Dict containing raw Gaussian parameters (before activation):
                - depth: Raw depth values [B, T, H, W, 1]
                - feature: Raw appearance features [B, T, H, W, C_feat]
                - opacity: Raw opacity values [B, T, H, W, 1]
                - scaling: Raw 3D scale factors [B, T, H, W, 3]
                - rotation: Raw rotation parameters [B, T, H, W, 4]
                - ms3: Raw marginal scale factors [B, T, H, W, 4*ms3_deg]
                - cov_t: Raw temporal covariance [B, T, H, W, 1]
                - omega: Raw angular velocity [B, T, H, W, 4*omega_deg]
                - dxyzt: Raw position perturbations [B, T, H, W, 4]
        """
        # Stage 1: Extract spatial-temporal dimensions from ray_d
        B, _, T, H, W = ray_d.shape  # [B, 3, T, H, W] extract dimensions
        Hp, Wp, Tp = self.patch_size  # Patch sizes for height, width, time
        
        # Stage 2: Project tokens to output dimension and unpatchify
        gs_params = (
            self.unpatchify(
                auto_grad_checkpoint(
                    self.output_head if output_head is None else output_head, token
                ),  # [B, N_tokens, C] -> [B, N_tokens, C_out] project to output channels
                self.out_channels,  # Total number of output channels
                H,  # Target height
                W,  # Target width  
                T,  # Target time
                Hp,  # Height patch size
                Wp,  # Width patch size
                Tp,  # Time patch size
            )  # [B, N_tokens, C_out] -> [B, C_out, T, H, W] unpatchify to image space
        )
        gs_params = rearrange(gs_params, 'b c t h w -> b t h w c')  # [B, C_out, T, H, W] -> [B, T, H, W, C_out]
        gs_params = gs_params.float()  # Ensure float32 precision
        
        # Stage 3: Split concatenated parameters into individual components
        depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = (
            gs_params.split(self.gs_params_split, dim=-1)
        )  # Split [B, T, H, W, C_out] into individual parameters based on gs_params_split
        
        # Stage 4: Package parameters into dictionary
        gs_params = {}
        gs_params["depth"] = depth  # [B, T, H, W, 1] depth along ray
        gs_params["feature"] = feat  # [B, T, H, W, C_feat] appearance features
        gs_params["opacity"] = opacity  # [B, T, H, W, 1] opacity/alpha
        gs_params["scaling"] = scale  # [B, T, H, W, 3] 3D scale factors
        gs_params["rotation"] = rotation  # [B, T, H, W, 4] rotation quaternion
        gs_params["ms3"] = ms3  # [B, T, H, W, 4*ms3_deg] marginal scale
        gs_params["cov_t"] = cov_t  # [B, T, H, W, 1] temporal covariance
        gs_params["omega"] = omega  # [B, T, H, W, 4*omega_deg] angular velocity
        gs_params["dxyzt"] = dxyzt  # [B, T, H, W, 4] position residuals

        return gs_params

    def decode(
        self,
        token: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ts: torch.Tensor,
        output_head=None,
    ):
        """Decode tokens into 4D Gaussian parameters with temporal dynamics.
        
        This function converts encoded feature tokens back into 4D Gaussian splatting
        parameters including position, appearance, and temporal components. It performs
        post-processing including activation functions, spatial-temporal perturbations,
        and coordinate transformations.
        
        Args:
            token: Encoded feature tokens from the encoder.
                Shape: [B, N_tokens, C] where N_tokens = (T*H*W)/(Tp*Hp*Wp)
            ray_o: Ray origins for each pixel.
                Shape: [B, 3, T, H, W] - camera origins in world space
            ray_d: Ray directions for each pixel (normalized).
                Shape: [B, 3, T, H, W] - ray directions in world space
            ts: Timestamps for temporal encoding.
                Shape: [B, 1, T, H, W] - normalized time values
            output_head: Optional output head to override self.output_head.
                Default: None (uses self.output_head)
        
        Returns:
            Dict containing 4D Gaussian parameters:
                - xyz: 3D positions [B, T, H, W, 3] - world space positions
                - t: Temporal positions [B, T, H, W, 1] - perturbed timestamps
                - depth: Depth values [B, T, H, W, 1] - distance along ray
                - feature: Appearance features [B, T, H, W, C_feat]
                - opacity: Opacity values [B, T, H, W, 1] - alpha channel
                - scaling: 3D scale factors [B, T, H, W, 3] - Gaussian scales
                - rotation: Rotation quaternions [B, T, H, W, 4] - Gaussian orientations
                - ms3: Marginal scale factors [B, T, H, W, 1] - 4D scale
                - cov_t: Temporal covariance [B, T, H, W, 1] - temporal extent
                - omega: Angular velocity [B, T, H, W, 3] - rotation speed
                - dxyzt: Position perturbations [B, T, H, W, 4] - residuals
        """
        # Stage 1: Initial decoding from tokens to raw parameters
        gs_params = self.decode_wo_act(token, ray_d, output_head)  # [B, T, H, W, X] raw parameters
        
        # Stage 2: Prepare and add ray information to parameters
        gs_params["ray_o"] = rearrange(ray_o.float(), 'b c t h w -> b t h w c')  # [B, 3, T, H, W] -> [B, T, H, W, 3]
        gs_params["ray_d"] = rearrange(ray_d.float(), 'b c t h w -> b t h w c')  # [B, 3, T, H, W] -> [B, T, H, W, 3]
        gs_params["ts"] = rearrange(ts.float(), 'b c t h w -> b t h w c')  # [B, 1, T, H, W] -> [B, T, H, W, 1]
        
        # Stage 3: Apply magic filter for spatial-temporal consistency
        gs_params = self.magic_filter(gs_params)  # Smooth parameters

        # Extract filtered parameters
        depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = (
            gs_params["depth"],  # [B, T, H, W, 1] depth along ray
            gs_params["opacity"],  # [B, T, H, W, 1] opacity
            gs_params["scaling"],  # [B, T, H, W, 3] 3D scales
            gs_params["rotation"],  # [B, T, H, W, 4] quaternions
            gs_params["dxyzt"],  # [B, T, H, W, 4] position residuals
            gs_params["ms3"],  # [B, T, H, W, 1] marginal scale
            gs_params["cov_t"],  # [B, T, H, W, 1] temporal covariance
            gs_params["omega"],  # [B, T, H, W, 3] angular velocity
            gs_params["feature"],  # [B, T, H, W, C_feat] appearance
        )

        ray_o, ray_d, ts = gs_params["ray_o"], gs_params["ray_d"], gs_params["ts"]

        # Stage 4: Apply activation functions and rescaling
        depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = (
            auto_grad_checkpoint(
                self.rescale_predictions,
                depth,
                opacity,
                scale,
                rotation,
                dxyzt,
                ms3,
                cov_t,
                omega,
                feat,
            )
        )  # Apply proper activations and ranges to all parameters

        # Stage 5: Construct final output dictionary
        gs_params = {}
        gs_params["depth"] = depth  # [B, T, H, W, 1]
        gs_params["feature"] = feat  # [B, T, H, W, C_feat]
        gs_params["opacity"] = opacity  # [B, T, H, W, 1]
        gs_params["scaling"] = scale  # [B, T, H, W, 3]
        gs_params["rotation"] = rotation  # [B, T, H, W, 4]

        # Stage 6: Compute 3D world positions from depth and rays
        gs_params["xyz"] = gs_params["depth"] * ray_d + ray_o  # [B, T, H, W, 3] world positions

        # Stage 7: Add 4D Gaussian temporal parameters
        gs_params["ms3"] = ms3  # [B, T, H, W, 1] marginal scale
        gs_params["cov_t"] = cov_t  # [B, T, H, W, 1] temporal covariance
        gs_params["omega"] = omega  # [B, T, H, W, 3] angular velocity

        # Stage 8: Apply spatial-temporal perturbations
        gs_params["dxyzt"] = dxyzt  # [B, T, H, W, 4] position residuals
        gs_params["xyz"] = gs_params["xyz"] + gs_params["dxyzt"][..., :3]  # [B, T, H, W, 3] perturbed positions
        gs_params["t"] = ts + gs_params["dxyzt"][..., 3:]  # [B, T, H, W, 1] perturbed time

        # Stage 9: Ensure float32 precision for all outputs
        for k in gs_params:
            gs_params[k] = gs_params[k].float()  # Convert to float32

        return gs_params

    def encode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        input_linear=None,
        concat_norm=None,
    ):
        """Encode RGB images with camera parameters into feature tokens.
        
        This function processes input RGB images along with camera parameters to create
        feature tokens for the transformer. It combines visual features from DINOv2,
        Plucker coordinates for ray encoding, and temporal information into a unified
        representation.
        
        Pipeline:
        1. Extract visual features using DINOv2 backbone
        2. Compute Plucker coordinates for camera rays
        3. Patchify inputs into spatial-temporal patches
        4. Concatenate all features and project to embedding dimension
        
        Args:
            x: RGB images.
                Shape: [B, T, 3, H, W] - batch of temporal sequences
            t: Normalized timestamps for each frame.
                Shape: [B, T, 1] - time values in [0, 1]
            K: Camera intrinsic matrices.
                Shape: [B, T, 3, 3] - focal length and principal point
            c2w: Camera-to-world transformation matrices.
                Shape: [B, T, 4, 4] - camera poses in world space
            input_linear: Optional linear projection layer.
                Default: None (uses self.input_linear)
            concat_norm: Optional normalization layer.
                Default: None (uses self.concat_norm)
        
        Returns:
            Tuple containing:
                - x: Encoded feature tokens [B, N_patches, C_embed]
                    where N_patches = (T*H*W)/(Tp*Hp*Wp)
                - rgb: Original RGB values [B, T, H, W, 3]
                - ray_o: Ray origins [B, 3, T, H, W]
                - ray_d: Ray directions [B, 3, T, H, W]
                - ts: Expanded timestamps [B, 1, T, H, W]
        """
        # Extract dimensions and prepare RGB
        B, T, _, H, W = x.shape  # [B, T, 3, H, W] batch, time, channels, height, width
        Hp, Wp, Tp = self.patch_size  # Spatial and temporal patch sizes
        Nt = T // Tp  # Number of temporal patches
        Nh = H // Hp  # Number of height patches
        Nw = W // Wp  # Number of width patches
        rgb = rearrange(x, 'b t c h w -> b t h w c')  # [B, T, H, W, 3] for output

        # Compute Plucker coordinates for ray representation
        plucker, ray_o, ray_d = self.plucker(H, W, K, c2w)  # [B, 6, T, H, W] Plucker coords
        plucker = plucker.to(dtype=self.dtype)  # Match model precision

        # Prepare temporal information
        ts = rearrange(t, 'b t c -> b c t')  # [B, 1, T] transpose for broadcasting
        ts = repeat(ts, 'b c t -> b c t h w', h=H, w=W)  # [B, 1, T, H, W] expand to spatial dims
        p = torch.cat([ts, plucker], dim=1)  # [B, 7, T, H, W] combine time + Plucker

        # Extract and concatenate with visual features using DINOv2
        # Resize if patch size doesn't match DINOv2's native 14x14
        if Hp != 14 or Wp != 14:
            f = F.interpolate(
                input=rearrange(x, 'b t c h w -> (b t) c h w'),  # [B*T, 3, H, W] flatten batch-time
                size=(H // Hp * 14, W // Wp * 14),  # Resize to match 14x14 patches
                mode="bilinear",
            )  # [B*T, 3, H', W'] resized for DINOv2
        else:
            f = rearrange(x, 'b t c h w -> (b t) c h w')  # [B*T, 3, H, W] merge batch-time
        
        # Extract features from DINOv2
        f = self.dinov2(f, is_training=True)["x_prenorm"]  # [B*T, N_tokens, C_dino] visual features
        n_reg = f.shape[1] - Nh * Nw  # Number of register tokens
        f = f[:, n_reg:]  # [B*T, Nh*Nw, C_dino] remove register tokens

        # Reshape DINOv2 features for temporal patches
        f = rearrange(
            f,
            "(B N_t T_p) (N_h N_w) C_xxx -> B (N_t N_h N_w) (C_xxx T_p)",
            T_p=Tp,  # Temporal patch size
            N_t=Nt,  # Number of temporal patches
            N_h=Nh,  # Number of height patches
            N_w=Nw,  # Number of width patches
        )  # [B, Nt*Nh*Nw, C_dino*Tp] grouped temporal features

        # Combine RGB and Plucker-time coordinates
        x = rearrange(x, 'b t c h w -> b c t h w')  # [B, 3, T, H, W] channel first
        x = torch.cat([x, p], dim=1)  # [B, 10, T, H, W] RGB + time + Plucker

        # Patchify combined features
        x = self.patchify(x)  # [B, N_patches, C_patch] spatial-temporal patches
        x = torch.cat([x, f], dim=-1)  # [B, N_patches, C_patch + C_dino*Tp] all features

        # Project to embedding dimension
        input_linear = input_linear if input_linear is not None else self.input_linear
        concat_norm = concat_norm if concat_norm is not None else self.concat_norm
        x = auto_grad_checkpoint(input_linear, x)  # [B, N_patches, C_embed] project to 768-d
        x = auto_grad_checkpoint(concat_norm, x)  # [B, N_patches, C_embed] normalize
        
        return x, rgb, ray_o, ray_d, ts

    def fusion(self, x: torch.Tensor, c: torch.Tensor = None, blocks=None, norm=None):
        # B, TOK+1, C
        c = c if c is not None else self.cls_token
        c = c.expand(x.shape[0], -1, -1)
        c = c.type(self.dtype)
        x = torch.cat([x, c], dim=1)
        blocks = blocks if blocks is not None else self.blocks
        # Strangely, still spiking memory usage even after using gradient checkpointing
        for _, block in enumerate(blocks):
            x = auto_grad_checkpoint(block, x)
        norm = norm if norm is not None else self.norm
        x = auto_grad_checkpoint(norm, x).float()
        x, c = x.split([x.shape[1] - 1, 1], dim=1)  # extract cls_token
        return x, c

    def forward_global(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        input_linear=None,
        concat_norm=None,
        blocks=None,
        norm=None,
        output_head=None,
        cls_token=None,
        extra_feat=None,
        extra_linear=None,
        return_feat=False,
        return_feat_and_cls=False,
        **kwargs,
    ):
        """
        Forward pass with global attention across all frames.
        
        This method processes input images through the full pipeline:
        1. Encodes input images into feature tokens
        2. Applies global self-attention across all spatial-temporal patches
        3. Decodes features to Gaussian splatting parameters
        
        The "global" designation means all patches attend to each other,
        enabling coherent temporal modeling across the entire sequence.
        
        Args:
            x: Input images
                Shape: [B, T, 3, H, W]
                B: Batch size, T: Temporal frames, 3: RGB channels
                H: Image height, W: Image width
            t: Timestamps for each frame
                Shape: [B, T, 1]
                Normalized timestamps, typically in [0, 1] range
            K: Camera intrinsic matrices
                Shape: [B, T, 3, 3]
                Contains focal lengths (fx, fy) and principal points (cx, cy)
            c2w: Camera-to-world transformation matrices
                Shape: [B, T, 4, 4]
                4x4 homogeneous transforms from camera to world coordinates
            input_linear: Optional input projection layer (uses self.input_linear if None)
            concat_norm: Optional normalization after feature concatenation
            blocks: Transformer blocks (uses self.blocks if None)
            norm: Final normalization layer (uses self.norm if None)  
            output_head: Output projection for Gaussians (uses self.head if None)
            cls_token: Optional learnable class token for aggregation
            extra_feat: Additional features to incorporate
            extra_linear: Projection for extra features
            return_feat: If True, returns intermediate features
            return_feat_and_cls: If True, returns features and class token
            
        Returns:
            gs_params: Dictionary containing Gaussian parameters:
                - xyz: 3D positions
                - rgb: Color values
                - opacity: Opacity/alpha values
                - scales: 3D scale factors
                - rotations: Rotation quaternions
            glo_feat (optional): Global features [B, N_tokens, C] if return_feat=True
            cls_token (optional): Class token [B, 1, C] if return_feat_and_cls=True
        """
        # Encode input images to feature tokens and rays
        img_feat, rgb, ray_o, ray_d, ts = self.encode(
            x, t, K, c2w, input_linear, concat_norm
        )  
        # img_feat: [B, N_tokens, C] - Encoded image features
        # rgb: [B, T, H, W, 3] - Original RGB values
        # ray_o: [B, 3, T, H, W] - Ray origins in world space
        # ray_d: [B, 3, T, H, W] - Ray directions in world space  
        # ts: [B, T, 1] - Timestamps

        # Incorporate additional features if provided (not used.) 
        if extra_feat is not None and extra_linear is not None:
            img_feat = torch.cat([img_feat, extra_feat], dim=-1)  # [B, N_tokens, C+C_extra]
            img_feat = extra_linear(img_feat)  # [B, N_tokens, C] - Project back to original dim

        # Configure 3D rotary position embeddings
        blocks = blocks if blocks is not None else self.blocks
        B, T, _, H, W = x.shape
        Hp, Wp, Tp = self.patch_size  # Patch dimensions for height, width, time
        Nt = T // Tp   # Number of temporal patches
        Nh = H // Hp   # Number of height patches  
        Nw = W // Wp   # Number of width patches
        
        # Update rotary embedding grid dimensions for each transformer block
        for b in blocks:
            if hasattr(b.attn, "rotary_emb"):
                b.attn.rotary_emb.D = Nt  # Depth/Temporal dimension
                b.attn.rotary_emb.H = Nh  # Height dimension
                b.attn.rotary_emb.W = Nw  # Width dimension

        # Process through transformer blocks with global attention
        glo_feat, cls_token = self.fusion(
            img_feat, cls_token, blocks=blocks, norm=norm
        )  # glo_feat: [B, N_tokens, C], cls_token: [B, 1, C] or None

        # Decode features to Gaussian parameters
        gs_params = self.decode(
            glo_feat, ray_o, ray_d, ts, output_head
        )  # Dictionary with Gaussian parameters (xyz, rgb, opacity, scales, etc.)

        # Return based on requested outputs
        if return_feat_and_cls:
            return gs_params, glo_feat, cls_token
        if return_feat:
            return gs_params, glo_feat
        else:
            return gs_params


    def forward_global_local_detail(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        **kwargs,
    ):
        """
        Forward pass for three-level hierarchical processing (global, local, detail).
        
        Processes the input at three different resolutions:
        - Global: Coarse features for overall structure
        - Local: Medium resolution for regional details  
        - Detail: Fine resolution for high-frequency details
        
        Args:
            x: Input images
                Shape: [B, T, 3, H, W]
            t: Timestamps for each frame
                Shape: [B, T, 1]
            K: Camera intrinsic matrices
                Shape: [B, T, 3, 3]
            c2w: Camera-to-world transformation matrices
                Shape: [B, T, 4, 4]
                
        Returns:
            Gaussian parameters at global resolution
        """
        # Extract downsampling factors
        Hd, Wd, Td = self.global_downsample  # Height, Width, Temporal downsampling
        
        # Prepare features for global attention in temporal domain
        x_g = x[:, ::Td, :, ::Hd, ::Wd]  # [B, T, 3, H, W] -> [B, T/Td, 3, H/Hd, W/Wd]
        t_g = t[:, ::Td]  # [B, T, 1] -> [B, T/Td, 1]
        K_g = K[:, ::Td].clone()  # [B, T, 3, 3] -> [B, T/Td, 3, 3] (clone to avoid in-place modification)
        K_g[..., 0:1, :] = K_g[..., 0:1, :] / Wd  # Adjust fx for downsampled width
        K_g[..., 1:2, :] = K_g[..., 1:2, :] / Hd  # Adjust fy for downsampled height
        c2w_g = c2w[:, ::Td]  # [B, T, 4, 4] -> [B, T/Td, 4, 4]
        
        # Prepare features for global attention within each local split (self.local_split)
        x_d = x[:, ::Td, :, ::Hd//2, ::Wd//2]  # [B, T, 3, H, W] -> [B, T/Td, 3, H/(Hd/2), W/(Wd/2)]
        t_d = t[:, ::Td]  # [B, T, 1] -> [B, T/Td, 1]
        K_d = K[:, ::Td].clone()  # [B, T, 3, 3] -> [B, T/Td, 3, 3]
        K_d[..., 0:1, :] = K_d[..., 0:1, :] / (Wd // 2)
        K_d[..., 1:2, :] = K_d[..., 1:2, :] / (Wd // 2)
        c2w_d = c2w[:, ::Td]

        # Split batch into local groups for frame-wise attention
        Ls = self.local_split  # Number of local splits
        x = rearrange(x, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls)  # [B, T, ...] -> [B*Ls, T/Ls, ...]
        t = rearrange(t, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls)  # [B, T, 1] -> [B*Ls, T/Ls, 1]
        K = rearrange(K, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls)  # [B, T, 3, 3] -> [B*Ls, T/Ls, 3, 3]
        c2w = rearrange(c2w, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls)  # [B, T, 4, 4] -> [B*Ls, T/Ls, 4, 4]

        gs_params_l, glo_feat, cls_token = self.forward_global(
            x,
            t,
            K,
            c2w,
            return_feat_and_cls=True,
        )

        # Convert to frame attention
        x_d = rearrange(x_d, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 2)
        t_d = rearrange(t_d, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 2)
        K_d = rearrange(K_d, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 2)
        c2w_d = rearrange(c2w_d, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 2)

        cls_token = cls_token.sum(0, keepdim=True).expand(
            x_d.shape[0], *cls_token.shape[1:]
        )

        gs_params_d, glo_feat, cls_token = self.forward_global(
            x_d,
            t_d,
            K_d,
            c2w_d,
            input_linear=self.detail_linear,
            concat_norm=self.detail_concat_norm,
            blocks=self.details,
            norm=self.detail_norm,
            output_head=self.detail_head,
            cls_token=cls_token,
            return_feat_and_cls=True,
        )

        # Prepare features for frame attention
        x_g = rearrange(x_g, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 4)
        t_g = rearrange(t_g, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 4)
        K_g = rearrange(K_g, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 4)
        c2w_g = rearrange(c2w_g, "B (Ls Tx) ... -> (B Ls) Tx ...", Ls=Ls // 4)

        cls_token = cls_token.sum(0, keepdim=True).expand(
            x_g.shape[0], *cls_token.shape[1:]
        )

        gs_params_g, glo_feat, cls_token = self.forward_global(
            x_g,
            t_g,
            K_g,
            c2w_g,
            input_linear=self.global_linear,
            concat_norm=self.global_concat_norm,
            blocks=self.globals,
            norm=self.global_norm,
            output_head=self.global_head,
            cls_token=cls_token.sum(0, keepdim=True),
            return_feat_and_cls=True,
        )

        # concatenate the tokens at different space-time scale
        gs_params = dotdict()
        for k in list(gs_params_g.keys()):
            B, _, _, _, C = gs_params_g[k].shape
            # B, T // Td, H // Hd, W // Wd, C
            g = gs_params_g[k].reshape(B, -1, C)
            # BT, 1, Hx, Wx, C
            d = gs_params_d[k].reshape(B, -1, C)  # noqa: E741
            # BT, 1, Hx, Wx, C
            l = gs_params_l[k].reshape(B, -1, C)  # noqa: E741
            gs_params[k] = torch.cat([g, d, l], dim=-2)


        return gs_params

    def magic_filter(self, gs_params: dotdict, magic_pattern=None):  # noqa: C901
        magic_pattern = (
            magic_pattern if magic_pattern is not None else self.magic_pattern
        )
        Mn = self.magic_num
        magic_pattern = [magic_pattern[0][:Mn], magic_pattern[1][:Mn]]

        Hp, Wp, Tp = self.patch_size
        if not self.recalc_magic_pattern:
            # Downsample gs_params to get the number of points smaller for rendering
            if np.prod([len(m) for m in magic_pattern]) > 1:
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val, "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw Hp Wp C", Hp=Hp, Wp=Wp
                    )
                    val = val[
                        :, :, :, :, magic_pattern[0], magic_pattern[1], :
                    ]  # sample the 10 important points
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
        else:
            if self.recalc_method == "patch_sorting":
                # Recalculate magic pattern for each patch
                Hp, Wp, Tp = self.patch_size
                occ = gs_params["opacity"].float()
                occ = rearrange(
                    occ, "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C", Hp=Hp, Wp=Wp
                )

                val, ind = occ.topk(
                    len(magic_pattern[0]), dim=-2
                )  # B, T, Nh, Nw, 10, 1
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C",
                        Hp=Hp,
                        Wp=Wp,
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, Nh, Nw, 10, C
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
            elif self.recalc_method == "patch_random":
                # Fixed patch random pattern
                # Recalculate magic pattern for each patch
                Hp, Wp, Tp = self.patch_size
                ind = self.random_pattern[None, None, None, None, :, None]  # 10, 1

                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C",
                        Hp=Hp,
                        Wp=Wp,
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, Nh, Nw, 10, C
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
            elif self.recalc_method == "global_random":
                # Fixed global random pattern
                # Recalculate magic pattern for each patch
                ind = self.random_pattern[None, None]  # N, 1

                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T H W C -> B T (H W) C",
                    )
                    max_rand_val = val.shape[-2]
                    ind = ind % max_rand_val  # B, T, N, 1
                    val = multi_gather(val, ind, dim=-2)  # B, T, X, C
                    val = rearrange(val, "B T X C -> B T X 1 C")
                    gs_params[key] = val
            elif self.recalc_method == "global_sorting":
                # Recalculate magic pattern for each patch
                Hp, Wp, Tp = self.patch_size
                occ = gs_params["opacity"].float()
                occ = rearrange(occ, "B T H W C -> B T (H W) C")

                val, ind = occ.topk(len(self.random_pattern), dim=-2)  # B, T, X, 1
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T H W C -> B T (H W) C",
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, X, C
                    val = rearrange(val, "B T X C -> B T X 1 C")
                    gs_params[key] = val
            elif self.recalc_method == "uniform":
                # Fixed global random pattern
                # Recalculate magic pattern for each patch
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    gs_params[key] = gs_params[key][..., ::4, ::5, :]  # 1 / 20
            else:
                raise NotImplementedError(
                    f"Recalc method {self.recalc_method} not implemented"
                )

        return gs_params

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        **kwargs,
    ):
        """
        Forward pass of the Temporal Level of Detail encoder.
        
        Args:
            x: Input images tensor
                Shape: [B, T, 3, H, W]
                B: Batch size
                T: Number of temporal frames/views
                3: RGB channels
                H: Image height
                W: Image width
            t: Timestamps for each frame
                Shape: [B, T, 1]
                Values typically normalized to [0, 1] or similar range
            K: Camera intrinsic matrices
                Shape: [B, T, 3, 3]
                Standard 3x3 intrinsic matrix for each camera
            c2w: Camera-to-world transformation matrices
                Shape: [B, T, 4, 4]
                4x4 homogeneous transformation from camera to world coordinates
                
        Returns:
            Encoded Gaussian parameters for 4D reconstruction
        """

        if self.upsample_ratio != 1.0:
            # Resize input images and parameters
            B, T, C, H, W = x.shape
            Ht = int(H * self.upsample_ratio // 56 * 56 + 0.5)  # Ensure height is divisible by 56
            Wt = int(W * self.upsample_ratio // 56 * 56 + 0.5)  # Ensure width is divisible by 56

            # Adjust intrinsics for new resolution
            K[..., :2, :1] *= Wt / W  # [B, T, 3, 3] Scale fx
            K[..., :2, 1:] *= Ht / H  # [B, T, 3, 3] Scale fy
            
            # Resize images to target resolution
            x = F.interpolate(
                x.reshape(B * T, C, H, W),  # [B, T, 3, H, W] -> [B*T, 3, H, W]
                size=(Ht, Wt),
                mode="bilinear",
                align_corners=False,
            )  # [B*T, 3, Ht, Wt]
            x = x.reshape(B, T, *x.shape[1:])  # [B*T, 3, Ht, Wt] -> [B, T, 3, Ht, Wt]

        # Process through appropriate level of detail
        if self.n_levels == 1:
            gs_params = self.forward_global(
                x, t, K, c2w, **kwargs
            )  # -> [B, N_gaussians, param_dim]
        else:
            gs_params = self.forward_global_local_detail(
                x, t, K, c2w, **kwargs
            )  # -> [B, N_gaussians, param_dim]

        return gs_params


@MODELS.register_module("ImageEncoderTLoD-B")
def ImageEncoderTLoD_B(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        raise NotImplementedError("Pretrained model loading not yet implemented")
    config = TLoDConfig(**kwargs)
    return ImageEncoderTLoD(config)
