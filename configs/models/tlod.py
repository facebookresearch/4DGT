# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Define model
# DINOv2 4DV with late fusion for queries and features
model = dict(  # noqa: C408
    type="GaussianModelModule",
)

image_encoder = dict(  # noqa: C408
    type="ImageEncoderTLoD-B",
    patch_size=(14, 14, 1),
    input_size=(504, 504, 64),
    hidden_size=1536,  # token_dim
    depth=12,
    num_heads=16,
    mlp_ratio=4.0,
    mlp_dim=256,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    use_dpt_head=False,
    use_sigmoid_ms3=True,
    use_sigmoid_omega=True,
    use_prenorm_dinov2=True,
    use_split_linear_head=True,
    color_space="rgbm",
    ms3_deg=4,
    omega_deg=2,
    dxyzt_min=-0.5,
    dxyzt_max=0.5,
    cov_t_min=0.1,  # at least span the whole sequence wo disappearing?
    cov_t_max=10000.0,  # much more relaxed for static scenes
    omega_clamp=0.0001,  # 1e-5
    ms3_clamp=0.0001,  # 1e-5
    sigmoid_ms3_max=2.0,
    sigmoid_omega_max=2.0,
    ms3_deg_downmax_mult=8.0,  # reduce max value for higher degrees
    omega_deg_downmax_mult=8.0,  # reduce max value for higher degrees
    n_levels=1,
)

renderer = dict(  # noqa: C408
    type="TLODRenderer",
    height=504,
    width=504,
    use_2dgs=True,
    vis_flow=False,  # FIXME: find a way to make this work with monochrome cameras too
    vis_motion=False,  # FIXME: make this faster
)