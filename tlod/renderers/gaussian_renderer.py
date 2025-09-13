# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Mapping

import torch

from gsplat.rendering import rasterization

from ..easyvolcap.utils.math_utils import affine_inverse

from ..registry import RENDERER


@RENDERER.register_module()
class GaussianRenderer:
    def __init__(
        self,
        height=512,
        width=512,
        bg_color=(1.0, 1.0, 1.0),
        znear=0.01,
        zfar=500,
    ):
        self.height = height
        self.width = width
        self.bg_color = bg_color
        self.znear = znear
        self.zfar = zfar

    def __call__(
        self,
        gs_params: Mapping[str, torch.Tensor],
        ts: torch.Tensor,
        Ks: torch.Tensor,
        RTs: torch.Tensor,
        height: int = 0,
        width: int = 0,
    ):
        # B, N, C for params
        xyz, rgb, scale, rotation, opacity = (
            gs_params["xyz"],
            gs_params["feature"],
            gs_params["scaling"],
            gs_params["rotation"],
            gs_params["opacity"],
        )

        w2cs = affine_inverse(RTs)  # B, V, 4, 4
        B, V, _, _ = w2cs.shape

        # B, V, 3, 3 & B, V, 4, 4 for camera parameters
        out_imgs = []
        out_alphas = []

        for b in range(B):
            out_img, out_alpha, _ = rasterization(
                means=xyz[b].float(),
                quats=rotation[b].float(),
                scales=scale[b].float(),
                opacities=opacity[b, ..., 0].float(),
                colors=rgb[b].float(),
                viewmats=w2cs[b].float(),
                Ks=Ks[b].float(),
                width=width if width else self.width,
                height=height if height else self.height,
                near_plane=self.znear,
                far_plane=self.zfar,
                backgrounds=torch.as_tensor(self.bg_color)
                .to("cuda", non_blocking=True)[None]
                .expand(V, -1),
            )

            out_img = out_img.permute(0, 3, 1, 2)  # V, C, H, W
            out_img = torch.clamp(2 * out_img - 1, -1, 1)
            out_alpha = out_alpha.permute(0, 3, 1, 2)  # V, C, H, W

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)

        return {"rgb": torch.stack(out_imgs), "mask": torch.stack(out_alphas)}
