# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Mapping

import torch

from gsplat.rendering import rasterization
from torch import nn

from ..acceleration.checkpoint import auto_grad_checkpoint

from ..easyvolcap.utils.math_utils import affine_inverse
from ..registry import RENDERER
from .gaussian_renderer import GaussianRenderer


@torch.jit.script
def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)


class SingleImageRasterization(nn.Module):
    def __init__(self, height: int, width: int, znear: float, zfar: float):
        super().__init__()
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar

    def forward(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
    ):
        out_img, out_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width if width else self.width,
            height=height if height else self.height,
            near_plane=self.znear,
            far_plane=self.zfar,
            # backgrounds=bg,
            # render_mode=("RGB+ED" if means.requires_grad else "RGB+ED")
            render_mode=render_mode.replace("MONO", "RGB"),
        )  # H, W, C since we're only rendering one image at a time
        if out_img.shape[-1] == 4 or out_img.shape[-1] == 2 or "D" in render_mode:
            out_dpt = out_img[..., -1:]  # 1, H, W, C
            out_img = out_img[..., :-1]  # 1, H, W, C
        else:
            out_dpt = torch.zeros_like(out_alpha)

        return out_img, out_alpha, out_dpt


class SingleBatchRasterization(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        znear: float,
        zfar: float,
        marginal_th: float = 0.05,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar
        self.single_image_rasterizer = SingleImageRasterization(
            height, width, znear, zfar
        )
        self.marginal_th = marginal_th

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
    ):
        B, T, _, _ = w2cs.shape

        # B, T, 3, 3 & B, T, 4, 4 for camera parameters
        out_imgs = []
        out_alphas = []
        out_dpts = []
        for b in range(B):
            # Perform marginalization given the input timestamps before rasterization
            # Compute updated opacity
            # Compute updated center (xyz)
            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            for i in range(T):
                means = xyz[b].float() + (ms3[b] * (ts[b, i] - t[b])).float()
                quats = rotation[b].float()
                scales = scale[b].float()
                opacities = (
                    opacity[b].float()
                    * compute_marginal_t(ts[b, i], t[b], cov_t[b]).float()
                )[..., 0]
                colors = (
                    rgb[b, ..., :3].float()
                    if "RGB" in render_mode
                    else rgb[b, ..., 3:].float()
                )
                viewmats = w2cs[b, i : i + 1].float()
                ixts = Ks[b, i : i + 1].float()

                out_img, out_alpha, out_dpt = auto_grad_checkpoint(
                    self.single_image_rasterizer,
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                    viewmats,
                    ixts,
                    height,
                    width,
                    render_mode,
                )

                out_imgs_batch.append(out_img)
                out_alphas_batch.append(out_alpha)
                out_dpts_batch.append(out_dpt)

            out_img = torch.cat(out_imgs_batch)
            out_alpha = torch.cat(out_alphas_batch)
            out_dpt = torch.cat(out_dpts_batch)

            out_img = out_img.permute(0, 3, 1, 2)  # V, C, H, W
            out_img = torch.clamp(2 * out_img - 1, -1, 1)
            out_alpha = out_alpha.permute(0, 3, 1, 2)  # V, C, H, W
            out_dpt = out_dpt.permute(0, 3, 1, 2)  # V, C, H, W

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )

        return img, mask, depth


@RENDERER.register_module()
class GaussianRenderer4D(GaussianRenderer):
    def __init__(
        self,
        height=512,
        width=512,
        znear=0.01,
        zfar=500,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.znear = znear
        self.zfar = zfar
        self.single_batch_rasterizer = SingleBatchRasterization(
            height, width, znear, zfar
        )

    def __call__(
        self,
        gs_params: Mapping[str, torch.Tensor],
        ts: torch.Tensor,
        Ks: torch.Tensor,
        RTs: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        **kwargs,
    ):
        # B, N, C for params
        xyz, rgb, scale, rotation, opacity = (
            gs_params["xyz"],
            gs_params["feature"],  # assume last channel is for monochrome
            gs_params["scaling"],
            gs_params["rotation"],
            gs_params["opacity"],
        )

        t, cov_t, ms3 = (
            gs_params["t"],
            gs_params["cov_t"],
            gs_params["ms3"],
        )

        w2cs = affine_inverse(RTs)  # B, V, 4, 4

        imgs = []
        masks = []
        depths = []
        B = ts.shape[0]
        for b in range(B):
            img, mask, depth = self.single_batch_rasterizer(
                xyz[b : b + 1],
                rgb[b : b + 1],
                scale[b : b + 1],
                rotation[b : b + 1],
                opacity[b : b + 1],
                t[b : b + 1],
                cov_t[b : b + 1],
                ms3[b : b + 1],
                w2cs[b : b + 1],
                Ks[b : b + 1],
                ts[b : b + 1],
                height=height,
                width=width,
                render_mode=render_mode,
            )
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
        img = torch.cat(imgs)
        mask = torch.cat(masks)
        depth = torch.cat(depths)

        # output = {"rgb": img, "mask": mask, "depth": depth, "dense_depth": dense_depth}
        output = {"rgb": img, "mask": mask, "depth": depth}
        if "MONO" in render_mode:
            output["mono"] = output.pop("rgb")
        return output
