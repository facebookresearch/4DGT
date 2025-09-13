# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Mapping

import torch
from gsplat.rendering import rasterization, rasterization_2dgs

from ..acceleration.checkpoint import auto_grad_checkpoint, dummy_function
from ..easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from ..easyvolcap.utils.console_utils import dotdict
from ..easyvolcap.utils.flow_utils import flow_to_color  # noqa: F401
from ..easyvolcap.utils.math_utils import affine_inverse
from ..easyvolcap.utils.quat_utils import angle_axis_to_quaternion, qmul
from ..registry import RENDERER
from .gaussian_renderer import GaussianRenderer


@torch.jit.script
def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)


def threedgs_rasterizer(
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
    out_img, out_alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        height=height,
        width=width,
        # backgrounds=bg,
        # render_mode=("RGB+ED" if means.requires_grad else "RGB+ED")
        render_mode=render_mode.replace("MONO", "RGB"),
    )  # H, W, C since we're only rendering one image at a time
    if out_img.shape[-1] == 4 or out_img.shape[-1] == 2 or "D" in render_mode:
        out_dpt = out_img[..., -1:]  # 1, H, W, C
        out_img = out_img[..., :-1]  # 1, H, W, C
    else:
        out_dpt = torch.zeros_like(out_alpha)

    return (
        out_img,
        out_alpha,
        out_dpt,
        dotdict(visible=meta["radii"] > 0),
    )  # not differentiable
    # return out_img, out_alpha, out_dpt, meta


def twodgs_rasterizer(
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
    # HACK: When having more than 8 channels in rgb, distorts and median_depths becomes strange
    channels = colors.shape[-1]
    if "D" in render_mode:
        channels += 1
    padded_channels = 0
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=colors.device),
            ],
            dim=-1,
        )
    # bg = torch.zeros(len(Ks), colors.shape[-1], device=colors.device)
    # Add camera dimension to colors to match expected shape [C, N, D]
    if colors.dim() == 2:  # [N, D]
        colors = colors.unsqueeze(0)  # [1, N, D]
    
    colors, alphas, normals, surf_normals, distorts, median_depths, meta = (
        rasterization_2dgs(
            means=means,    # N, 3
            quats=quats,    # N, 4
            scales=scales,  # N, 3
            opacities=opacities,  # N
            colors=colors,  # C, N, D (where C=1 for single view)
            viewmats=viewmats,  # V, 4, 4
            Ks=Ks,  # V, 3, 3
            height=height,
            width=width,
            # backgrounds=bg,
            # render_mode=("RGB+ED" if means.requires_grad else "RGB+ED")
            # render_mode="RGB",  # always use median depth
            render_mode=render_mode.replace("MONO", "RGB"),
            # distloss="D" in render_mode,
            # depth_mode="median",  # use median depth for surface normal calculation
        )
    )  # H, W, C since we're only rendering one image at a time
    if "D" in render_mode:
        depths = colors[..., -1:]  # 1, H, W, C
        colors = colors[..., :-1]  # 1, H, W, C
    else:
        depths = median_depths

    if padded_channels > 0:
        colors = colors[..., :-padded_channels]

    # Rendered normal is in world space, convert back to camera space
    # viewmats: 4, 4, world to camera
    # right multiply the tranposed of viewmats to convert world space to camera space
    normals = normals @ viewmats[..., :3, :3].mT
    flipper = torch.as_tensor([-1, -1, -1]).to(means.device, non_blocking=True)
    normals = normals * flipper

    if surf_normals is not None:
        surf_normals = surf_normals[None]  # 1, H, W, 3
        surf_normals = surf_normals @ viewmats[..., :3, :3].mT
        surf_normals = surf_normals * flipper
    else:
        surf_normals = normals

    colors = torch.cat(
        [colors, normals, surf_normals, distorts, median_depths], dim=-1
    )  # to retain gradient

    return (
        colors,
        alphas,
        depths,
        dotdict(),
    )


def compute_means(
    xyz: torch.Tensor,
    t: torch.Tensor,
    ms3: torch.Tensor,
    w2cs: torch.Tensor,
    ts: torch.Tensor,
    b: int = 0,
    i: int = 0,
):
    dt = (ts[b, i] - t[b]).float()
    ms3_deg = ms3.shape[-1] // 3
    dmeans = torch.stack(
        [ms3[b, ..., i * 3 : (i + 1) * 3] * dt ** (i + 1) for i in range(ms3_deg)]
    ).sum(0)
    means = xyz[b].float() + dmeans
    viewmats = w2cs[b, i : i + 1].float()  # 1, 3, 4
    next_means = means @ viewmats[0, :3, :3].mT + viewmats[0, :3, -1]
    return next_means


def fourdgs_rasterizer(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    t: torch.Tensor,
    cov_t: torch.Tensor,
    ms3: torch.Tensor,  # B, N, X3
    omega: torch.Tensor,  # B, N, X3
    w2cs: torch.Tensor,  # B, T, 4, 4
    Ks: torch.Tensor,
    ts: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB+ED",
    b: int = 0,
    i: int = 0,
    use_2dgs: bool = False,
    vis_motion: bool = False,  # heavy on computation, thus skip if possible
    vis_flow: bool = False,  # heavy on computation, thus skip if possible
    marginal_th: float = 0.05,
    opacity_th: float = 0.0001,
    motion_mask_th: float = 0.25,  # velocity or angular velocity threshold for filtering actual motion
    retain_all_rgb: bool = False,  # retain all RGB channels, not just the first 3
):
    # Prepare time difference for movements
    dt = (ts[b, i] - t[b]).float()

    # Compute updated location
    ms3_deg = ms3.shape[-1] // 3
    dmeans = torch.stack(
        [ms3[b, ..., i * 3 : (i + 1) * 3] * dt ** (i + 1) for i in range(ms3_deg)]
    ).sum(0)
    means = xyz[b].float() + dmeans

    # Compute updated rotation
    omega_deg = omega.shape[-1] // 3
    domega = torch.stack(
        [omega[b, ..., i * 3 : (i + 1) * 3] * dt ** (i + 1) for i in range(omega_deg)]
    ).sum(0)
    dquats = angle_axis_to_quaternion(domega)
    quats = qmul(rotation[b].float(), dquats).float()

    if vis_motion:
        # Motion mask: split static and dynamic part
        # Decoded zero-degree offset at current timestamp or the motion themselves
        motion_mask = ms3[b].norm(dim=-1, keepdim=True) > motion_mask_th

    # Compute scale
    scales = scale[b].float()

    # Compute updated opacity
    marginal_t = compute_marginal_t(ts[b, i], t[b], cov_t[b]).float()
    opacities = (opacity[b].float() * marginal_t)[..., 0]

    # Compute color
    if not retain_all_rgb:
        colors = (
            rgb[b, ..., :3].float() if "RGB" in render_mode else rgb[b, ..., 3:].float()
        )
    else:
        colors = rgb[b, ...].float()
    viewmats = w2cs[b, i : i + 1].float()  # 1, 3, 4
    ixts = Ks[b, i : i + 1].float()

    if vis_flow:
        # Compute mean in next camera frame
        curr_means = compute_means(xyz, t, ms3, w2cs, ts, b=b, i=i)  # N, 3
        if i < w2cs.shape[1] - 1:  # padding last manually
            next_means = compute_means(xyz, t, ms3, w2cs, ts, b=b, i=i + 1)  # N, 3
        else:
            next_means = curr_means

        # Render camera space mean
        colors = torch.cat([colors, curr_means, next_means], dim=-1)

    if vis_motion:
        colors = torch.cat([colors, motion_mask], dim=-1)  # N, C
        # colors = torch.cat([colors, velocity], dim=-1)  # N, C

    # Perform filtering before rasterization to same computational cost
    mask = (marginal_t[..., 0] > marginal_th) & (opacities > opacity_th)
    ind = mask.nonzero()[..., 0]  # S, # MARK: SYNC
    means, quats, scales, opacities, colors = (
        multi_gather(means, ind),
        multi_gather(quats, ind),
        multi_gather(scales, ind),
        multi_gather(opacities, ind, -1),
        multi_gather(colors, ind),
    )

    rgb, occ, dpt, meta = (threedgs_rasterizer if not use_2dgs else twodgs_rasterizer)(
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

    # For points with shorter life span
    # They will be filtered using the marginal and opacity threshold
    # We do not want extra regularization on those points
    # We only want the points that could be possibly seen on novel views to be regularized
    # Note that non-visible points are regularized
    for key in meta:
        canvas = torch.zeros_like(mask[None])  # N, # defaults to visible
        meta[key] = multi_scatter(canvas, ind[None], meta[key], dim=-1)

    return rgb, occ, dpt, meta


@RENDERER.register_module()
class TLODRenderer(GaussianRenderer):
    def __init__(
        self,
        height=512,
        width=512,
        marginal_th: float = 0.05,
        use_2dgs: bool = False,
        vis_flow: bool = False,
        vis_motion: bool = False,
        use_grad_checkpoint: bool = False,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.marginal_th = marginal_th
        self.use_2dgs = use_2dgs
        self.vis_flow = vis_flow
        self.vis_motion = vis_motion
        self.use_grad_checkpoint = use_grad_checkpoint

    def single_batch_rasterizer(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        ckpt_function = (
            dummy_function if not self.use_grad_checkpoint else auto_grad_checkpoint
        )

        B, T, _, _ = w2cs.shape

        # B, T, 3, 3 & B, T, 4, 4 for camera parameters
        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()
        for b in range(B):
            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()
            for i in range(T):
                out_img, out_alpha, out_dpt, meta = ckpt_function(
                    fourdgs_rasterizer,
                    xyz,
                    rgb,
                    scale,
                    rotation,
                    opacity,
                    t,
                    cov_t,
                    ms3,
                    omega,
                    w2cs,
                    Ks,
                    ts,
                    height or self.height,
                    width or self.width,
                    render_mode,
                    b,
                    i,
                    self.use_2dgs,
                    self.vis_motion or not xyz.requires_grad,  # directly computed
                    self.vis_flow or not xyz.requires_grad,  # directly computed
                )

                if xyz.requires_grad:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)
                else:
                    # To save memory during inference and rendering
                    out_imgs_batch.append(out_img.detach().cpu())
                    out_alphas_batch.append(out_alpha.detach().cpu())
                    out_dpts_batch.append(out_dpt.detach().cpu())
                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def __call__(
        self,
        gs_params: Mapping[str, torch.Tensor],  # B, T, H, W, C
        ts: torch.Tensor,  # B, T, 1
        Ks: torch.Tensor,  # B, T, 3, 3
        RTs: torch.Tensor,  # B, T, 3, 4
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        **kwargs,
    ):
        # B, T, H, W, C for params
        xyz, rgb, scale, rotation, opacity = (
            gs_params["xyz"],
            gs_params["feature"],  # assume last channel is for monochrome
            gs_params["scaling"],
            gs_params["rotation"],
            gs_params["opacity"],
        )

        t, cov_t, ms3, omega = (
            gs_params["t"],
            gs_params["cov_t"],
            gs_params["ms3"],
            gs_params["omega"],
        )

        xyz = xyz.reshape(xyz.shape[0], -1, xyz.shape[-1])
        rgb = rgb.reshape(rgb.shape[0], -1, rgb.shape[-1])
        scale = scale.reshape(scale.shape[0], -1, scale.shape[-1])
        rotation = rotation.reshape(rotation.shape[0], -1, rotation.shape[-1])
        opacity = opacity.reshape(opacity.shape[0], -1, opacity.shape[-1])
        t = t.reshape(t.shape[0], -1, t.shape[-1])
        cov_t = cov_t.reshape(cov_t.shape[0], -1, cov_t.shape[-1])
        ms3 = ms3.reshape(ms3.shape[0], -1, ms3.shape[-1])
        omega = omega.reshape(omega.shape[0], -1, omega.shape[-1])
        Ks = Ks.float()  # B, V, 3, 3
        w2cs = affine_inverse(RTs).float()  # B, V, 4, 4

        imgs = []
        masks = []
        depths = []
        metas = dotdict()
        B = ts.shape[0]
        for b in range(B):
            img, mask, depth, meta = self.single_batch_rasterizer(
                xyz[b : b + 1],
                rgb[b : b + 1],
                scale[b : b + 1],
                rotation[b : b + 1],
                opacity[b : b + 1],
                t[b : b + 1],
                cov_t[b : b + 1],
                ms3[b : b + 1],
                omega[b : b + 1],
                w2cs[b : b + 1],
                Ks[b : b + 1],
                ts[b : b + 1],
                height=height or self.height,
                width=width or self.width,
                render_mode=render_mode,
            )
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
            for key in meta:
                if key not in metas:
                    metas[key] = []
                metas[key].append(meta[key])
        img = torch.cat(imgs).float()  # B, V, C, H, W
        mask = torch.cat(masks).float()  # B, V, C, H, W
        depth = torch.cat(depths).float()  # B, V, C, H, W
        meta = dotdict()
        for key in metas:
            meta[key] = torch.cat(metas[key]).float()  # B, V, N

        if self.use_2dgs:
            C = img.shape[-3]
            img, normal, surf_normal, distort, median_depth = img.split(
                [C - 8, 3, 3, 1, 1], dim=-3
            )
            meta.normal = normal  # already in -1, 1 range
            meta.surf_normal = surf_normal  # already in -1, 1 range
            meta.distort = distort * 2 - 1
            meta.median_depth = median_depth

        # Universal
        if self.vis_motion or not xyz.requires_grad:
            C = img.shape[-3]
            img, motion_mask = img.split([C - 1, 1], dim=-3)
            meta.motion_mask = motion_mask * 2 - 1

        # Approximated 2D flow rendering
        if self.vis_flow or not xyz.requires_grad:
            Ks = Ks.to(img.device)  # MARK: SYNC
            C = img.shape[-3]
            img, mean0, mean1 = img.split([C - 3 - 3, 3, 3], dim=-3)
            # B, T, 3, H, W, camera space xyz map
            B, T, _, H, W = mean1.shape
            meta.mean0 = mean0
            meta.mean1 = mean1

            # B, T, N, 3 @ B, T, 3, 3 -> B, T, N, 3
            screen0 = mean0.reshape(B, T, 3, H * W).permute(0, 1, 3, 2) @ Ks.mT
            screen0 = screen0[..., :2] / (screen0[..., 2:].clip(1e-8))
            screen0 = screen0.permute(0, 1, 3, 2).reshape(B, T, 2, H, W)
            screen1 = mean1.reshape(B, T, 3, H * W).permute(0, 1, 3, 2) @ Ks.mT
            screen1 = screen1[..., :2] / (screen1[..., 2:].clip(1e-8))
            screen1 = screen1.permute(0, 1, 3, 2).reshape(B, T, 2, H, W)
            flow = screen1 - screen0
            del screen0, screen1
            # flow = torch.cat([flow, flow[:, -1:]], dim=1)
            meta.flow = flow

            flow = flow.reshape(-1, *flow.shape[-3:])
            flow_vis = flow_to_color(flow) / 255  # don't be too large
            flow_vis = flow_vis.reshape(img.shape[:-3] + flow_vis.shape[1:])
            meta.flow_vis = flow_vis * 2 - 1

        img = (img * 2 - 1).clamp(-1, 1)

        output = {
            "rgb": img,
            "mask": mask,
            "depth": depth,
            **meta,
        }
        if "MONO" in render_mode:
            output["mono"] = output.pop("rgb")
        return output
