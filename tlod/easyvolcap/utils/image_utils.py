# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import lru_cache
from typing import List

import numpy as np

import torch
from PIL import ImageFont
from torch import nn
from torch.nn import functional as F


def gkern2d(l=21, sig=3, device="cpu"):
    """Returns a 2D Gaussian kernel array."""
    ax = torch.arange(-l // 2 + 1.0, l // 2 + 1.0, device=device)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sig**2))
    return kernel


class Shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size**2)
        if self.kernel_size == 3:
            self.pad = 1
        elif self.kernel_size == 5:
            self.pad = 2
        elif self.kernel_size == 7:
            self.pad = 3

    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        # Alias for convenience
        cpg = self.channels_per_group
        cat_layers = []
        for i in range(self.in_planes):
            # Parse in row-major
            for y in range(0, self.kernel_size):
                y2 = y + h
                for x in range(0, self.kernel_size):
                    x2 = x + w
                    xx = x_pad[:, i : i + 1, y:y2, x:x2]
                    cat_layers += [xx]
        return torch.cat(cat_layers, 1)


class BilateralFilter(nn.Module):
    """BilateralFilter computes:
    If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(
        self,
        channels=3,
        k=7,
        height=480,
        width=640,
        sigma_space=5,
        sigma_color=0.1,
        device="cuda",
    ):
        super(BilateralFilter, self).__init__()

        # space gaussian kernel
        self.gw = gkern2d(k, sigma_space, device=device)

        self.g = torch.tile(
            self.gw.reshape(channels, k * k, 1, 1), (1, 1, height, width)
        )
        # shift
        self.shift = Shift(channels, k)
        self.sigma_color = 2 * sigma_color**2

        self.to(device=device)

    def forward(self, I):
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = (
            Is - Iex
        ) ** 2  # here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)
        Dd = De * self.g
        W_denom = torch.sum(Dd, dim=1)
        If = torch.sum(Dd * Is, dim=1) / W_denom
        return If


def get_xywh_from_mask(msk):
    import torchvision

    X, Y, H, W = 0, 0, *msk.shape[-3:-1]  # EMOTIONAL DAMAGE # noqa: F841
    bbox = torchvision.ops.boxes.masks_to_boxes(msk.view(-1, H, W))
    x = bbox[..., 0].min().round().int().item()  # round, smallest of all images
    y = bbox[..., 1].min().round().int().item()  # round, smallest of all images
    w = (
        (bbox[..., 2] - bbox[..., 0]).max().round().int().item()
    )  # round, biggest of all
    h = (
        (bbox[..., 3] - bbox[..., 1]).max().round().int().item()
    )  # round, biggest of all
    return x, y, w, h


def crop_using_mask(
    msk: torch.Tensor, K: torch.Tensor, *list_of_imgs: List[torch.Tensor]
):
    # Deal with empty batch dimension
    bs = msk.shape[:-3]
    msk = msk.view(-1, *msk.shape[-3:])
    K = K.view(-1, *K.shape[-2:])
    list_of_imgs = [im.view(-1, *im.shape[-3:]) for im in list_of_imgs]

    # Assumes channel last format
    # Assumes batch dimension for msk
    # Will crop all images using msk
    # !: EVIL LIST COMPREHENSION
    xs, ys, ws, hs = zip(*[get_xywh_from_mask(m) for m in msk])  # all sizes
    K, *list_of_imgs = zip(
        *[
            crop_using_xywh(x, y, w, h, k, *im)
            for x, y, w, h, k, *im in zip(xs, ys, ws, hs, K, *list_of_imgs)
        ]
    )  # HACK: This is doable... # outermost: source -> outermost: batch
    K = torch.stack(K)  # stack source dim
    # Resize instead of filling things up?
    # Filling things up might be easier for masked output (bkgd should be black)
    H_max = max(hs)
    W_max = max(ws)
    list_of_imgs = [
        torch.stack([fill_nhwc_image(im, size=(H_max, W_max)) for im in img])
        for img in list_of_imgs
    ]  # HACK: evil list comprehension

    # Restore original dimensionality
    msk = msk.view(*bs, *msk.shape[-3:])
    K = K.view(*bs, *K.shape[-2:])
    list_of_imgs = [im.view(*bs, *im.shape[-3:]) for im in list_of_imgs]
    return K, *list_of_imgs


def crop_using_xywh(x, y, w, h, K, *list_of_imgs):
    K = K.clone()
    K[..., :2, -1] -= torch.as_tensor([x, y], device=K.device)  # crop K
    list_of_imgs = [
        img[..., y : y + h, x : x + w, :]
        if isinstance(img, torch.Tensor)
        else [
            im[..., y : y + h, x : x + w, :] for im in img
        ]  # HACK: evil list comprehension
        for img in list_of_imgs
    ]
    return K, *list_of_imgs


def pad_image_to_divisor(img: torch.Tensor, div: List[int], mode="constant", value=0.0):
    H, W = img.shape[-2:]
    H = (H + div[0] - 1) // div[0] * div[0]
    W = (W + div[0] - 1) // div[0] * div[0]
    return pad_image(img, (H, W), mode, value)


def pad_image(img: torch.Tensor, size: List[int], mode="constant", value=0.0):
    bs = img.shape[:-3]  # batch size
    img = img.reshape(-1, *img.shape[-3:])
    H, W = img.shape[-2:]  # H, W
    Ht, Wt = size
    pad = (0, Wt - W, 0, Ht - H)

    if Wt >= W and Ht >= H:
        img = F.pad(img, pad, mode, value)
    else:
        if Wt < W and Ht >= H:
            img = F.pad(img, (0, 0, 0, Ht - H), mode, value)
        if Wt >= W and Ht < H:
            img = F.pad(img, (0, Wt - W, 0, 0), mode, value)
        img = img[..., :Ht, :Wt]

    img = img.reshape(*bs, *img.shape[-3:])
    return img


def fill_nchw_image(
    img: torch.Tensor, size: List[int], value: float = 0.0, center: bool = False
):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-3:-2]
    zeros = img.new_full((*bs, *cs, *size), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-2], img.shape[-1]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    start_h = (target_h - h) // 2 if center else 0
    start_w = (target_w - w) // 2 if center else 0
    zeros[..., start_h : start_h + h, start_w : start_w + w] = img[..., :h, :w]
    return zeros


def fill_nhwc_image(
    img: torch.Tensor, size: List[int], value: float = 0.0, center: bool = False
):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-1:]
    zeros = img.new_full((*bs, *size, *cs), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-3], img.shape[-2]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    start_h = (target_h - h) // 2 if center else 0
    start_w = (target_w - w) // 2 if center else 0
    zeros[..., start_h : start_h + h, start_w : start_w + w, :] = img[..., :h, :w, :]
    return zeros


def interpolate_image(
    img: torch.Tensor, mode="bilinear", align_corners=False, *args, **kwargs
):
    # Performs F.interpolate as images (always augment to B, C, H, W)
    sh = img.shape
    img = img.view(-1, *sh[-3:])
    img = F.interpolate(
        img,
        *args,
        mode=mode,
        align_corners=align_corners if mode != "nearest" else None,
        **kwargs,
    )
    img = img.view(sh[:-3] + img.shape[-3:])
    return img


def resize_image(
    img: torch.Tensor, mode="bilinear", align_corners=False, *args, **kwargs
):
    sh = img.shape
    if len(sh) == 4:  # assumption
        img = img.permute(0, 3, 1, 2)
    elif len(sh) == 3:  # assumption
        img = img.permute(2, 0, 1)[None]
    img = interpolate_image(
        img,
        mode=mode,
        align_corners=align_corners,
        *args,  # noqa: B026
        **kwargs,
    )  # uH, uW, 3
    if len(sh) == 4:
        img = img.permute(0, 2, 3, 1)
    elif len(sh) == 3:  # assumption
        img = img[0].permute(1, 2, 0)
    return img


@lru_cache
def get_pil_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)
