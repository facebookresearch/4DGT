# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PerceptualLoss(nn.Module):
    def __init__(self, loss_type):
        """
        loss_type:
            "zhang": Richard Zhang's networkd learned on BAPPS: https://arxiv.org/abs/1801.03924
            "chen": Qifeng Chen's loss with VGG19 pretrained on ImageNet: https://arxiv.org/abs/1707.09405
        """
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "alex":
            self.net = lpips.LPIPS(
                net="alex", verbose=False
            )  # faster forward & backward during training
        elif loss_type == "vgg" or loss_type == "zhang":
            self.net = lpips.LPIPS(net="vgg", verbose=False)
        elif loss_type == "chen":
            self.net = Vgg19FeatureExtractor(input_normalize_method="shift_scale")
        elif loss_type == "squeeze":
            self.net = lpips.LPIPS(
                net="squeeze", verbose=False
            )  # faster forward & backward during training
        else:
            raise ValueError("Unknown perc loss type {}".format(loss_type))

        for param in self.net.parameters():
            param.requires_grad = False

    def _resize_mask(self, mask, target):
        _, _, H, W = target.shape
        return F.interpolate(mask, [H, W], mode="area")

    def forward(self, image_1, image_2, mask=None):
        """
        image_1: Float[Tensor, "B C H W"], value range already normalized to [0, 1]
        image_2: Float[Tensor, "B C H W"], value range already normalized to [0, 1]
        mask: Float[Tensor, "B 1 H W"], specifies the regional loss weights.
        """

        if (
            self.loss_type == "alex"
            or self.loss_type == "vgg"
            or self.loss_type == "zhang"
            or self.loss_type == "squeeze"
        ):
            if mask is None:
                self.net.spatial = False
                loss = self.net(image_1, image_2).mean()
            else:
                self.net.spatial = True
                loss = self.net(image_1, image_2) * mask  # [B, C, H, W]
                loss = loss.mean()

        elif self.loss_type == "chen":
            f1 = self.net(image_1)
            f2 = self.net(image_2)
            loss = 0
            for k in f1.keys():
                if mask is None:
                    loss = loss + (f1[k] - f2[k]).abs().mean()
                else:
                    mask_k = self._resize_mask(mask, target=f1[k])
                    loss = loss + ((f1[k] - f2[k]).abs().mean(1) * mask_k[:, 0]).mean()

        else:
            raise NotImplementedError(self.loss_type)

        return loss


class Vgg19FeatureExtractor(nn.Module):
    def __init__(self, input_normalize_method="shift_scale"):
        """
        Implement the perc loss mentioned in https://arxiv.org/abs/1707.09405
        LRM authors mention this loss works better than LPIPS, and being used in the subsequent works of tLVSM.

        Note extracted layers follow the following definition:
        https://github.com/CQFIO/PhotographicImageSynthesis/blob/e89c281e996aaac8669df0df66fbefec704f0588/GTA_Diversity_256p.py#L85-L90
        """
        super().__init__()

        torchvision_version = torchvision.__version__
        if "+" in torchvision_version:
            torchvision_version = torchvision_version.split("+")[0]
        self.vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1")

        if input_normalize_method == "shift":
            # Somehow the authors use this shift
            # https://github.com/CQFIO/PhotographicImageSynthesis/blob/e89c281e996aaac8669df0df66fbefec704f0588/GTA_Diversity_256p.py#L31C34-L31C62
            self.register_buffer(
                "_vgg_mean",
                torch.FloatTensor(
                    [123.6800 / 255, 116.7790 / 255, 103.9390 / 255]
                ).reshape(1, 3, 1, 1),
            )
            self.input_normalize_fn = lambda x: (0.5 * (x + 1) - self._vgg_mean)
        elif input_normalize_method == "shift_scale":
            self.register_buffer(
                "_vgg_mean",
                torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
            )
            self.register_buffer(
                "_vgg_std", torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            )
            self.input_normalize_fn = (
                lambda x: (0.5 * (x + 1) - self._vgg_mean) / self._vgg_std
            )
        else:
            raise ValueError(input_normalize_method)

        self.layer_idx_to_name = {
            2 + 1: "conv1_2",
            7 + 1: "conv2_2",
            12 + 1: "conv3_2",  # 16 for `conv3_4`
            21 + 1: "conv4_2",  # 25 for `conv4_4`
            30 + 1: "conv5_2",  # 34 for `conv5_4`
        }
        self.feature_extract_inds = self.layer_idx_to_name.keys()

        # Paper mentioned there per-layer weighting is learnable, but they were written as constants in the released implementaiton.
        self.feature_weights = {
            "conv1_2": 1 / 2.6,
            "conv2_2": 1 / 4.8,
            "conv3_2": 1 / 3.7,
            "conv4_2": 1 / 5.6,
            "conv5_2": 10 / 1.5,
        }

        # Remove unused layers
        max_layer_ind = max(self.feature_extract_inds)
        if max_layer_ind < len(self.vgg.features) - 1:
            for layer in self.vgg.features[max_layer_ind + 1 :]:
                del layer

    def forward(self, image):
        """
        image: Float[Tensor, "B C H W"], value range already normalized to [0, 1]
        """
        f = {}
        h = self.input_normalize_fn(image)
        for i, layer in enumerate(self.vgg.features):
            h = layer(h)
            if i in self.feature_extract_inds:
                layer_name = self.layer_idx_to_name[i]
                f[layer_name] = h * self.feature_weights[layer_name]
        return f
