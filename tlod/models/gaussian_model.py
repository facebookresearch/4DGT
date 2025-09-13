# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from transformers import PretrainedConfig, PreTrainedModel

from ..easyvolcap.utils.data_utils import to_cpu

from ..registry import build_module, MODELS, RENDERER
from .encoders.image_encoder_TLoD import ImageEncoderTLoD


class GaussianModelConfig(PretrainedConfig):
    model_type = "GaussianModel"

    def __init__(self, image_encoder_config={}, renderer_config={}, **kwargs):  # noqa: B006
        self.image_encoder_config = image_encoder_config
        self.renderer_config = renderer_config
        super().__init__(**kwargs)


@MODELS.register_module()
class GaussianModel(PreTrainedModel):
    config_class = GaussianModelConfig

    def __init__(self, config):
        super(GaussianModel, self).__init__(config)
        self.config = config
        self.image_coder: ImageEncoderTLoD = build_module(
            self.config.image_encoder_config, MODELS
        )
        self.renderer = build_module(
            self.config.renderer_config, RENDERER
        )

    @torch.amp.autocast("cuda", enabled=False)
    def render(  # noqa: C901
        self,
        gaussian_parameters,
        supervising_timestamps,
        supervising_intrinsics,
        supervising_extrinsics,
        skip_render: bool = False,
        skip_saving_dense: bool = True,
        **renderer_kwargs,
    ):
        monochrome = renderer_kwargs.get("monochrome", None)

        if not skip_render:
            if monochrome is None or not monochrome.any():
                # Render things out as usual
                rendered_results = self.renderer(
                    gaussian_parameters,
                    supervising_timestamps,
                    supervising_intrinsics,
                    supervising_extrinsics,
                    **renderer_kwargs,
                )
            else:
                output = {}
                # NOTE: Assuming monochrome to be continuous and last of the cameras
                sep = (1 - monochrome).sum().int().item()  # MARK: SYNC
                n_cams = monochrome.shape[1] // sep
                for k in range(n_cams):
                    render_mode = renderer_kwargs.get("render_mode", "RGB+ED")
                    start = sep * k
                    end = sep * (k + 1)
                    if k > 0:
                        render_mode = render_mode.replace("RGB", "MONO")
                    renderer_kwargs["render_mode"] = render_mode
                    rendered_results = self.renderer(
                        gaussian_parameters,
                        supervising_timestamps[:, start:end],
                        supervising_intrinsics[:, start:end],
                        supervising_extrinsics[:, start:end],
                        **renderer_kwargs,
                    )
                    for k in rendered_results:
                        if k not in output:
                            output[k] = []
                        output[k].append(rendered_results[k])
                for k in output:
                    if len(output[k]):
                        # Do not cat empty list for mono
                        output[k] = torch.cat(output[k], dim=1)
                rendered_results = output
        else:
            rendered_results = {}

        rendered_results["gs"] = (
            gaussian_parameters if self.training else to_cpu(gaussian_parameters)
        )

        return rendered_results

    def forward(  # noqa: C901
        self,
        input_images,
        input_timestamps,
        input_intrinsics,
        input_extrinsics,
        supervising_timestamps,
        supervising_intrinsics,
        supervising_extrinsics,
        skip_render: bool = False,
        **renderer_kwargs,
    ):
        """
        input_images.shape = [b, num_input, 3, h, w]
        intrinsics.shape = [b, num_input, 3, 3]    camera2image
        extrinsics.shape = [b, num_input, 4, 4]    camera2world
        render_intrinsics = [b, num_render, 3, 3]  camera2image
        render_extrinsics = [b, num_render, 4, 4]  camera2world
        """

        gaussian_parameters = self.image_coder(
            input_images,
            input_timestamps,
            input_intrinsics,
            input_extrinsics,
            **renderer_kwargs,
        )
        
        rendered_results = self.render(
            gaussian_parameters,
            supervising_timestamps,
            supervising_intrinsics,
            supervising_extrinsics,
            skip_render,
            **renderer_kwargs,
        )

        return rendered_results


@MODELS.register_module("GaussianModelModule")
def GaussianModelModule(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        model = None
        pass
    else:
        config = GaussianModelConfig(**kwargs)
        model = GaussianModel(config)
    return model
