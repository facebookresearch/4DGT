# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Fully sharded multi-view Aria dataset loader
Preload things we need into memory, including camera parameters and videos
"""

import math
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..easyvolcap.utils.cam_utils import (
    align_c2ws,
    average_c2ws,
    generate_spiral_path,
    interpolate_camera_lins,
    interpolate_camera_path,
)
from ..easyvolcap.utils.console_utils import (
    dotdict,
    join,
    logger,
    magenta,
)
from ..easyvolcap.utils.data_utils import as_numpy_func
from ..easyvolcap.utils.math_utils import affine_inverse, affine_padding
from ..easyvolcap.utils.parallel_utils import parallel_execution

from ..misc.io_helper import pathmgr
from .utils import (  # noqa: F401
    load_aria_images,
    load_camera_poses,
    load_image,
    pack_c2ws_to_cameras,
    prepare_images,
    rotate_cameras,
    rotate_images,
)


class AriaDataset(Dataset):
    def __init__(  # noqa: C901
        self,
        mode: str = "TEST",
        data_root: str = "data/adt",
        input_image_res: Tuple[int] = (256, 256),
        input_image_num: int = 8,
        output_image_res: Tuple[int] = (256, 256),
        output_image_num: int = 8,
        seq_sample: Tuple[Optional[int]] = (0, None, 1),
        frame_sample: Tuple[Optional[int]] = (0, None, 1),
        view_sample: Tuple[Optional[int]] = (0, 1, 1),
        sample_interval: int = 128,
        seq_data_roots: Tuple[str] = ("camera-rgb-rectified-600-h1000",),
        video_file: str = "videos.mp4",
        vignette_file: str = "no_vignette.png",
        transforms_file: str = "transforms.json",
        seq_list: str = "",
        align_cameras: bool = False,
        novel_time_sampling: bool = False,
        novel_time_frame_sample: Tuple[Optional[int]] = (0, None, 2),
        rotate_rgb: bool = True,
        novel_view_interp_input: bool = False,
        novel_view_timestamps: Tuple[float] = (),
        novel_view_spiral_window: int = 32,
        loaded_to_seconds: float = 1e9,
        loaded_to_meters: float = 1.0,
        force_reload: bool = False,
        **kwargs,
    ):
        """
        AriaDataset for inference/testing only.
        
        Args:
            mode: Must be "TEST" - this dataset only supports inference mode
            data_root: Root directory containing the data
            
            # Image definition
            input_image_res: Resolution for input images
            input_image_num: Number of input images (rgb only)
            output_image_res: Resolution for output images
            output_image_num: Number of output images (rgb and slam)
            
            # Frame definition
            seq_sample: Sequence sampling [start, end, step]
            frame_sample: Frame sampling [start, end, step] - use all possible frames
            view_sample: View sampling [start, end, step] - only input the first view and supervise on all
            sample_interval: Distance between images to sample as starting index
            
            # Data paths
            seq_data_roots: Root directories for sequence data
            video_file: Video file name
            vignette_file: Vignette file name
            transforms_file: Transforms file name
            seq_list: Name of the sequences
            
            # Camera alignment
            align_cameras: Use average camera pose, might lead to jumpy results
            
            # Novel view/time settings
            novel_time_sampling: Whether to do novel time sampling
            novel_time_frame_sample: Novel time frame sampling parameters. 
                Default is (0, None, 2), which means starting from the first frame to the last with stride 2. 
            rotate_rgb: Whether to rotate RGB data
            novel_view_interp_input: Whether to interpolate novel views for input
            novel_view_timestamps: Specific timestamps for novel views
            novel_view_spiral_window: Number of frames centering target for spiral lookat
            
            # Data loading settings
            loaded_to_seconds: Conversion factor for timestamps to seconds
            loaded_to_meters: Conversion factor for distances to meters
            force_reload: Force reload data even if cached
        """
        super().__init__()
        
        # This dataset only supports TEST mode for inference
        assert mode == "TEST", f"AriaDataset only supports TEST mode, got {mode}. Use this dataset for inference only."

        if pathmgr.isdir(join(data_root, seq_list)):
            seqs = [seq_list]
        else:
            seqs = pathmgr.ls(data_root)
            seqs = [s for s in seqs if pathmgr.isdir(join(data_root, s))]
            seqs = sorted(seqs)

            b, e, s = seq_sample
            seqs = seqs[b:e:s]

        self.data_root = data_root
        self.input_image_res = input_image_res
        self.input_segment_length = input_image_num
        self.output_image_res = output_image_res
        self.output_segment_length = output_image_num

        self.non_aria = "camera-rgb" not in seq_data_roots[0]
        if self.non_aria:
            loaded_to_seconds = 1.0
            # loaded_to_meters = 1.0
            if "cop3d" in data_root:
                loaded_to_meters = 5.0
            if "epic-fields" in data_root:
                loaded_to_meters = 5.0
            rotate_rgb = False

        self.seq_data_roots = deepcopy(seq_data_roots)
        self.view_sample = view_sample
        self.video_file = video_file
        self.vignette_file = vignette_file
        self.transforms_file = transforms_file
        # Depth and normal flags removed
        self.batch_image_num = max(input_image_num, output_image_num)

        self.novel_time_sampling = novel_time_sampling
        self.novel_time_frame_sample = novel_time_frame_sample

        self.align_cameras = align_cameras
        self.sample_interval = sample_interval

        self.rotate_rgb = rotate_rgb

        self.novel_view_interp_input = novel_view_interp_input
        self.novel_view_timestamps = novel_view_timestamps
        self.novel_view_spiral_window = novel_view_spiral_window

        max_image_size = max(max(input_image_res), max(output_image_res))

        b, e, s = view_sample

        seqs = np.asarray(seqs)
        logger.info(
            f"Number of sequences used for {magenta(mode)}: {len(seqs)}, seqs: {seqs}"
        )

        # Preload camera parameters from nerf-format (aria modified) transforms.json file
        def print_progress(i, total, result):
            if i % math.ceil(total / 20) == 0 or i == total - 1:
                logger.info(f"Loaded {i+1}/{total} ({(i+1)/total*100:.2f}%) sequences")
            return result

        output_list = []
        for i, seq_data_root in enumerate(self.seq_data_roots):
            # In TEST mode, only subsample the first camera
            should_subsample = (i == 0)
            mock_frame_sample = (0, None, 1)
            b, e, s = (
                frame_sample if should_subsample else mock_frame_sample
            )  # subsampling
            logger.info(f'Loading "{seq_data_root}" camera poses')
            inputs = [
                join(data_root, key, seq_data_root, transforms_file) for key in seqs
            ]
            outputs = parallel_execution(
                inputs,
                action=partial(
                    load_camera_poses,
                    frame_sample=(b, e, s),
                    loaded_to_seconds=loaded_to_seconds,
                    loaded_to_meters=loaded_to_meters,
                ),
                print_progress=False,
                callback=print_progress,
            )
            output_list.append(outputs)

        total = len(seqs)
        self.lengths = []
        self.seqs = dotdict()
        # b, e, s = frame_sample
        for i, (seq_data_root, outputs) in enumerate(
            zip(self.seq_data_roots, output_list)
        ):
            # In TEST mode, only subsample the first camera
            should_subsample = (i == 0)
            mock_frame_sample = (0, None, 1)
            b, e, s = (
                frame_sample if should_subsample else mock_frame_sample
            )  # subsampling

            logger.info(f'Parsing "{seq_data_root}" videos')
            for j, (key, output) in enumerate(
                zip(seqs, outputs),
            ):
                ims, Hs, Ws, Ks, RTs, ts = output
                if key not in self.seqs:
                    self.seqs[key] = dotdict()
                self.seqs[key][seq_data_root] = dotdict()
                self.seqs[key][seq_data_root].ims = ims
                self.seqs[key][seq_data_root].Hs = Hs
                self.seqs[key][seq_data_root].Ws = Ws
                self.seqs[key][seq_data_root].Ks = Ks
                self.seqs[key][seq_data_root].RTs = RTs  # w2c
                self.seqs[key][seq_data_root].ts = ts  # nanoseconds to seconds
                # self.seqs[key][seq_data_root].fps = 1 / np.mean(np.diff(ts))

                # Only load vignette if it exists
                if pathmgr.exists(join(data_root, key, seq_data_root, vignette_file)):
                    self.seqs[key][seq_data_root].vignette = load_image(
                        join(data_root, key, seq_data_root, vignette_file)
                    )

                self.seqs[key][seq_data_root].ims = prepare_images(
                    data_root, 
                    key,  # sequence name
                    seq_data_root,  # camera name
                    video_file,  # actual video file name
                    (b, e, s),  # subsampling
                    ims,  # image names
                    resize=max_image_size,
                    force_reload=force_reload,  # In TEST mode, use force_reload parameter
                )

                if not len(self.seqs[key][seq_data_root].ims):
                    logger.warn(f"Empty sequence {key} {seq_data_root}")
                    self.seqs.pop(key)
                    seqs = np.concatenate([seqs[:i], seqs[i + 1:]])
                    continue

                # Print progress
                if j % math.ceil(total / 20) == 0 or j == total - 1:
                    logger.info(
                        f"Parsed {j+1}/{total} ({(j+1)/total*100:.2f}%) sequences"
                    )

                # Only need to calculate length of videos once
                if i == 0:
                    self.lengths.append(
                        int(
                            len(self.seqs[key][seq_data_root].ims)
                            // self.sample_interval
                        )
                    )

        self.seq_keys = list(self.seqs.keys())
        self.lengths = np.asarray(self.lengths)
        self.cumsum_lengths = [0] + self.lengths.cumsum(-1).tolist()

        assert (
            len(self.seq_keys) == len(self.cumsum_lengths) - 1
            and len(self.seq_keys) == len(self.lengths)
            and len(self.seq_keys) == len(self.seqs)
        ), f"Lengths of seq_keys, cumsum_lengths, lengths, and seqs should be the same, but got {len(self.seq_keys)}, {len(self.cumsum_lengths) - 1}, {len(self.lengths)}, and {len(self.seqs)}"


    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx: int):  # noqa: C901
        idx = idx % self.cumsum_lengths[-1]

        # Searchsorted from the right would ignore lengths of zeros
        seq_idx = np.searchsorted(self.cumsum_lengths, idx, side="right").item() - 1
        key = self.seq_keys[seq_idx]
        sub_idx = idx - self.cumsum_lengths[seq_idx]
        global_abs_start_ind = sub_idx * self.sample_interval

        rgb_input = []
        rgb_output = []
        rays_t_un_input = []
        rays_t_un_output = []
        cameras_input = []
        cameras_output = []
        img_name_output = []
        ratios_output = []
        c2w_avg = None
        vb, ve, vs = self.view_sample
        fb, _, fs = self.novel_time_frame_sample
        inputs = np.arange(len(self.seq_data_roots))[vb:ve:vs]
        n_inputs = len(inputs)

        for i, seq_data_root in enumerate(self.seq_data_roots):
            curr_len = len(self.seqs[key][seq_data_root].ts)
            pre = join(self.data_root, key, seq_data_root)
            is_input = i in inputs
            input_sampling_starting_offset = max(
                (fs // n_inputs) * (i - n_inputs // 2), -fb
            )
            dist = 1
            abs_start_ind = global_abs_start_ind
            abs_end_ind = (
                global_abs_start_ind + self.batch_image_num * dist
            )
            target_num_images = self.batch_image_num
            if i != 0:
                abs_end_ind = min(curr_len, abs_end_ind)
                start = self.seqs[key][self.seq_data_roots[0]].ts[abs_start_ind]
                end = self.seqs[key][self.seq_data_roots[0]].ts[abs_end_ind - 1]
                ts = self.seqs[key][seq_data_root].ts
                abs_start_ind = np.searchsorted(ts, start, side="left")
                abs_end_ind = np.searchsorted(ts, end, side="left")
                target_num_images = abs_end_ind - abs_start_ind
            abs_end_ind = min(
                curr_len, abs_end_ind
            )
            abs_start_ind = min(abs_start_ind, abs_end_ind - target_num_images)
            abs_start_ind = max(0, abs_start_ind)
            sample_inds = np.arange(abs_start_ind, abs_end_ind, dist)
            if len(sample_inds) != target_num_images:
                # In TEST mode, truncate to target number of images
                sample_inds = sample_inds[:target_num_images]

            ims = self.seqs[key][seq_data_root].ims[sample_inds].copy()
            Hs = self.seqs[key][seq_data_root].Hs[sample_inds].copy()
            Ws = self.seqs[key][seq_data_root].Ws[sample_inds].copy()
            Ks = self.seqs[key][seq_data_root].Ks[sample_inds].copy()
            RTs = self.seqs[key][seq_data_root].RTs[sample_inds].copy()
            ts = self.seqs[key][seq_data_root].ts[sample_inds].copy()
            if "vignette" not in self.seqs[key][seq_data_root]:
                vignette = np.ones((Hs[0], Ws[0], 3), dtype=np.float32)
            else:
                vignette = self.seqs[key][seq_data_root].vignette

            ts -= ts.min()
            ts = ts.astype(np.float32)

            h, w = self.input_image_res if is_input else self.output_image_res

            # Handle center crop and resizing
            xs, ys = [], []
            ws, hs = [], []
            ratios = []
            for i in range(len(ims)):
                if Hs[i] > Ws[i]:
                    ratio_x = w / Ws[i]
                    ratio_y = int(ratio_x * Hs[i] + 0.5) / Hs[i]
                    ratio = ratio_x
                else:
                    ratio_y = h / Hs[i]
                    ratio_x = int(ratio_y * Ws[i] + 0.5) / Ws[i]
                    ratio = ratio_y
                if h / Hs[i] > w / Ws[i]:
                    x, y = int((Ws[i] * ratio_x - w + 0.5) // 2), 0
                else:
                    x, y = 0, int((Hs[i] * ratio_y - h + 0.5) // 2)
                Ks[i, 0:1] *= ratio_x
                Ks[i, 1:2] *= ratio_y
                Ks[i, 0, 2] -= x
                Ks[i, 1, 2] -= y
                xs.append(x)
                ys.append(y)
                ws.append(int(ratio_x * Ws[i] + 0.5))
                hs.append(int(ratio_y * Hs[i] + 0.5))
                Hs[i] = h
                Ws[i] = w
                ratios.append(ratio)
            ratios = np.asarray(ratios)

            # Vignette
            i = 0
            vignette = cv2.resize(
                (vignette * 255).astype(np.uint8), dsize=(ws[i], hs[i])
            )
            vignette = vignette[
                ys[i]: ys[i] + Hs[i], xs[i]: xs[i] + Ws[i]
            ]
            vignette = (vignette / 255).astype(np.float32)
            if self.rotate_rgb:
                vignette = rotate_images(vignette[None])[0]
            vignette = np.transpose(vignette[None], (0, 3, 1, 2))[0]

            imgs = load_aria_images(ims, pre, xs, ys, Ws, Hs, hs, ws, self.rotate_rgb)
            imgs = imgs[:, :3]

            imgs = imgs / vignette
            imgs = np.clip(imgs, 0, 1)
            imgs = imgs * 2 - 1

            if self.rotate_rgb:
                rotate_cameras(RTs, Ks, Ws, Hs)
            c2ws = as_numpy_func(affine_inverse)(RTs)

            if self.align_cameras and c2w_avg is None:
                c2w_avg = average_c2ws(
                    as_numpy_func(affine_inverse)(RTs),
                    align_cameras=False,
                    look_at_center=True,
                )
            if c2w_avg is not None:
                c2ws = align_c2ws(c2ws, c2w_avg)
                c2ws = as_numpy_func(affine_padding)(c2ws)
            cameras = pack_c2ws_to_cameras(c2ws, Ks, Hs, Ws)

            if is_input:
                if self.novel_time_sampling:
                    b, e, s = self.novel_time_frame_sample
                    b = b + input_sampling_starting_offset
                    e = len(imgs)
                    inds = np.arange(b, e, s)
                else:
                    inds = np.arange(len(imgs))

                imgs_input = imgs[inds]
                cams_input = cameras[inds]

                rgb_input.append(imgs_input)
                cameras_input.append(cams_input)

                rays_t_un_input.append(ts[inds])

            if self.novel_view_interp_input:
                interp_c2ws = c2ws[inds]
                interp_Ks = Ks[inds]
                interp_ts = ts[inds]
                
                c2ws = interpolate_camera_path(
                    interp_c2ws, n_render_views=len(c2ws)
                )
                fx = interp_Ks[..., 0, 0]
                fy = interp_Ks[..., 1, 1]
                cx = interp_Ks[..., 0, 2]
                cy = interp_Ks[..., 1, 2]
                ts = interp_ts
                lins = np.stack(
                    [fx, fy, cx, cy, ts],
                    axis=-1,
                )
                lins = interpolate_camera_lins(lins, n_render_views=len(c2ws))
                fx, fy, cx, cy, ts = (
                    lins[..., 0],
                    lins[..., 1],
                    lins[..., 2],
                    lins[..., 3],
                    lins[..., 4],
                )
                Ks[..., 0, 0] = fx
                Ks[..., 1, 1] = fy
                Ks[..., 0, 2] = cx
                Ks[..., 1, 2] = cy
                ts = ts

                cameras = pack_c2ws_to_cameras(c2ws, Ks, Hs, Ws)
            elif len(self.novel_view_timestamps):
                n_render_views = len(c2ws)
                interps = dotdict()
                half_window = self.novel_view_spiral_window // 2
                for t in self.novel_view_timestamps:
                    idx = np.searchsorted(ts, t)
                    start = max(0, idx - half_window)
                    end = min(len(ts), idx + half_window)
                    interp_c2ws = c2ws[start:end]
                    interp_Ks = Ks[start:end]
                    interp_c2ws = generate_spiral_path(
                        interp_c2ws,
                        n_render_views=n_render_views,
                        radii_overwrite=(0.05, 0.05, 0.005),
                        c2w_avg_overwrite=c2ws[idx],
                    )
                    fx = interp_Ks[..., 0, 0]
                    fy = interp_Ks[..., 1, 1]
                    cx = interp_Ks[..., 0, 2]
                    cy = interp_Ks[..., 1, 2]
                    lins = np.stack(
                        [fx, fy, cx, cy],
                        axis=-1,
                    )
                    lins = interpolate_camera_lins(lins, n_render_views=n_render_views)
                    fx, fy, cx, cy = (
                        lins[..., 0],
                        lins[..., 1],
                        lins[..., 2],
                        lins[..., 3],
                    )
                    interp_Ks = np.ascontiguousarray(
                        np.broadcast_to(
                            np.eye(3, dtype=np.float32)[None], (n_render_views, 3, 3)
                        )
                    )
                    interp_Ks[..., 0, 0] = fx
                    interp_Ks[..., 1, 1] = fy
                    interp_Ks[..., 0, 2] = cx
                    interp_Ks[..., 1, 2] = cy

                    interp_ts = np.full_like(ts, t)

                    interps[idx] = dotdict()
                    interps[idx].cameras = pack_c2ws_to_cameras(
                        interp_c2ws,
                        interp_Ks,
                        Hs,
                        Ws,
                    )
                    interps[idx].ts = interp_ts

                    interps[idx].imgs = np.broadcast_to(
                        imgs[idx: idx + 1],
                        (n_render_views,) + imgs.shape[1:],
                    )

                indices = list(interps.keys())[::-1]

                for idx in indices:
                    cameras = np.concatenate(
                        [cameras[:idx], interps[idx].cameras, cameras[idx:]]
                    )
                    ts = np.concatenate([ts[:idx], interps[idx].ts, ts[idx:]])
                    imgs = np.concatenate(
                        [
                            imgs[:idx],
                            interps[idx].imgs,
                            imgs[idx:],
                        ]
                    )
                    ims = np.concatenate(
                        [
                            ims[:idx],
                            np.broadcast_to(
                                ims[idx: idx + 1],
                                (n_render_views,) + ims.shape[1:],
                            ),
                            ims[idx:],
                        ]
                    )
                    ratios = np.concatenate(
                        [
                            ratios[:idx],
                            np.broadcast_to(
                                ratios[idx: idx + 1],
                                (n_render_views,) + ratios.shape[1:],
                            ),
                            ratios[idx:],
                        ]
                    )

            inds = np.arange(len(imgs))

            rgb_output.append(imgs[inds])
            rays_t_un_output.append(ts[inds])
            cameras_output.append(cameras[inds])
            img_name_output.append(
                np.asarray([join(key, seq_data_root, im) for im in ims[inds]])
            )
            ratios_output.append(np.asarray(ratios))

        batch = dotdict()
        batch.rgb_input = np.concatenate(rgb_input).astype(np.float32)
        batch.rays_t_un_input = np.concatenate(rays_t_un_input).astype(np.float32)
        batch.cameras_input = np.concatenate(cameras_input).astype(np.float32)
        batch.rgb_output = np.concatenate(rgb_output).astype(np.float32)
        batch.rays_t_un_output = np.concatenate(rays_t_un_output).astype(np.float32)
        batch.cameras_output = np.concatenate(cameras_output).astype(np.float32)
        batch.img_name_output = np.concatenate(img_name_output)
        batch.ratios_output = np.concatenate(ratios_output).astype(np.float32)

        batch.img_name_output = batch.img_name_output.tolist()
        batch.c2w_avg = c2w_avg if c2w_avg is not None else np.eye(4)
        return batch
