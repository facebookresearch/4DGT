# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Video and Image Saving System for 4DGT.

This module handles all file I/O operations for saving rendered outputs,
including images, videos, and various visualization formats.
"""

# fmt: off
import os
import time
import torch
import logging
import imageio
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from os.path import join, dirname, exists
from tqdm import tqdm
from queue import Queue
import threading

from tlod.misc import utils
from tlod.misc.io_helper import mkdirs
from tlod.easyvolcap.utils.console_utils import logger, blue, green
from tlod.easyvolcap.utils.data_utils import dotdict
# fmt: on


def norm_to_uint8(array):
    """Convert normalized array to uint8."""
    import numpy as np
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)



@dataclass
class SaveConfig:
    """Configuration for saving outputs."""
    # What to save
    rgb: bool = True
    depth: bool = False
    normal: bool = False
    motion_mask: bool = False
    flow: bool = False
    gaussians: bool = False

    # Gaussian saving options
    gaussian_format: str = 'npz'  # Format for saving gaussians
    save_gaussians_async: bool = True  # Save gaussians asynchronously

    # How to save
    save_raw: bool = True  # Save raw tensor outputs
    save_visualization: bool = True  # Save visualizations (e.g., depth colormap)
    save_as_video: bool = False  # Save as video instead of individual images

    # Video settings
    video_fps: int = 30
    video_codec: str = 'h264'
    video_quality: int = 8  # 0-10, higher is better

    # Performance settings
    process_in_chunks: bool = True
    chunk_size: int = 32  # Process this many frames at a time for video
    use_async_save: bool = True  # Use async I/O for saving
    max_workers: int = 4  # Max concurrent save operations


class VideoSystem:
    """
    Handles all video and image saving operations for 4DGT outputs.

    This class manages:
    - Individual frame saving (images)
    - Video creation and encoding
    - Asynchronous I/O operations
    - Memory-efficient chunked processing
    - Various output formats (RGB, depth, normal, flow, etc.)
    """

    def __init__(self,
                 output_dir: str = "outputs",
                 verbose: bool = False):
        """
        Initialize the VideoSystem.

        Args:
            output_dir: Base directory for saving outputs
            verbose: Whether to print detailed logs
        """
        self.output_dir = output_dir
        self.verbose = verbose

        # Setup directories
        self.images_dir = join(output_dir, "images")
        self.videos_dir = join(output_dir, "videos")
        mkdirs(self.images_dir)
        mkdirs(self.videos_dir)

        # Async saving
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.save_futures = []

        # Gaussian queue for async rendering
        self.gaussian_queue = Queue(maxsize=10)  # Limited size to control memory
        self.gaussian_save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.gaussian_futures = []

        # Paths for gaussians
        self.gaussians_dir = join(output_dir, "gaussians")
        mkdirs(self.gaussians_dir)

        # Video writers (for frame-by-frame video saving)
        self._frame_writers = {}
        self._frame_count = 0

    def save_outputs(self,
                     outputs: Dict[str, torch.Tensor],
                     batch_idx: int,
                     save_config: SaveConfig) -> None:
        """
        Save model outputs to disk.

        Args:
            outputs: Dictionary of model outputs
            batch_idx: Batch index for naming
            save_config: Configuration for what/how to save
        """
        if save_config.use_async_save:
            self._save_outputs_async(outputs, batch_idx, save_config)
        else:
            self._save_outputs_sync(outputs, batch_idx, save_config)

    def _save_outputs_async(self,
                            outputs: Dict[str, torch.Tensor],
                            batch_idx: int,
                            save_config: SaveConfig) -> None:
        """Save outputs asynchronously using thread pool."""
        # Transfer to CPU immediately to free GPU
        cpu_outputs = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                cpu_outputs[key] = value.cpu()
            else:
                cpu_outputs[key] = value

        # Submit save task
        future = self.save_executor.submit(
            self._save_outputs_sync,
            cpu_outputs,
            batch_idx,
            save_config
        )
        self.save_futures.append(future)

        if self.verbose:
            logger.info(f"Queued save for batch {batch_idx} (async)")

    def _save_outputs_sync(self,
                           outputs: Dict[str, torch.Tensor],
                           batch_idx: int,
                           save_config: SaveConfig) -> None:
        """Synchronously save outputs to disk."""
        start_time = time.time()

        if save_config.save_as_video:
            self._save_as_video(outputs, batch_idx, save_config)
        else:
            self._save_as_images(outputs, batch_idx, save_config)

        save_time = time.time() - start_time
        if self.verbose:
            logger.info(f"Saved batch {batch_idx} in {save_time:.2f}s")

    def _save_as_images(self,
                        outputs: Dict[str, torch.Tensor],
                        batch_idx: int,
                        save_config: SaveConfig) -> None:
        """Save outputs as individual image files."""
        # Save RGB
        if save_config.rgb and 'rgb' in outputs:
            rgb_path = join(self.images_dir, f"batch_{batch_idx:06d}_rgb.png")
            self._save_tensor_as_image(outputs['rgb'], rgb_path, normalize=True)  # RGB needs normalization
            logger.info(f"Saved RGB to {blue(rgb_path)}")

        # Save depth
        if save_config.depth and 'depth' in outputs:
            if save_config.save_raw:
                depth_path = join(self.images_dir, f"batch_{batch_idx:06d}_depth_raw.npy")
                np.save(depth_path, outputs['depth'].cpu().numpy())
                logger.info(f"Saved raw depth to {blue(depth_path)}")

            if save_config.save_visualization:
                depth_vis_path = join(self.images_dir, f"batch_{batch_idx:06d}_depth.png")
                depth_vis = self._visualize_depth(outputs['depth'])
                self._save_tensor_as_image(depth_vis, depth_vis_path, normalize=False)  # Already in [0, 1]
                logger.info(f"Saved depth visualization to {blue(depth_vis_path)}")

        # Save normal
        if save_config.normal and 'normal' in outputs:
            if save_config.save_raw:
                normal_raw_path = join(self.images_dir, f"batch_{batch_idx:06d}_normal_raw.npy")
                np.save(normal_raw_path, outputs['normal'].cpu().numpy())
                logger.info(f"Saved raw normal to {blue(normal_raw_path)}")

            if save_config.save_visualization:
                normal_path = join(self.images_dir, f"batch_{batch_idx:06d}_normal.png")
                normal_vis = self._visualize_normal(outputs['normal'])
                self._save_tensor_as_image(normal_vis, normal_path, normalize=False)  # Already normalized
                logger.info(f"Saved normal to {blue(normal_path)}")

        # Save motion mask
        if save_config.motion_mask and 'motion_mask' in outputs:
            mask_path = join(self.images_dir, f"batch_{batch_idx:06d}_motion_mask.png")
            self._save_tensor_as_image(outputs['motion_mask'], mask_path, normalize=False)  # Mask is already [0, 1]
            logger.info(f"Saved motion mask to {blue(mask_path)}")

        # Save flow
        if save_config.flow and 'flow' in outputs:
            if save_config.save_raw:
                flow_raw_path = join(self.images_dir, f"batch_{batch_idx:06d}_flow_raw.npy")
                np.save(flow_raw_path, outputs['flow'].cpu().numpy())
                logger.info(f"Saved raw flow to {blue(flow_raw_path)}")

            if save_config.save_visualization:
                flow_path = join(self.images_dir, f"batch_{batch_idx:06d}_flow.png")
                flow_vis = self._visualize_flow(outputs['flow'])
                self._save_tensor_as_image(flow_vis, flow_path, normalize=False)  # Already normalized
                logger.info(f"Saved flow to {blue(flow_path)}")

    def _save_as_video(self,
                       outputs: Dict[str, torch.Tensor],
                       batch_idx: int,
                       save_config: SaveConfig) -> None:
        """Save outputs as video files."""
        B = outputs['rgb'].shape[0] if 'rgb' in outputs else 1

        for b in range(B):
            # Extract single batch
            batch_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] > b:
                    batch_outputs[key] = value[b:b + 1]

            # Save each output type as video
            if save_config.rgb and 'rgb' in batch_outputs:
                video_path = join(self.videos_dir, f"batch_{batch_idx:06d}_b{b}_rgb.mp4")
                self._encode_video(batch_outputs['rgb'], video_path, save_config, normalize=True)
                logger.info(f"Saved RGB video to {blue(video_path)}")

            if save_config.depth and 'depth' in batch_outputs:
                depth_vis = self._visualize_depth(batch_outputs['depth'])
                video_path = join(self.videos_dir, f"batch_{batch_idx:06d}_b{b}_depth.mp4")
                self._encode_video(depth_vis, video_path, save_config, normalize=False)  # Already in [0, 1]
                logger.info(f"Saved depth video to {blue(video_path)}")

    def _encode_video(self,
                      frames: torch.Tensor,
                      output_path: str,
                      save_config: SaveConfig,
                      normalize: bool = True) -> None:
        """
        Encode tensor frames as video.

        Args:
            frames: Tensor of shape [B, N, C, H, W] or [N, C, H, W]
            output_path: Path to save video
            save_config: Video encoding settings
            normalize: If True, normalize from [-1, 1] to [0, 1] (for RGB)
        """
        mkdirs(dirname(output_path))

        # Ensure we have [N, C, H, W]
        if frames.dim() == 5:
            frames = frames.squeeze(0)  # [1, N, C, H, W] -> [N, C, H, W]

        # Normalize if needed (RGB images from model are typically in [-1, 1])
        if normalize:
            frames = (frames + 1) / 2  # [-1, 1] -> [0, 1]

        # Convert to numpy and prepare for video
        frames_np = frames.cpu().numpy()
        N, C, H, W = frames_np.shape

        # Convert to uint8
        frames_uint8 = []
        for i in range(N):
            frame = frames_np[i]  # [C, H, W]
            if C == 1:
                frame = frame.squeeze(0)  # [H, W]
            else:
                frame = frame.transpose(1, 2, 0)  # [H, W, C]

            frame_uint8 = norm_to_uint8(frame)
            frames_uint8.append(frame_uint8)

        # Write video with proper ffmpeg parameters
        # Use crf instead of quality to avoid warnings
        # CRF scale: 0-51, lower is better quality (18-23 is good range)
        crf_value = 51 - (save_config.video_quality * 5)  # Map 0-10 to 51-1
        crf_value = max(1, min(51, int(crf_value)))  # Clamp to valid range

        writer = imageio.get_writer(
            output_path,
            fps=save_config.video_fps,
            codec=save_config.video_codec,
            ffmpeg_params=['-crf', str(crf_value), '-pix_fmt', 'yuv420p']
        )

        for frame in frames_uint8:
            writer.append_data(frame)

        writer.close()

    def create_frame_save_callback(self,
                                   batch_idx: int,
                                   save_config: SaveConfig,
                                   post_process_fn: Optional[Callable] = None) -> Callable:
        """
        Create a callback function for saving individual frames during sequential rendering.

        Args:
            batch_idx: Current batch index
            save_config: Save configuration
            post_process_fn: Optional function to post-process outputs before saving

        Returns:
            Callback function that saves a single frame
        """
        # Initialize video writers if needed
        if save_config.save_as_video and not self._frame_writers:
            self._frame_count = 0

        def save_callback(output: Dict[str, torch.Tensor], _batch_idx: int, view_idx: int):
            """Save a single rendered frame."""
            try:
                # Post-process if function provided
                if post_process_fn:
                    output_processed = post_process_fn(output)
                else:
                    output_processed = output

                if save_config.save_as_video:
                    # Initialize writers on first frame
                    if not self._frame_writers:
                        self._init_video_writers_for_frame(output_processed, batch_idx, save_config)

                    # Write frame to video
                    self._write_frame_to_video(output_processed, save_config)
                    self._frame_count += 1
                else:
                    # Save as individual image
                    self._save_single_frame(output_processed, batch_idx, view_idx, save_config)

                # Free memory immediately after saving
                del output_processed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to save frame {view_idx} of batch {batch_idx}: {e}")

        return save_callback

    def _init_video_writers_for_frame(self,
                                      first_frame: Dict[str, torch.Tensor],
                                      batch_idx: int,
                                      save_config: SaveConfig) -> None:
        """Initialize video writers based on first frame."""
        # Get frame dimensions
        if 'rgb' in first_frame:
            shape = first_frame['rgb'].shape
            if len(shape) == 5:  # [B, N, C, H, W]
                _, _, _, H, W = shape
            elif len(shape) == 4:  # [B, C, H, W]
                _, _, H, W = shape
            else:
                H, W = shape[-2:]
        else:
            # Use any available tensor to get dimensions
            for key, tensor in first_frame.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    H, W = tensor.shape[-2:]
                    break

        # Create writers for each output type
        # Calculate CRF value from quality setting
        crf_value = 51 - (save_config.video_quality * 5)  # Map 0-10 to 51-1
        crf_value = max(1, min(51, int(crf_value)))  # Clamp to valid range

        if save_config.rgb and 'rgb' in first_frame:
            path = join(self.videos_dir, f"batch_{batch_idx:06d}_rgb.mp4")
            self._frame_writers['rgb'] = imageio.get_writer(
                path, fps=save_config.video_fps,
                codec=save_config.video_codec,
                ffmpeg_params=['-crf', str(crf_value), '-pix_fmt', 'yuv420p']
            )
            logger.info(f"Started RGB video: {blue(path)}")

        if save_config.depth and 'depth' in first_frame:
            path = join(self.videos_dir, f"batch_{batch_idx:06d}_depth.mp4")
            self._frame_writers['depth'] = imageio.get_writer(
                path, fps=save_config.video_fps,
                codec=save_config.video_codec,
                ffmpeg_params=['-crf', str(crf_value), '-pix_fmt', 'yuv420p']
            )
            logger.info(f"Started depth video: {blue(path)}")

    def _write_frame_to_video(self,
                              frame_output: Dict[str, torch.Tensor],
                              save_config: SaveConfig) -> None:
        """Write a single frame to the appropriate video writers."""
        # Write RGB
        if save_config.rgb and 'rgb' in frame_output and 'rgb' in self._frame_writers:
            frame = self._prepare_frame_for_video(frame_output['rgb'], normalize=True)  # RGB needs normalization
            self._frame_writers['rgb'].append_data(frame)

        # Write depth
        if save_config.depth and 'depth' in frame_output and 'depth' in self._frame_writers:
            depth_vis = self._visualize_depth(frame_output['depth'])
            frame = self._prepare_frame_for_video(depth_vis, normalize=False)  # Already in [0, 1]
            self._frame_writers['depth'].append_data(frame)

    def _prepare_frame_for_video(self, tensor: torch.Tensor, normalize: bool = False) -> np.ndarray:
        """Convert tensor to numpy array suitable for video writing.

        Args:
            tensor: Input tensor
            normalize: If True, normalize from [-1, 1] to [0, 1] before converting to uint8
        """
        # Handle various input shapes
        if tensor.dim() == 5:  # [B, N, C, H, W]
            tensor = tensor[0, 0]  # Take first batch, first view
        elif tensor.dim() == 4:  # [B, C, H, W] or [N, C, H, W]
            tensor = tensor[0]  # Take first
        # Now tensor is [C, H, W]

        # Normalize if needed (RGB images from model are typically in [-1, 1])
        if normalize:
            tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]

        frame = tensor.cpu().numpy()
        if frame.shape[0] == 1:  # Grayscale
            frame = frame.squeeze(0)  # [H, W]
        else:  # RGB
            frame = frame.transpose(1, 2, 0)  # [H, W, C]

        return norm_to_uint8(frame)

    def _save_single_frame(self,
                           output: Dict[str, torch.Tensor],
                           batch_idx: int,
                           view_idx: int,
                           save_config: SaveConfig) -> None:
        """Save a single frame as an image file."""
        # Save RGB
        if save_config.rgb and 'rgb' in output:
            path = join(self.images_dir, f"batch_{batch_idx:06d}_frame_{view_idx:04d}_rgb.png")
            mkdirs(dirname(path))

            # Extract single frame and normalize
            rgb_frame = output['rgb']
            # Handle different shapes
            if rgb_frame.dim() == 5:  # [B, N, C, H, W]
                rgb_frame = rgb_frame[0, 0]  # -> [C, H, W]
            elif rgb_frame.dim() == 4:  # [B, C, H, W]
                rgb_frame = rgb_frame[0]  # -> [C, H, W]

            # Normalize from [-1, 1] to [0, 1]
            rgb_frame = (rgb_frame + 1) / 2
            utils.save_image(rgb_frame, path)

        # Save depth
        if save_config.depth and 'depth' in output:
            if save_config.save_raw:
                depth_path = join(self.images_dir, f"batch_{batch_idx:06d}_frame_{view_idx:04d}_depth.npz")
                mkdirs(dirname(depth_path))
                depth_frame = output['depth']
                if depth_frame.dim() == 5:  # [B, N, C, H, W]
                    depth_frame = depth_frame[0, 0]  # -> [C, H, W]
                elif depth_frame.dim() == 4:  # [B, C, H, W]
                    depth_frame = depth_frame[0]  # -> [C, H, W]
                np.savez_compressed(depth_path, depth=depth_frame.cpu().numpy())

            if save_config.save_visualization:
                depth_vis = self._visualize_depth(output['depth'])
                path = join(self.images_dir, f"batch_{batch_idx:06d}_frame_{view_idx:04d}_depth_vis.png")
                mkdirs(dirname(path))

                # Extract single frame if needed
                if depth_vis.dim() == 5:  # [B, N, C, H, W]
                    depth_vis = depth_vis[0, 0]  # -> [C, H, W]
                elif depth_vis.dim() == 4:  # [B, C, H, W]
                    depth_vis = depth_vis[0]  # -> [C, H, W]

                utils.save_image(depth_vis, path)

    def _save_tensor_as_image(self, tensor: torch.Tensor, path: str, normalize: bool = True) -> None:
        """Save a tensor as an image file.

        Args:
            tensor: Tensor to save
            path: Path to save to
            normalize: If True, normalize from [-1, 1] to [0, 1] (for RGB images)
        """
        mkdirs(dirname(path))

        # Handle different tensor shapes
        if tensor.dim() == 5:  # [B, N, C, H, W]
            # Create grid of images
            shape = tensor.shape
            B, N = shape[0], shape[1]
            tensor = tensor.reshape(B * N, *shape[2:])

        # Normalize if needed (RGB images are typically in [-1, 1])
        if normalize:
            tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]

        # Use torchvision's save_image
        utils.save_image(tensor, path)

    def _visualize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Convert depth to RGB visualization."""
        # Normalize depth to [0, 1]
        if depth.dim() == 5:  # [B, N, 1, H, W]
            shape = depth.shape
            B, N, _, H, W = shape
            depth_flat = depth.reshape(B * N, 1, H, W)
        else:
            depth_flat = depth

        depth_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min() + 1e-8)

        # Apply colormap (simple jet-like colormap)
        # For now, just repeat to make RGB
        depth_rgb = depth_norm.repeat(1, 3, 1, 1)

        if depth.dim() == 5:
            depth_rgb = depth_rgb.reshape(B, N, 3, H, W)

        return depth_rgb

    def _visualize_normal(self, normal: torch.Tensor) -> torch.Tensor:
        """Convert normal map to RGB visualization."""
        # Normals are in [-1, 1], map to [0, 1]
        return (normal + 1) / 2

    def _visualize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Convert optical flow to RGB visualization."""
        # Simple visualization: normalize and use first 2 channels as R,G
        if flow.shape[-3] >= 2:  # Has at least 2 channels
            flow_rg = flow[..., :2, :, :]  # Take first 2 channels
            # Normalize to [0, 1]
            flow_norm = (flow_rg - flow_rg.min()) / (flow_rg.max() - flow_rg.min() + 1e-8)
            # Add blue channel
            B = flow_norm.shape[0] if flow_norm.dim() == 4 else 1
            blue = torch.zeros_like(flow_norm[..., :1, :, :])
            flow_rgb = torch.cat([flow_norm, blue], dim=-3)
            return flow_rgb
        return flow

    def close_video_writers(self) -> None:
        """Close all open video writers."""
        for writer in self._frame_writers.values():
            writer.close()
        self._frame_writers = {}
        self._frame_count = 0
        logger.info("Closed all video writers")

    def finalize_videos(self, batch_idx: int, save_config: SaveConfig) -> None:
        """Finalize video files after sequential rendering."""
        if self._frame_writers:
            self.close_video_writers()
            logger.info(f"Finalized videos for batch {batch_idx}")

    def save_gaussians(self,
                       gaussian_data: Dict[str, torch.Tensor],
                       batch_idx: int,
                       save_config: SaveConfig,
                       unalign_output: bool = False,
                       c2w_avg: Optional[torch.Tensor] = None) -> str:
        """
        Save 4D Gaussian parameters to disk.

        Args:
            gaussian_data: Dictionary containing Gaussian parameters
            batch_idx: Batch index for naming
            save_config: Save configuration
            unalign_output: Whether to unalign the output
            c2w_avg: Average camera-to-world transform for unalignment

        Returns:
            Path to saved file
        """
        if not save_config.gaussians:
            return None

        # Prepare Gaussian data
        gs = gaussian_data.get('gs', gaussian_data)

        # Unalign if needed
        if unalign_output and c2w_avg is not None:
            gs = self._unalign_gaussians(gs, c2w_avg)

        # Convert to save format
        fdgs = dotdict()
        if "xyz" in gs:
            fdgs.xyz3 = gs["xyz"].float().detach().cpu().numpy()
        if "feature" in gs:
            fdgs.rgb3 = gs["feature"][..., :3].float().detach().cpu().numpy()
        if "opacity" in gs:
            fdgs.occ1 = gs["opacity"].float().detach().cpu().numpy()
        if "scaling" in gs:
            fdgs.scale3 = gs["scaling"].float().detach().cpu().numpy()
        if "rotation" in gs:
            fdgs.rot4 = gs["rotation"].float().detach().cpu().numpy()
        if "t" in gs:
            fdgs.t1 = gs["t"].float().detach().cpu().numpy()

        # Additional 4D parameters if present
        if "ms3" in gs:
            fdgs.ms3 = gs["ms3"].float().detach().cpu().numpy()
        if "omega" in gs:
            fdgs.omega3 = gs["omega"].float().detach().cpu().numpy()
        if "cov_t" in gs:
            fdgs.cov_t1 = gs["cov_t"].float().detach().cpu().numpy()

        # Save to file
        gs_path = join(self.gaussians_dir, f"batch_{batch_idx:06d}_gs.npz")
        mkdirs(dirname(gs_path))

        if save_config.save_gaussians_async:
            # Save asynchronously - check if executor is still running
            try:
                future = self.gaussian_save_executor.submit(
                    self._save_gaussians_sync, fdgs, gs_path
                )
                self.gaussian_futures.append(future)
                logger.info(f"Queued Gaussian save for batch {batch_idx}")
            except RuntimeError as e:
                if "cannot schedule new futures" in str(e):
                    logger.warning(f"Executor shutting down, saving batch {batch_idx} synchronously")
                    self._save_gaussians_sync(fdgs, gs_path)
                else:
                    raise
        else:
            # Save synchronously
            self._save_gaussians_sync(fdgs, gs_path)

        return gs_path

    def _save_gaussians_sync(self, fdgs: dotdict, gs_path: str) -> None:
        """Synchronously save Gaussian data."""
        with open(gs_path, "wb") as fp:
            np.savez_compressed(fp, **fdgs)
        logger.info(f"Saved 4D Gaussians to {blue(gs_path)}")

    def _unalign_gaussians(self, gs: Dict[str, torch.Tensor], c2w_avg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unalign Gaussian parameters using average camera transform."""
        try:
            from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
        except ImportError:
            logger.warning("pytorch3d not available, using simplified unalignment")
            # Simplified version without pytorch3d
            return gs

        # Clone to avoid modifying original
        gs = {k: v.clone() for k, v in gs.items()}

        # Transform xyz
        gs['xyz'] = gs['xyz'] @ c2w_avg[..., :3, :3].mT + c2w_avg[..., None, :3, 3]

        # Transform rotation
        gs['rotation'] = matrix_to_quaternion(
            c2w_avg[..., :3, :3] @ quaternion_to_matrix(gs['rotation'])
        )

        # Transform ms3 if present
        if 'ms3' in gs:
            sh = gs['ms3'].shape
            ms3_deg = gs['ms3'].shape[-1] // 3
            gs['ms3'] = gs['ms3'].reshape(*sh[:-1], ms3_deg, 3)
            gs['ms3'] = gs['ms3'] @ c2w_avg[..., None, :3, :3].mT
            gs['ms3'] = gs['ms3'].reshape(*sh)

        # Transform omega if present
        if 'omega' in gs:
            sh = gs['omega'].shape
            omega3_deg = gs['omega'].shape[-1] // 3
            gs['omega'] = gs['omega'].reshape(*sh[:-1], omega3_deg, 3)
            angle = gs['omega'].norm(dim=-1, keepdim=True)
            axis = gs['omega'] / (angle + 1e-8)
            axis = axis @ c2w_avg[..., None, :3, :3].mT
            gs['omega'] = angle * axis
            gs['omega'] = gs['omega'].reshape(*sh)

        return gs

    def load_gaussians(self, batch_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load saved Gaussian parameters from disk.

        Args:
            batch_idx: Batch index to load

        Returns:
            Dictionary of Gaussian parameters or None if not found
        """
        gs_path = join(self.gaussians_dir, f"batch_{batch_idx:06d}_gs.npz")

        if not exists(gs_path):
            return None

        try:
            with np.load(gs_path) as data:
                fdgs = dotdict({k: data[k] for k in data.files})

            # Convert back to standard naming
            gs = {}
            if 'xyz3' in fdgs:
                gs['xyz'] = fdgs.xyz3
            if 'rgb3' in fdgs:
                gs['feature'] = np.concatenate([fdgs.rgb3, np.zeros_like(fdgs.rgb3)], axis=-1)
            if 'occ1' in fdgs:
                gs['opacity'] = fdgs.occ1
            if 'scale3' in fdgs:
                gs['scaling'] = fdgs.scale3
            if 'rot4' in fdgs:
                gs['rotation'] = fdgs.rot4
            if 't1' in fdgs:
                gs['t'] = fdgs.t1
            if 'ms3' in fdgs:
                gs['ms3'] = fdgs.ms3
            if 'omega3' in fdgs:
                gs['omega'] = fdgs.omega3
            if 'cov_t1' in fdgs:
                gs['cov_t'] = fdgs.cov_t1

            logger.info(f"Loaded Gaussians from {blue(gs_path)}")
            return gs

        except Exception as e:
            logger.error(f"Failed to load Gaussians from {gs_path}: {e}")
            return None

    def queue_gaussians_for_rendering(self, gaussian_data: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Add Gaussian parameters to the rendering queue.

        Args:
            gaussian_data: Gaussian parameters to queue
            batch_idx: Batch index
        """
        try:
            self.gaussian_queue.put((batch_idx, gaussian_data), timeout=60)
            logger.info(f"Queued batch {batch_idx} for rendering (queue size: {self.gaussian_queue.qsize()})")
        except:
            logger.warning(f"Failed to queue batch {batch_idx} - queue may be full")

    def get_next_gaussians(self, timeout: float = 1.0) -> Optional[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Get the next Gaussian parameters from the queue.

        Args:
            timeout: How long to wait for data

        Returns:
            Tuple of (batch_idx, gaussian_data) or None if timeout
        """
        try:
            return self.gaussian_queue.get(timeout=timeout)
        except:
            return None

    def wait_for_all_saves(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all pending save operations to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self.save_futures:
            logger.info(f"Waiting for {len(self.save_futures)} save operations to complete...")
            done, not_done = concurrent.futures.wait(
                self.save_futures,
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED
            )

            # Check for exceptions
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Save operation failed: {e}")

            if not_done:
                logger.warning(f"{len(not_done)} save operations did not complete within timeout")

            # Clear completed futures
            self.save_futures = list(not_done)

    def cleanup(self) -> None:
        """Clean up resources."""
        # Close video writers
        if self._frame_writers:
            self.close_video_writers()

        # Wait for pending saves
        self.wait_for_all_saves(timeout=30)

        # Wait for Gaussian saves
        if self.gaussian_futures:
            logger.info(f"Waiting for {len(self.gaussian_futures)} Gaussian save operations...")
            done, not_done = concurrent.futures.wait(
                self.gaussian_futures,
                timeout=30,
                return_when=concurrent.futures.ALL_COMPLETED
            )
            if not_done:
                logger.warning(f"{len(not_done)} Gaussian saves did not complete")

        # Shutdown executors
        self.save_executor.shutdown(wait=True)
        self.gaussian_save_executor.shutdown(wait=True)
        logger.info("VideoSystem cleanup complete")
