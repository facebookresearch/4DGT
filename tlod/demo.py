#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
4DGT Demo Class for model inference.

This module provides a clean interface for 4DGT model inference,
focusing on model operations and delegating file I/O to VideoSystem.
"""

import os
import torch
import time
import logging
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict
import concurrent.futures
import threading
from queue import Queue

from mmengine import Config as MMConfig
from tlod.easyvolcap.utils.console_utils import logger, blue, green, tqdm
from tlod.misc import utils

# Import VideoSystem for file handling
from tlod.video_system import VideoSystem, SaveConfig

# Import model and renderers to register them
from tlod.models.gaussian_model import GaussianModel  # noqa: F401
from tlod.renderers.tlod_renderer import TLODRenderer  # noqa: F401
from tlod.renderers.gaussian_renderer import GaussianRenderer  # noqa: F401


class FourDGTDemo:
    """
    4DGT Demo class for model inference.

    This class encapsulates all model-related operations including:
    - Model initialization and checkpoint loading
    - Inference on prepared batches
    - Post-processing of outputs
    - Async video saving for efficiency
    """

    def __init__(self,
                 config_path: str,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 output_dir: str = "outputs",
                 fp16: bool = False,
                 fp32: bool = False):
        """
        Initialize 4DGT Demo.

        Args:
            config_path: Path to model configuration file
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to run inference on ('cuda' or 'cpu')
            output_dir: Directory to save outputs
            fp16: Use float16 precision
            fp32: Use float32 precision (default is bfloat16)
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = output_dir

        # Set precision
        if fp16:
            self.dtype = torch.float16
        elif fp32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16

        # Initialize video system for file handling
        self.video_system = VideoSystem(output_dir=self.output_dir, verbose=True)

        # Suppress imageio warnings if needed
        logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

        # Initialize timing stats
        self.timing_stats = defaultdict(list)
        self.enable_profiling = True

        # Initialize model
        self.model = None
        self.model_cfg = None
        self._load_model()

    def _load_model_config(self) -> MMConfig:
        """Load model configuration from Python file."""
        config_path = os.path.abspath(self.config_path)
        logger.info(f"Loading model config from {blue(config_path)}")
        cfg = MMConfig.fromfile(config_path)
        return cfg

    def _load_model(self) -> None:
        """Initialize model and load checkpoint."""
        from tlod.registry import build_module, MODELS

        # Load configuration
        self.model_cfg = self._load_model_config()

        logger.info("Building model...")
        # Build model using the registry system
        self.model = build_module(
            self.model_cfg.model,
            MODELS,
            image_encoder_config=self.model_cfg.image_encoder,
            renderer_config=self.model_cfg.renderer if hasattr(self.model_cfg, 'renderer') else None,
        )
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Load checkpoint if provided
        if self.checkpoint_path:
            logger.info(f"Loading checkpoint from {blue(self.checkpoint_path)}")
            utils.restart_from_checkpoint(
                self.checkpoint_path,
                model=self.model,
                strict=False  # Allow partial loading for flexibility
            )

        # Calculate and display model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # Format parameter counts with commas for readability
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Model type: {self.model.__class__.__name__}")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        if trainable_params != total_params:
            logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
            logger.info(f"  Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.1f}M)")

        # Show model size in memory
        param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        logger.info(f"  Estimated model size: {param_size_mb:.1f} MB (float32)")

        # Show key model components if available
        if hasattr(self.model, 'image_coder'):
            logger.info(f"  Image encoder: {self.model.image_coder.__class__.__name__}")
        if hasattr(self.model, 'renderer'):
            logger.info(f"  Renderer: {self.model.renderer.__class__.__name__ if self.model.renderer else 'None'}")

    def prepare_cameras(self,
                        cameras: torch.Tensor,
                        height: int,
                        width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare camera parameters for the model.

        Args:
            cameras: Camera parameters tensor [B, N, 20]
            height: Image height
            width: Image width

        Returns:
            extrinsics: Camera extrinsics [B, N, 4, 4]
            intrinsics: Camera intrinsics [B, N, 3, 3]
        """
        # Extract extrinsics and convert from OpenGL to OpenCV
        from einops import rearrange
        exts = rearrange(cameras[..., :16].clone(), '... (h w) -> ... h w', h=4, w=4)  # [B, N, 16] -> [B, N, 4, 4]
        exts[..., :3, 1:3] = -exts[..., :3, 1:3]  # [B, N, 4, 4] flip Y and Z axes (OpenGL to OpenCV)

        # Compute intrinsics from FOV and principal point
        ixts = torch.zeros(*cameras.shape[:2], 3, 3)  # [B, N, 3, 3]
        ixts[..., 0, 0] = width / (2 * torch.tan(cameras[..., 16] / 2))  # fx from fov_x
        ixts[..., 1, 1] = height / (2 * torch.tan(cameras[..., 17] / 2))  # fy from fov_y
        ixts[..., 0, 2] = width * cameras[..., 18]  # cx from principal_x
        ixts[..., 1, 2] = height * cameras[..., 19]  # cy from principal_y
        ixts[..., 2, 2] = 1  # homogeneous coordinate

        return exts, ixts

    def _record_time(self, category: str, duration: float) -> None:
        """Record timing statistics."""
        if self.enable_profiling:
            self.timing_stats[category].append(duration)

    @torch.inference_mode()
    def encode(self,
               images_input: torch.Tensor,
               cameras_input: torch.Tensor,
               timestamps_input: torch.Tensor,
               height: Optional[int] = None,
               width: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Encode input images to Gaussian parameters.

        Args:
            images_input: Input images [B, N_in, C, H, W]
            cameras_input: Input camera parameters [B, N_in, 20]
            timestamps_input: Input timestamps [B, N_in]
            height: Output height (defaults to input height)
            width: Output width (defaults to input width)

        Returns:
            Dictionary containing Gaussian parameters with timestamp info
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _load_model() first.")

        # Get dimensions
        _, _, _, height_in, width_in = images_input.shape  # [B, N_in, C, H, W]
        height_out = height or height_in
        width_out = width or width_in

        # Prepare timestamps
        t_input = timestamps_input[..., None]  # [B, N_in] -> [B, N_in, 1]

        # Prepare camera parameters
        exts_input, ixts_input = self.prepare_cameras(cameras_input, height_in, width_in)  # [B, N_in, 20] -> [B, N_in, 4, 4], [B, N_in, 3, 3]

        # Move to device
        images_input = images_input.to(self.device, non_blocking=True)
        t_input = t_input.to(self.device, non_blocking=True)
        ixts_input = ixts_input.to(self.device, non_blocking=True)
        exts_input = exts_input.to(self.device, non_blocking=True)

        if self.device != 'cpu':
            torch.cuda.synchronize()

        # Run encoder with autocast
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=(self.device != 'cpu')):
            start_time = time.perf_counter()

            # Call the encoder directly
            gaussian_parameters = self.model.image_coder(
                images_input,
                t_input,
                ixts_input,
                exts_input,
                height=height_out,
                width=width_out,
            )

            if self.device != 'cpu':
                torch.cuda.synchronize()

            encoding_time = time.perf_counter() - start_time
            # Log encoding time (this is usually quick and worth logging)
            logger.info(f"Encoding time: {encoding_time:.3f}s")

        # Store metadata with Gaussians
        result = {
            'gaussian_parameters': gaussian_parameters,
            'timestamps': timestamps_input,  # Keep original timestamps
            'height': height_out,
            'width': width_out,
            'encoding_time': encoding_time
        }

        return result

    @torch.inference_mode()
    def render_image(self,
                     gaussian_data: Dict[str, torch.Tensor],
                     camera: torch.Tensor,
                     timestamp: torch.Tensor,
                     render_mode: str = 'RGB+ED',
                     monochrome: Optional[torch.Tensor] = None,
                     ratio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Render a single image from Gaussian parameters.

        This function is designed for memory-efficient sequential rendering.
        Unlike render_batch which processes all views at once, this renders
        one view at a time, allowing for immediate saving and memory release.

        Args:
            gaussian_data: Dictionary containing Gaussian parameters from encode()
            camera: Single camera parameters [1, 1, 20] or [1, 20]
            timestamp: Single timestamp [1, 1] or [1]
            render_mode: Rendering mode - one of ['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED']
            monochrome: Monochrome flag [1, 1] or [1] (optional)
            ratio: Aspect ratio [1, 1] or [1] (optional)

        Returns:
            Dictionary containing single rendered image

        Example:
            # Render images sequentially for memory efficiency
            for i in range(num_views):
                output = demo.render_image(
                    gaussian_data,
                    cameras[0, i:i+1],  # [1, 1, 20]
                    timestamps[0, i:i+1]  # [1, 1]
                )
                save_and_free(output)  # Save immediately and free memory
        """
        # Ensure proper shape for single image
        if camera.dim() == 2:  # [1, 20]
            camera = camera.unsqueeze(1)  # [1, 20] -> [1, 1, 20]
        if timestamp.dim() == 1:  # [1]
            timestamp = timestamp.unsqueeze(1)  # [1] -> [1, 1]
        if monochrome is not None and monochrome.dim() == 1:
            monochrome = monochrome.unsqueeze(1)  # [1] -> [1, 1]
        if ratio is not None and ratio.dim() == 1:
            ratio = ratio.unsqueeze(1)  # [1] -> [1, 1]

        # Use render_batch for single image
        return self.render_batch(
            gaussian_data=gaussian_data,
            cameras_output=camera,
            timestamps_output=timestamp,
            render_mode=render_mode,
            monochrome=monochrome,
            ratios=ratio
        )

    @torch.inference_mode()
    def render_batch(self,
                     gaussian_data: Dict[str, torch.Tensor],
                     cameras_output: torch.Tensor,
                     timestamps_output: torch.Tensor,
                     render_mode: str = 'RGB+ED',
                     monochrome: Optional[torch.Tensor] = None,
                     ratios: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Render multiple views from Gaussian parameters in batch.

        This function processes all views at once, which is faster but uses
        more memory. For memory-constrained scenarios or when rendering many
        views, consider using render_image() in a loop instead.

        Args:
            gaussian_data: Dictionary containing Gaussian parameters from encode()
            cameras_output: Output camera parameters [B, N_out, 20]
            timestamps_output: Output timestamps [B, N_out]
            render_mode: Rendering mode - one of ['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED']
            monochrome: Monochrome flags for output views [B, N_out]
            ratios: Aspect ratios for output views [B, N_out]

        Returns:
            Dictionary containing rendered outputs for all views
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _load_model() first.")

        # Extract Gaussian parameters and metadata
        gaussian_parameters = gaussian_data['gaussian_parameters']
        height_out = gaussian_data['height']
        width_out = gaussian_data['width']

        # Prepare output timestamps
        t_output = timestamps_output[..., None]  # [B, N_out] -> [B, N_out, 1]

        # Prepare output camera parameters
        exts_output, ixts_output = self.prepare_cameras(cameras_output, height_out, width_out)  # [B, N_out, 20] -> [B, N_out, 4, 4], [B, N_out, 3, 3]

        # Move to device
        t_output = t_output.to(self.device, non_blocking=True)
        ixts_output = ixts_output.to(self.device, non_blocking=True)
        exts_output = exts_output.to(self.device, non_blocking=True)

        if self.device != 'cpu':
            torch.cuda.synchronize()

        # Run renderer with autocast
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=(self.device != 'cpu')):
            start_time = time.perf_counter()

            # Call the renderer directly
            output = self.model.render(
                gaussian_parameters,
                t_output,
                ixts_output,
                exts_output,
                skip_render=False,
                render_mode=render_mode,
                monochrome=monochrome,
                ratios=ratios,
                height=height_out,
                width=width_out,
            )

            if self.device != 'cpu':
                torch.cuda.synchronize()

            rendering_time = time.perf_counter() - start_time
            # Only log in batch mode, sequential mode uses progress bar
            if not hasattr(self, '_suppress_render_log'):
                logger.info(f"Rendering time: {rendering_time:.3f}s")

        # Add timing info
        output['rendering_time'] = rendering_time

        return output

    @torch.inference_mode()
    def render(self,
               gaussian_data: Dict[str, torch.Tensor],
               cameras_output: torch.Tensor,
               timestamps_output: torch.Tensor,
               render_mode: str = 'RGB+ED',
               monochrome: Optional[torch.Tensor] = None,
               ratios: Optional[torch.Tensor] = None,
               sequential: bool = False,
               save_callback: Optional[callable] = None,
               return_outputs: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        """
        Flexible render function that can use batch or sequential rendering.

        Args:
            gaussian_data: Dictionary containing Gaussian parameters from encode()
            cameras_output: Output camera parameters [B, N_out, 20]
            timestamps_output: Output timestamps [B, N_out]
            render_mode: Rendering mode - one of ['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED']
            monochrome: Monochrome flags for output views [B, N_out]
            ratios: Aspect ratios for output views [B, N_out]
            sequential: If True, render images one by one to save memory
            save_callback: Optional function to call after each image (for sequential mode)
            return_outputs: If False and save_callback is provided, don't store outputs in memory

        Returns:
            Dictionary containing rendered outputs, or None if return_outputs=False with callback

        Note:
            - sequential=False (default): Fast batch rendering, high memory usage
            - sequential=True: Slower but memory-efficient, good for many views
        """
        if sequential:
            # Sequential rendering for memory efficiency
            B, N_out = cameras_output.shape[:2]
            all_outputs = [] if return_outputs else None

            # Create progress bar for sequential rendering
            total_frames = B * N_out
            start_sequential = time.perf_counter()
            with tqdm(total=total_frames, desc="Rendering frames", unit="frame") as pbar:
                for b in range(B):
                    for n in range(N_out):
                        # Update progress bar with current frame info
                        pbar.set_postfix({
                            'batch': f'{b+1}/{B}',
                            'view': f'{n+1}/{N_out}',
                            'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                        })

                        # Render single image (suppress individual render logs)
                        self._suppress_render_log = True
                        try:
                            single_output = self.render_image(
                                gaussian_data=gaussian_data,
                                camera=cameras_output[b:b + 1, n:n + 1],  # [1, 1, 20]
                                timestamp=timestamps_output[b:b + 1, n:n + 1],  # [1, 1]
                                render_mode=render_mode,
                                monochrome=monochrome[b:b + 1, n:n + 1] if monochrome is not None else None,
                                ratio=ratios[b:b + 1, n:n + 1] if ratios is not None else None
                            )
                        finally:
                            delattr(self, '_suppress_render_log')

                        # Optional callback for immediate saving
                        if save_callback is not None:
                            save_callback(single_output, b, n)

                            # Clear GPU memory after saving if not returning outputs
                            if not return_outputs:
                                # Delete tensors to free memory immediately
                                for key in list(single_output.keys()):
                                    if isinstance(single_output[key], torch.Tensor):
                                        del single_output[key]
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                        # Only store outputs if we need to return them
                        if return_outputs:
                            all_outputs.append(single_output)

                        # Update progress bar
                        pbar.update(1)

            # Log summary after sequential rendering
            total_time = time.perf_counter() - start_sequential
            fps = total_frames / total_time if total_time > 0 else 0
            logger.info(f"Sequential rendering complete: {total_frames} frames in {total_time:.1f}s ({fps:.1f} fps)")

            # If not returning outputs, we're done
            if not return_outputs:
                return None

            # Combine outputs only if we stored them
            if all_outputs:
                combined = {}
                for key in all_outputs[0].keys():
                    if key == 'rendering_time':
                        continue
                    elif key == 'gs':
                        # Gaussian parameters are special - just take from first (they're the same)
                        combined[key] = all_outputs[0][key]
                    else:
                        # Stack tensors along view dimension
                        views = [out[key] for out in all_outputs]
                        if isinstance(views[0], torch.Tensor):
                            combined[key] = torch.cat(views, dim=1)  # [B, 1, ...] -> [B, N, ...]
                        else:
                            # For non-tensor outputs, just take the first
                            combined[key] = views[0]

                return combined
            else:
                return None
        else:
            # Batch rendering (existing behavior)
            return self.render_batch(
                gaussian_data=gaussian_data,
                cameras_output=cameras_output,
                timestamps_output=timestamps_output,
                render_mode=render_mode,
                monochrome=monochrome,
                ratios=ratios
            )

    def run_async_inference(self,
                            dataloader,
                            save_config: Optional[SaveConfig] = None,
                            max_parallel_batches: int = 3) -> None:
        """
        Run inference with asynchronous Gaussian generation and rendering.

        This method runs Gaussian generation and rendering in parallel:
        1. Gaussian generation runs in one thread, saving results to queue
        2. Rendering runs in another thread, loading from queue
        3. Main thread coordinates and monitors progress

        Args:
            dataloader: DataLoader providing batches
            save_config: Configuration for saving
            max_parallel_batches: Maximum number of batches to process in parallel
        """
        if save_config is None:
            save_config = SaveConfig(gaussians=True, save_gaussians_async=True)

        # Queues for communication
        batch_queue = Queue(maxsize=max_parallel_batches)
        rendering_done = threading.Event()
        shutdown_event = threading.Event()  # For graceful shutdown
        
        # Statistics
        stats = {
            'total_batches': len(dataloader),
            'gaussians_generated': 0,
            'frames_rendered': 0,
            'generation_time': [],
            'rendering_time': []
        }

        # Gaussian generation thread
        def gaussian_generator():
            """Generate Gaussians and queue them for rendering."""
            try:
                for batch_idx, batch in enumerate(dataloader):
                    # Check for shutdown
                    if shutdown_event.is_set():
                        logger.info("[Generator] Shutdown requested, stopping generation")
                        break
                    start_time = time.time()

                    # Handle list wrapper if present
                    if isinstance(batch, list):
                        batch = batch[0]

                    # Extract inputs
                    images_input = batch["rgb_input"]
                    cameras_input = batch["cameras_input"]
                    timestamps_input = batch["rays_t_un_input"]

                    # Handle monochrome cameras if present
                    if 'mono_input' in batch:
                        monos = batch["mono_input"]
                        B, _, C1, H, W = images_input.shape
                        _, N2, _, _, _ = monos.shape
                        images_input = torch.cat([images_input, monos.expand(B, N2, C1, H, W)], dim=1)

                    # Generate Gaussians
                    logger.info(f"[Generator] Processing batch {batch_idx + 1}/{stats['total_batches']}")
                    gaussian_data = self.encode(
                        images_input=images_input,
                        cameras_input=cameras_input,
                        timestamps_input=timestamps_input
                    )
                    
                    # Save Gaussians asynchronously (check for shutdown first)
                    if save_config.gaussians and not shutdown_event.is_set():
                        gs_path = self.video_system.save_gaussians(
                            gaussian_data['gaussian_parameters'],
                            batch_idx,
                            save_config
                        )
                        gaussian_data['saved_path'] = gs_path

                    # Queue for rendering
                    batch_data = {
                        'batch_idx': batch_idx,
                        'gaussian_data': gaussian_data,
                        'cameras_output': batch.get("cameras_output"),
                        'timestamps_output': batch.get("rays_t_un_output"),
                        'monochrome': batch.get("monochrome_output"),
                        'ratios': batch.get("ratios_output")
                    }

                    batch_queue.put(batch_data)
                    stats['gaussians_generated'] += 1
                    stats['generation_time'].append(time.time() - start_time)

                    logger.info(f"[Generator] Queued batch {batch_idx} (queue size: {batch_queue.qsize()})")
                    
            except KeyboardInterrupt:
                logger.info("[Generator] Interrupted by user")
                shutdown_event.set()
            except Exception as e:
                logger.error(f"[Generator] Error: {e}")
                shutdown_event.set()
            finally:
                # Signal completion
                if not shutdown_event.is_set():
                    batch_queue.put(None)
                logger.info("[Generator] Finished generating all Gaussians")

        # Rendering thread
        def renderer():
            """Render frames from queued Gaussians."""
            try:
                while not shutdown_event.is_set():
                    try:
                        # Get next batch with timeout to check for shutdown
                        batch_data = batch_queue.get(timeout=1.0)
                        if batch_data is None:
                            break  # Generation complete
                    except:
                        continue  # Timeout, check shutdown and continue
                    
                    start_time = time.time()
                    batch_idx = batch_data['batch_idx']

                    logger.info(f"[Renderer] Processing batch {batch_idx}")

                    # Create save callback
                    save_callback = None
                    if save_config:
                        save_callback = self.video_system.create_frame_save_callback(
                            batch_idx, save_config, post_process_fn=self.post_process_outputs
                        )

                    # Render sequentially with saving
                    output = self.render(
                        gaussian_data=batch_data['gaussian_data'],
                        cameras_output=batch_data['cameras_output'],
                        timestamps_output=batch_data['timestamps_output'],
                        render_mode='RGB+ED',
                        monochrome=batch_data['monochrome'],
                        ratios=batch_data['ratios'],
                        sequential=True,
                        save_callback=save_callback,
                        return_outputs=False  # Don't keep in memory
                    )

                    # Finalize videos if needed
                    if save_config and save_config.save_as_video:
                        self.video_system.finalize_videos(batch_idx, save_config)

                    stats['frames_rendered'] += 1
                    stats['rendering_time'].append(time.time() - start_time)

                    logger.info(f"[Renderer] Completed batch {batch_idx}")
                    
            except KeyboardInterrupt:
                logger.info("[Renderer] Interrupted by user")
                shutdown_event.set()
            except Exception as e:
                logger.error(f"[Renderer] Error: {e}")
                shutdown_event.set()
            finally:
                rendering_done.set()
                logger.info("[Renderer] Finished rendering all frames")

        # Start threads
        logger.info("Starting asynchronous inference pipeline...")
        generator_thread = threading.Thread(target=gaussian_generator, name="GaussianGenerator")
        renderer_thread = threading.Thread(target=renderer, name="Renderer")

        generator_thread.start()
        renderer_thread.start()
        
        # Monitor progress with proper shutdown handling
        try:
            with tqdm(total=stats['total_batches'], desc="Overall Progress") as pbar:
                while generator_thread.is_alive() or renderer_thread.is_alive():
                    # Update progress
                    pbar.n = stats['frames_rendered']
                    pbar.set_postfix({
                        'generated': stats['gaussians_generated'],
                        'rendered': stats['frames_rendered'],
                        'queue': batch_queue.qsize()
                    })
                    pbar.refresh()
                    time.sleep(0.5)
                
                # Final update
                pbar.n = stats['frames_rendered']
                pbar.refresh()
        except KeyboardInterrupt:
            logger.info("\n[Main] Shutdown requested, waiting for threads to finish...")
            shutdown_event.set()
            
            # Drain the queue to unblock the generator
            try:
                while not batch_queue.empty():
                    batch_queue.get_nowait()
            except:
                pass
        
        # Wait for threads to complete with timeout
        generator_thread.join(timeout=5.0)
        renderer_thread.join(timeout=5.0)
        
        if generator_thread.is_alive() or renderer_thread.is_alive():
            logger.warning("Some threads did not shutdown cleanly")
        
        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("Asynchronous Inference Complete")
        logger.info("=" * 60)

        if stats['generation_time']:
            avg_gen = sum(stats['generation_time']) / len(stats['generation_time'])
            logger.info(f"Gaussian Generation: {avg_gen:.2f}s average per batch")

        if stats['rendering_time']:
            avg_render = sum(stats['rendering_time']) / len(stats['rendering_time'])
            logger.info(f"Rendering: {avg_render:.2f}s average per batch")

        total_time = sum(stats['generation_time']) + sum(stats['rendering_time'])
        logger.info(f"Total processing time: {total_time:.1f}s")

        # Cleanup
        self.video_system.cleanup()
        logger.info("Pipeline cleanup complete")

    def render_from_saved_gaussians(self,
                                    start_batch: int,
                                    end_batch: int,
                                    cameras_output: torch.Tensor,
                                    timestamps_output: torch.Tensor,
                                    render_mode: str = 'RGB+ED',
                                    save_config: Optional[SaveConfig] = None) -> None:
        """
        Render frames from previously saved Gaussian parameters.

        This method loads saved Gaussians and renders them without
        needing to run the encoder again.

        Args:
            start_batch: Starting batch index
            end_batch: Ending batch index (exclusive)
            cameras_output: Output camera parameters [B, N_out, 20]
            timestamps_output: Output timestamps [B, N_out]
            render_mode: Rendering mode
            save_config: Configuration for saving rendered outputs
        """
        if save_config is None:
            save_config = SaveConfig()

        logger.info(f"Rendering from saved Gaussians: batches {start_batch} to {end_batch-1}")

        for batch_idx in range(start_batch, end_batch):
            # Load saved Gaussians
            gs_data = self.video_system.load_gaussians(batch_idx)

            if gs_data is None:
                logger.warning(f"No saved Gaussians found for batch {batch_idx}, skipping")
                continue

            # Convert to torch tensors
            gaussian_data = {
                'gaussian_parameters': {
                    k: torch.from_numpy(v).to(self.device, self.dtype)
                    for k, v in gs_data.items()
                },
                'height': cameras_output.shape[-2],  # Infer from camera params
                'width': cameras_output.shape[-1]
            }

            logger.info(f"Rendering batch {batch_idx} from loaded Gaussians")

            # Create save callback if needed
            save_callback = None
            if save_config:
                save_callback = self.video_system.create_frame_save_callback(
                    batch_idx, save_config, post_process_fn=self.post_process_outputs
                )

            # Render
            output = self.render(
                gaussian_data=gaussian_data,
                cameras_output=cameras_output,
                timestamps_output=timestamps_output,
                render_mode=render_mode,
                sequential=True,
                save_callback=save_callback,
                return_outputs=False
            )

            # Finalize videos if needed
            if save_config and save_config.save_as_video:
                self.video_system.finalize_videos(batch_idx, save_config)

            logger.info(f"Completed rendering batch {batch_idx}")

        logger.info("Finished rendering from saved Gaussians")

    @torch.inference_mode()  # More efficient than no_grad for pure inference
    def run_inference(self,
                      images_input: torch.Tensor,
                      cameras_input: torch.Tensor,
                      timestamps_input: torch.Tensor,
                      cameras_output: torch.Tensor,
                      timestamps_output: torch.Tensor,
                      render_mode: str = 'RGB+ED',
                      monochrome: Optional[torch.Tensor] = None,
                      ratios: Optional[torch.Tensor] = None,
                      skip_render: bool = False,
                      sequential_render: bool = False,
                      save_config: Optional[SaveConfig] = None,
                      batch_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Run inference on a single batch.

        Args:
            images_input: Input images [B, N_in, C, H, W]
            cameras_input: Input camera parameters [B, N_in, 20]
            timestamps_input: Input timestamps [B, N_in]
            cameras_output: Output camera parameters [B, N_out, 20]
            timestamps_output: Output timestamps [B, N_out]
            render_mode: Rendering mode - one of ['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED'] (default: 'RGB+ED')
            monochrome: Monochrome flags for output views [B, N_out]
            ratios: Aspect ratios for output views [B, N_out]
            skip_render: Skip rendering (only compute 3D representation)
            sequential_render: If True, render views one by one to save memory (slower but memory-efficient)
            save_config: If provided with sequential_render=True, saves frames during rendering
            batch_idx: Batch index for saving (required if save_config is provided)

        Returns:
            Dictionary containing model outputs
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _load_model() first.")

        start_total = time.time()

        # Get dimensions for output height/width
        _, _, _, height_in, width_in = images_input.shape
        height_out = height_in  # For now, use same as input
        width_out = width_in

        # Use the new separated encode and render functions
        # Note: Camera preparation and device transfer are handled by encode() and render()
        start_model = time.time()

        # import pdb; pdb.set_trace()  # Removed debug line

        # Step 1: Encode input images to Gaussian parameters
        logger.info("Encoding input images...")
        gaussian_data = self.encode(
            images_input=images_input,
            cameras_input=cameras_input,
            timestamps_input=timestamps_input,
            height=height_out,
            width=width_out
        )
        self._record_time('Generating 4D Gaussians', gaussian_data.get('encoding_time', 0))

        # Step 2: Render output views (if not skipped)
        if skip_render:
            output = {
                "gs": gaussian_data['gaussian_parameters'],
                "encoding_time": gaussian_data.get('encoding_time', 0)
            }
            logger.info("Rendering skipped (skip_render=True)")
        else:
            if sequential_render:
                B, N = cameras_output.shape[:2]
                logger.info(f"Rendering {B*N} views sequentially (memory efficient)...")
            else:
                logger.info(f"Rendering views in batch...")

            # Create save callback if sequential rendering with save config
            save_callback = None
            # Don't return outputs if we're saving them (to save memory)
            return_outputs = True

            if sequential_render and save_config is not None and batch_idx is not None:
                save_callback = self.video_system.create_frame_save_callback(
                    batch_idx, save_config, post_process_fn=self.post_process_outputs
                )
                # If we're saving, don't keep outputs in memory
                return_outputs = False

            output = self.render(
                gaussian_data=gaussian_data,
                cameras_output=cameras_output,
                timestamps_output=timestamps_output,
                render_mode=render_mode,
                monochrome=monochrome,
                ratios=ratios,
                sequential=sequential_render,
                save_callback=save_callback,
                return_outputs=return_outputs
            )
            if output is not None:
                self._record_time('rendering', output.get('rendering_time', 0))
            else:
                # If output is None, we still want to record that rendering happened
                self._record_time('rendering', 0)

            # Log the breakdown
            encoding_time = gaussian_data.get('encoding_time', 0)
            rendering_time = output.get('rendering_time', 0) if output is not None else 0
            total_forward = encoding_time + rendering_time
            if total_forward > 0:
                logger.info(f"Total forward time: {total_forward:.3f}s (encode: {encoding_time/total_forward*100:.1f}%, render: {rendering_time/total_forward*100:.1f}%)")

        # Sync to get accurate timing
        if self.device != 'cpu':
            torch.cuda.synchronize()
        self._record_time('model_forward', time.time() - start_model)
        self._record_time('total_inference', time.time() - start_total)

        # Return output (might be None if we didn't store it to save memory)
        if output is None and not skip_render:
            # Return minimal info when outputs weren't stored
            return {
                "gs": gaussian_data['gaussian_parameters'],
                "encoding_time": gaussian_data.get('encoding_time', 0),
                "saved_to_disk": True
            }
        return output

    @torch.inference_mode()
    def post_process_outputs(self,
                             outputs: Dict[str, torch.Tensor],
                             normalize_depth: bool = True,
                             depth_min: float = 0.1,
                             depth_max: float = 100.0) -> Dict[str, torch.Tensor]:
        """
        Post-process model outputs.

        Args:
            outputs: Raw model outputs
            normalize_depth: Whether to normalize depth values
            depth_min: Minimum depth value for normalization
            depth_max: Maximum depth value for normalization

        Returns:
            Post-processed outputs
        """
        processed = {}

        for key, value in outputs.items():
            if key == 'gs':
                # Gaussian splatting parameters - keep as is
                processed[key] = value
            elif 'depth' in key and normalize_depth:
                # Normalize depth to [0, 1] for visualization
                normalized = (value - depth_min) / (depth_max - depth_min)  # [...] element-wise normalization
                normalized = torch.clamp(normalized, 0, 1)  # [...] clamp to [0, 1]
                processed[key] = normalized
            elif 'flow' in key:
                # Process optical flow if present
                processed[key] = value
            else:
                # Keep other outputs as is
                processed[key] = value

        return processed

    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        # Clean up video system
        self.video_system.cleanup()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Demo cleanup complete")

    def _combine_sequential_outputs(self, outputs_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Combine outputs from sequential rendering."""
        combined = {}

        for key in outputs_list[0].keys():
            if key == 'rendering_time':
                continue
            elif key == 'gs':
                # Gaussian parameters are the same for all frames
                combined[key] = outputs_list[0][key]
            else:
                # Stack tensors along view dimension
                views = [out[key] for out in outputs_list]
                if isinstance(views[0], torch.Tensor):
                    combined[key] = torch.cat(views, dim=1)
                else:
                    combined[key] = views[0]

        return combined

    # Old file handling methods removed - now handled by VideoSystem

    # Placeholder - old file handling methods removed

    @torch.inference_mode()
    def process_batch(self,
                      batch: Dict[str, Any],
                      batch_idx: int = 0,
                      save_outputs: bool = True,
                      save_config: Optional[SaveConfig] = None) -> Dict[str, torch.Tensor]:
        """
        Process a complete batch from dataloader.

        Args:
            batch: Batch dictionary from dataloader
            batch_idx: Batch index
            save_outputs: Whether to save outputs to disk
            save_config: Configuration for what to save (defaults to RGB only)

        Returns:
            Processed model outputs
        """
        if save_config is None:
            save_config = SaveConfig()  # Default: RGB only

        # Extract required inputs
        images_input = batch["rgb_input"]  # [B, N_in, C, H, W]

        # Handle monochrome cameras if present
        if 'mono_input' in batch:
            monos = batch["mono_input"]  # [B, N_mono, 1, H, W]
            B, _, C1, H, W = images_input.shape  # Get batch size and channel count
            _, N2, _, _, _ = monos.shape  # Get number of mono views
            images_input = torch.cat([images_input, monos.expand(B, N2, C1, H, W)], dim=1)  # [B, N_in, C, H, W] + [B, N_mono, C, H, W] -> [B, N_in+N_mono, C, H, W]

        # Extract camera and timestamp information
        cameras_input = batch["cameras_input"]  # [B, N_in, 20]
        cameras_output = batch["cameras_output"]  # [B, N_out, 20]
        timestamps_input = batch["rays_t_un_input"]  # [B, N_in]
        timestamps_output = batch["rays_t_un_output"]  # [B, N_out]

        # Optional parameters
        monochrome = batch.get("monochrome_output", None)
        ratios = batch.get("ratios_output", None)

        # Run inference with save config for frame-by-frame saving
        outputs = self.run_inference(
            images_input=images_input,
            cameras_input=cameras_input,
            timestamps_input=timestamps_input,
            cameras_output=cameras_output,
            timestamps_output=timestamps_output,
            render_mode='RGB+ED',  # Use valid render mode
            monochrome=monochrome,
            ratios=ratios,
            sequential_render=save_outputs,  # Use sequential rendering when saving
            save_config=save_config if save_outputs else None,
            batch_idx=batch_idx
        )

        # Post-process and save outputs if not already done during sequential rendering
        if outputs is not None and not save_outputs:  # If we have outputs and not saving
            start_post = time.time()
            outputs = self.post_process_outputs(outputs)
            self._record_time('post_process', time.time() - start_post)
        else:
            # Video writers are now handled by VideoSystem

            # For sequential rendering, outputs are already post-processed and saved
            # Just record the timing
            self._record_time('save_queue', 0)  # Saving happened during rendering

            # Finalize videos if needed
            if save_config and save_config.save_as_video:
                self.video_system.finalize_videos(batch_idx, save_config)

            # Clean up GPU memory after batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Print batch timing stats
        self.print_batch_timing(batch_idx)

        # Return outputs (might be None or minimal dict if saved to disk)
        return outputs if outputs is not None else {'saved_to_disk': True}

    def wait_for_all_saves(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all pending save operations to complete.
        Now delegates to VideoSystem.

        Args:
            timeout: Maximum time to wait in seconds
        """
        self.video_system.wait_for_all_saves(timeout=timeout)

    def print_batch_timing(self, batch_idx: int) -> None:
        """
        Print timing statistics for the current batch.

        Args:
            batch_idx: Current batch index
        """
        if not self.timing_stats:
            return

        # Get the latest timing for this batch
        latest_times = {}
        for category in self.timing_stats:
            if self.timing_stats[category]:
                latest_times[category] = self.timing_stats[category][-1]

        # Calculate percentages
        total_time = latest_times.get('total_inference', 0)
        if total_time > 0:
            # Get GPU stats if available
            gpu_util = "N/A"
            gpu_mem = "N/A"
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.memory_allocated(0) / (1024**3):.1f}GB"
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = f"{util.gpu}%"
                except:
                    pass

            # Format timing breakdown
            breakdown = []
            for category in ['model_forward', 'post_process', 'save_queue']:
                if category in latest_times:
                    pct = (latest_times[category] / total_time) * 100
                    breakdown.append(f"{category}: {latest_times[category]:.2f}s ({pct:.1f}%)")

            logger.info(
                f"Batch {batch_idx + 1} | "
                f"Total: {total_time:.2f}s | "
                f"{' | '.join(breakdown)} | "
                f"GPU: {gpu_util} | "
                f"Mem: {gpu_mem}"
            )

    def print_timing_summary(self) -> None:
        """
        Print a summary of timing statistics.
        """
        if not self.timing_stats:
            logger.info("No timing statistics available")
            return

        logger.info("\n" + "=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        total_batches = len(self.timing_stats.get('total_inference', []))
        if total_batches > 0:
            logger.info(f"Total batches processed: {total_batches}")

        for category in sorted(self.timing_stats.keys()):
            times = self.timing_stats[category]
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)

                logger.info(f"\n{category}:")
                logger.info(f"  Average: {avg_time:.3f}s")
                logger.info(f"  Min: {min_time:.3f}s")
                logger.info(f"  Max: {max_time:.3f}s")
                logger.info(f"  Total: {total_time:.3f}s")

        # Calculate throughput
        if 'total_inference' in self.timing_stats:
            total_time = sum(self.timing_stats['total_inference'])
            throughput = total_batches / total_time if total_time > 0 else 0
            logger.info(f"\nOverall throughput: {throughput:.2f} batches/second")

        logger.info("=" * 60 + "\n")
