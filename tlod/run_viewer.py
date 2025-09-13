# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import time
import threading
import torch
import hydra
import numpy as np
import viser
from queue import Queue
from jaxtyping import UInt8
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from os.path import join
from einops import rearrange
import matplotlib.pyplot as plt

from tlod import splatsviewer
from tlod.demo import FourDGTDemo
from tlod.video_system import SaveConfig
from tlod.easyvolcap.utils.console_utils import logger
from tlod.download_model import download_4dgt_model
from pathlib import Path


def create_dataloader(cfg: DictConfig) -> DataLoader:
    """Create dataloader based on configuration - matches run.py."""
    from tlod.data_loader.mvaria_dataset import AriaDataset
    from torch.utils.data.dataloader import default_collate

    logger.info("Creating dataset for viewer...")

    # Validate required parameters
    if cfg.data_path is None:
        raise ValueError("data_path must be specified. Use: data_path=/path/to/data")
    if cfg.seq_list is None:
        raise ValueError("seq_list must be specified. Use: seq_list=sequence_name")

    # Ensure seq_data_root is a list
    if isinstance(cfg.seq_data_root, str):
        seq_data_root = [cfg.seq_data_root]
    else:
        seq_data_root = list(cfg.seq_data_root)
    
    # Handle image resolutions
    if isinstance(cfg.input_image_res, (list, tuple)):
        input_image_res = tuple(cfg.input_image_res)
    else:
        input_image_res = (cfg.input_image_res, cfg.input_image_res)
        
    if isinstance(cfg.output_image_res, (list, tuple)):
        output_image_res = tuple(cfg.output_image_res)
    else:
        output_image_res = (cfg.output_image_res, cfg.output_image_res)
    
    # Create dataset
    dataset = AriaDataset(
        data_root=cfg.data_path,
        seq_list=cfg.seq_list,
        mode=cfg.mode,
        input_image_num=cfg.image_num_per_batch,
        input_image_res=input_image_res,
        output_image_num=cfg.output_image_num,
        output_image_res=output_image_res,
        seq_sample=cfg.seq_sample,
        seq_data_roots=seq_data_root,
        frame_sample=cfg.frame_sample,
        sample_interval=cfg.sample_interval,
        loaded_to_seconds=cfg.loaded_to_seconds,
        loaded_to_meters=cfg.loaded_to_meters,
        novel_time_sampling=cfg.novel_time_sampling,
        novel_time_frame_sample=cfg.novel_time_frame_sample,
        align_cameras=cfg.align_cameras,
        rotate_rgb=cfg.rotate_rgb,
        novel_view_interp_input=cfg.novel_view_interp_input,
        view_sample=cfg.view_sample,
        novel_view_timestamps=cfg.novel_view_timestamps,
        novel_view_spiral_window=cfg.novel_view_spiral_window,
        force_reload=cfg.force_reload,
    )
    
    # Calculate normalized dataset length
    normalized_dataset_length = int(len(dataset) / dataset.batch_image_num + 0.5)
    logger.info(f"Dataset: {len(dataset)} samples ({normalized_dataset_length} 3D models)")
    
    # Create dataloader - use vanilla DataLoader for single-GPU inference
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        collate_fn=default_collate
    )
    
    return dataloader


def setup_output_dirs(cfg: DictConfig) -> DictConfig:
    """Set up output directories based on configuration."""
    from omegaconf import OmegaConf
    from tlod.misc.io_helper import mkdirs

    # Temporarily disable struct mode to add new keys
    OmegaConf.set_struct(cfg, False)

    # Create output directory structure
    cfg.output_dir = join(cfg.exp_root, cfg.exp_name)
    cfg.checkpoint_dir = join(cfg.output_dir, cfg.checkpoints_dir)

    if cfg.eval_prefix:
        cfg.test_images_dir = join(cfg.test_images_dir, cfg.eval_prefix)
    cfg.images_dir = join(cfg.output_dir, "images")

    # Create directories
    mkdirs(cfg.output_dir)
    mkdirs(cfg.images_dir)

    # Re-enable struct mode for safety
    OmegaConf.set_struct(cfg, True)
    
    return cfg


def tensor2image(tensor, height, width, value_range=(-1, 1), colormap=None):
    """
    Convert a tensor to a numpy image array suitable for display.
    
    Args:
        tensor: Input tensor [B, C, H, W], [B, N, C, H, W], [C, H, W], or [H, W, C]
        height: Target height for output image
        width: Target width for output image
        value_range: Tuple of (min, max) values in the input tensor
            - (-1, 1): For normalized RGB images
            - (0, 1): For masks or already normalized data
            - (min, max): For depth maps or other modalities
        colormap: Optional colormap to apply (e.g., 'viridis' for depth)
    
    Returns:
        numpy array [H, W, 3] in uint8 format ready for display
    """
    # Handle tensor dimensions
    if tensor.dim() == 5:  # [B, N, C, H, W]
        tensor = tensor[0, 0]  # Get first batch, first view
    elif tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # Get first batch
    # tensor is now [C, H, W] or [H, W, C]
    
    # Normalize to [0, 1]
    if value_range != (0, 1):
        min_val, max_val = value_range
        tensor = (tensor - min_val) / (max_val - min_val)
    tensor = torch.clamp(tensor, 0, 1)
    
    # Handle single channel tensors (depth, masks, etc.)
    if tensor.dim() == 2:  # [H, W]
        tensor = tensor.unsqueeze(0)  # [1, H, W]
    
    # Ensure tensor is in CHW format for interpolation
    if tensor.shape[-1] in [1, 3] and tensor.shape[0] not in [1, 3]:  # Likely HWC
        tensor = rearrange(tensor, 'h w c -> c h w')
    
    # Resize on GPU if needed
    if tensor.shape[1] != height or tensor.shape[2] != width:
        tensor = tensor.unsqueeze(0)  # Add batch dimension for interpolate
        tensor = torch.nn.functional.interpolate(
            tensor, 
            size=(height, width), 
            mode='bilinear', 
            align_corners=False
        )
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Apply colormap if specified (for single channel)
    if colormap and tensor.shape[0] == 1:
        # Convert to numpy first for colormap application
        tensor_np = tensor[0].cpu().numpy()  # [H, W], values in [0, 1]
        
        # Get the colormap from matplotlib
        cmap = plt.get_cmap(colormap)
        
        # Apply colormap (returns RGBA in [0, 1])
        colored = cmap(tensor_np)  # [H, W, 4]
        
        # Extract RGB channels and convert to [3, H, W]
        tensor_np = colored[:, :, :3].transpose(2, 0, 1)  # [3, H, W]
        tensor = torch.from_numpy(tensor_np.astype(np.float32))
    elif tensor.shape[0] == 1:
        # Replicate single channel to RGB
        tensor = tensor.repeat(3, 1, 1)  # [3, H, W]
    
    # Convert to numpy and uint8
    image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Transpose from CHW to HWC if needed
    if image.shape[0] in [1, 3]:  # Channels first
        image = image.transpose(1, 2, 0)
    
    # Ensure 3 channels for display
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    return image


class FourDGTViewer:
    """Interactive viewer for 4DGT with async Gaussian generation."""
    
    def __init__(self, 
                 demo: FourDGTDemo,
                 dataloader: DataLoader,
                 save_config: SaveConfig,
                 port: int = 8080,
                 max_render_size: int = 512):
        """
        Initialize the 4DGT viewer.
        
        Args:
            demo: FourDGTDemo instance with loaded model
            dataloader: DataLoader for input data
            save_config: Configuration for saving Gaussians
            port: Port for the viewer server
            max_render_size: Maximum render width/height (default 512)
        """
        self.demo = demo
        self.dataloader = dataloader
        self.save_config = save_config
        self.port = port
        self.max_render_size = max_render_size
        
        # Gaussian storage
        self.gaussian_queue = Queue(maxsize=1)
        self.gaussian_scenes = []  # List to track all loaded scenes
        self.current_scene_idx = 0  # Index of current scene in gaussian_scenes
        self.current_gaussians = None
        self.current_batch_data = None  # Store full batch data including cameras
        
        # Camera parameters for rendering
        self.center_camera = None  # Center camera from output cameras
        self.center_timestamp = None  # Center timestamp from output
        
        # Threading control
        self.shutdown_event = threading.Event()
        self.generation_thread = None
        self.generating_status = True  # Control whether to continue generating
        self.generation_paused_event = threading.Event()  # Event to pause/resume generation
        self.max_scenes_before_pause = 5  # Maximum scenes to generate before pausing
        
        # Viewer components
        self.server = None
        self.viewer = None
        
        # Statistics tracking
        self.generation_times = []  # Track generation time for each scene
        self.last_render_time = 0  # Track last render time
        self.total_scenes_to_generate = 0  # Total number of scenes from dataloader
        
    def compute_center_camera_from_input(self, cameras_input):
        """
        Compute the center camera from input cameras.
        
        Args:
            cameras_input: Input camera parameters [B, N, 20]
        
        Returns:
            Center camera parameters [B, 1, 20]
        """
        _, N, _ = cameras_input.shape
        center_idx = N // 2
        return cameras_input[:, center_idx:center_idx+1]  # [B, 1, 20]
    
    def camera_state_to_camera_params(self, camera_state, base_camera=None):
        """
        Convert camera_state from viewer to camera parameters format.
        
        Args:
            camera_state: CameraState object with fov, aspect, and c2w
            base_camera: Base camera parameters [B, 1, 20] to use as reference
        
        Returns:
            Camera parameters [B, 1, 20] compatible with the model
        """
        
        # If we don't have a base camera, use the center camera
        if base_camera is None:
            base_camera = self.center_camera
        
        if base_camera is None:
            logger.warning("No base camera available for conversion")
            return None
            
        # Clone the base camera to modify
        camera_params = base_camera.clone()  # [B, 1, 20]
        
        # Convert c2w matrix to camera parameters format
        c2w = camera_state.c2w  # [4, 4] in OpenCV convention from viser
        
        # The model expects OpenGL convention, so flip Y and Z axes
        c2w_gl = c2w.copy()
        c2w_gl[:3, 1:3] = -c2w_gl[:3, 1:3]
        
        # Flatten the 4x4 matrix to [16]
        c2w_flat = torch.from_numpy(c2w_gl).float().flatten()
        
        # Update the extrinsics part (first 16 values)
        camera_params[0, 0, :16] = c2w_flat
        
        # Keep FOV from base camera instead of recalculating
        # The base camera already has the correct fov_x and fov_y
        # Only update if we want to use viewer's FOV (optional)
        # camera_params[0, 0, 16] = camera_state.fov  # fov_x
        # camera_params[0, 0, 17] = camera_state.fov / camera_state.aspect  # fov_y
        
        # Keep principal points from base camera (indices 18, 19)
        
        return camera_params
    
    def get_gaussian_with_cameras(self):
        """
        Get gaussian data with associated cameras from the queue.
        
        Returns:
            Dictionary with gaussian_data, cameras_input, timestamps_input, and batch_idx
            or None if queue is empty
        """
        if not self.gaussian_queue.empty():
            return self.gaussian_queue.get()
        return None
    
    def switch_to_gaussian_batch(self, queue_data):
        """
        Switch to a different gaussian batch with its associated cameras.
        
        Args:
            queue_data: Dictionary from the queue containing gaussian_data, cameras_input, etc.
        """
        if queue_data is None:
            return
            
        # Update current gaussians and batch info
        self.current_gaussians = queue_data['gaussian_data']
        self.current_batch_data = queue_data.get('batch_data')
        
        # Update cameras and timestamps
        if 'cameras_input' in queue_data:
            # Compute and set center camera for this batch
            cameras_input = queue_data['cameras_input']
            self.center_camera = self.compute_center_camera_from_input(cameras_input)

            self.set_viewer_camera_from_params(self.center_camera)
            
            # Update frame slider max value
            if self.viewer and hasattr(self.viewer, 'frame_slider'):
                num_cameras = self.get_input_camera_count()
                if num_cameras > 0:
                    self.viewer.frame_slider.max = num_cameras - 1
                    # Reset to center camera
                    self.viewer.frame_slider.value = num_cameras // 2
            
        if 'timestamps_input' in queue_data:
            # Update timestamp range
            timestamps = queue_data['timestamps_input']
            self._start_time = timestamps.min(dim=1)[0]  # [B, N] -> [B]
            self._end_time = timestamps.max(dim=1)[0]  # [B, N] -> [B]
            
            # Reset time slider to middle
            if self.viewer and hasattr(self.viewer, 'render_tab_state'):
                self.viewer.render_tab_state.time_ratio = 0.5
                        
        logger.info(f"[Viewer] Switched to gaussian batch {queue_data['batch_idx']}")
    
    def switch_to_next_scene(self):
        """Switch to the next Gaussian scene."""
        if not self.gaussian_scenes:
            logger.warning("[Viewer] No scenes available to switch")
            return
            
            
        # Check if we're at the last scene and generation is paused
        if self.current_scene_idx == len(self.gaussian_scenes) - 1 and not self.generating_status:
            # Resume generation for one more batch
            logger.info("[Viewer] At last scene, resuming generation for next batch")
            self.generating_status = True
            self.max_scenes_before_pause = len(self.gaussian_scenes) + 1  # Allow one more scene
            
        # Move to next scene
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.gaussian_scenes)
        scene_data = self.gaussian_scenes[self.current_scene_idx]
        logger.info(f"[Viewer] Switching to next scene {self.current_scene_idx + 1}/{len(self.gaussian_scenes)}")
        self.switch_to_gaussian_batch(scene_data)
        
    def switch_to_previous_scene(self):
        """Switch to the previous Gaussian scene."""
        if not self.gaussian_scenes:
            logger.warning("[Viewer] No scenes available to switch")
            return
            
            
        # Move to previous scene
        self.current_scene_idx = (self.current_scene_idx - 1) % len(self.gaussian_scenes)
        scene_data = self.gaussian_scenes[self.current_scene_idx]
        logger.info(f"[Viewer] Switching to previous scene {self.current_scene_idx + 1}/{len(self.gaussian_scenes)}")
        self.switch_to_gaussian_batch(scene_data)
    
    def update_display(self):
        """Update display stats in the viewer."""
        # Calculate generation statistics
        num_generated = len(self.gaussian_scenes)
        avg_gen_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        
        # Get current scene info
        current_scene_display = f"{self.current_scene_idx + 1}/{num_generated}" if num_generated > 0 else "0/0"
        
        # Get generation state
        gen_state = "Active" if self.generating_status else "Paused"
        
        # Format the status message
        update_status = f"""<sub>
            **Generation**: {num_generated}/{self.total_scenes_to_generate} scenes | State: {gen_state} | Avg: {avg_gen_time:.2f}s/scene \\
            **Rendering**: Scene {current_scene_display} | Last: {self.last_render_time:.3f}s
        </sub>"""

        if self.server and self.viewer and hasattr(self.viewer, '_stats_text'):
            with self.server.atomic():
                self.viewer._stats_text.content = update_status 

    def set_camera_to_input_index(self, camera_index):
        """
        Set the viewer camera to a specific index from the current input cameras.
        
        Args:
            camera_index: Index of the camera in the input camera array
        """
        if not self.gaussian_scenes or self.current_scene_idx < 0:
            logger.warning("[Viewer] No scene available to get camera from")
            return
            
        # Get current scene data
        current_scene = self.gaussian_scenes[self.current_scene_idx]
        if 'cameras_input' not in current_scene:
            logger.warning("[Viewer] No input cameras in current scene")
            return
            
        cameras_input = current_scene['cameras_input']  # [B, N, 20]
        _, N, _ = cameras_input.shape
        
        # Clamp camera index to valid range
        camera_index = max(0, min(camera_index, N - 1))
        
        # Extract the specific camera
        camera = cameras_input[:, camera_index:camera_index+1]  # [B, 1, 20]

        # Set viewer camera to this position (but don't update self.center_camera)
        self.set_viewer_camera_from_params(camera)
        logger.info(f"[Viewer] Set camera to input camera {camera_index + 1}/{N}")
    
    def recenter_viewer(self):
        """
        Recenter the viewer camera to the original center camera of the current scene.
        """
        if not self.gaussian_scenes or self.current_scene_idx < 0:
            logger.warning("[Viewer] No scene available to recenter")
            return
            
        # Get current scene data
        current_scene = self.gaussian_scenes[self.current_scene_idx]
        if 'cameras_input' not in current_scene:
            logger.warning("[Viewer] No input cameras in current scene")
            return
            
        # Recompute the center camera from the scene's cameras
        cameras_input = current_scene['cameras_input']
        self.center_camera = self.compute_center_camera_from_input(cameras_input)
                
        # Set viewer camera to center position
        self.set_viewer_camera_from_params(self.center_camera)
        
        # Reset frame slider to center if it exists
        if self.viewer and hasattr(self.viewer, 'frame_slider'):
            num_cameras = self.get_input_camera_count()
            if num_cameras > 0:
                self.viewer.frame_slider.value = num_cameras // 2
        
        logger.info("[Viewer] Recentered camera to scene center")
    
    def get_input_camera_count(self):
        """Get the number of input cameras in the current scene."""
        if not self.gaussian_scenes or self.current_scene_idx < 0:
            return 0
            
        current_scene = self.gaussian_scenes[self.current_scene_idx]
        if 'cameras_input' not in current_scene:
            return 0
            
        cameras_input = current_scene['cameras_input']
        return cameras_input.shape[1] if len(cameras_input.shape) > 1 else 1
    
    def set_viewer_camera_from_params(self, camera_params):
        """
        Set the viewer camera to match the given camera parameters.
        
        Args:
            camera_params: Camera parameters [B, N, 20] or [1, 20]
        """
        if self.server is None:
            return
            
        import viser.transforms as vt
        
        # Get the first camera if multiple
        if camera_params.dim() == 3:
            camera_params = camera_params[0, 0]  # [20]
        elif camera_params.dim() == 2:
            camera_params = camera_params[0]  # [20]
        
        # Extract the 4x4 transformation matrix
        c2w_gl = camera_params[:16].view(4, 4)  # [4, 4]
        
        # Convert from OpenGL to OpenCV convention
        c2w = c2w_gl.clone()
        c2w[:3, 1:3] = -c2w[:3, 1:3]
        
        # Extract rotation and translation
        R = c2w[:3, :3].cpu().numpy()
        t = c2w[:3, 3].cpu().numpy()
        
        # Convert rotation matrix to quaternion
        so3 = vt.SO3.from_matrix(R)
        wxyz = so3.wxyz
        
        # Extract FOV
        fov_x = camera_params[16].item()
        
        # Set camera for all connected clients
        clients = self.server.get_clients()
        for client in clients.values():            
            client.camera.position = t
            client.camera.wxyz = wxyz
            client.camera.fov = fov_x            
            logger.info(f"Set viewer camera to input camera center")
    
    def gaussian_generator(self):
        """Generate Gaussians asynchronously and save them."""
        try:
            # Get total number of scenes
            self.total_scenes_to_generate = len(self.dataloader)
            
            for batch_idx, batch in enumerate(self.dataloader):
                # Check for shutdown
                if self.shutdown_event.is_set():
                    logger.info("[Generator] Shutdown requested, stopping generation")
                    break
                
                start_time = time.time()
                
                # Handle list wrapper if present
                if isinstance(batch, list):
                    batch = batch[0]
                
                # Extract inputs
                images_input = batch["rgb_input"]       # (B, N, C, H, W)
                cameras_input = batch["cameras_input"]  # (B, N, 20)
                timestamps_input = batch["rays_t_un_input"]     # (B, N)

                self._start_time = timestamps_input.min(dim=1)[0]  # [B, N] -> [B]
                self._end_time = timestamps_input.max(dim=1)[0]  # [B, N] -> [B]
                
                # Set center camera only once at the beginning
                    
                # Handle monochrome cameras if present
                if 'mono_input' in batch:
                    monos = batch["mono_input"]
                    B, _, C, H, W = images_input.shape
                    _, N, _, _, _ = monos.shape
                    images_input = torch.cat([images_input, monos.expand(B, N, C, H, W)], dim=1)

                # Generate Gaussians
                logger.info(f"[Generator] Processing batch {batch_idx + 1}/{len(self.dataloader)}")
                gaussian_data = self.demo.encode(
                    images_input=images_input,
                    cameras_input=cameras_input.clone(),
                    timestamps_input=timestamps_input
                )

                if self.center_camera is None:
                    self.center_camera = self.compute_center_camera_from_input(cameras_input)  # [B, 1, 20]
                    self.set_viewer_camera_from_params(self.center_camera)
                    logger.info("[Generator] Set initial center camera from first batch")


                # Save Gaussians asynchronously
                if self.save_config.gaussians and not self.shutdown_event.is_set():
                    gs_path = self.demo.video_system.save_gaussians(
                        gaussian_data['gaussian_parameters'],
                        batch_idx,
                        self.save_config
                    )
                    gaussian_data['saved_path'] = gs_path
                
                # Store for viewer with camera data
                self.current_gaussians = gaussian_data
                self.current_batch_data = batch
                
                # Store cameras_input with gaussian_data for later use
                gaussian_data['cameras_input'] = cameras_input.clone()  # Clone to avoid contamination
                
                # Store output cameras and timestamps for future use
                # if "cameras_output" in batch and "rays_t_un_output" in batch:
                #     cameras_output = batch["cameras_output"]       # (B, N_out, 20)
                #     timestamps_output = batch["rays_t_un_output"]  # (B, N_out)
                
                # Create scene data dictionary
                scene_data = {
                    'batch_idx': batch_idx,
                    'gaussian_data': gaussian_data,
                    'cameras_input': cameras_input.clone(),  # Clone to avoid contamination
                    'timestamps_input': timestamps_input.clone(),  # Clone to avoid contamination
                    # 'batch_data': batch  # Note: batch is a new dict each iteration from dataloader
                }
                
                # Add to scenes list with maximum limit
                max_scenes = 10  # Maximum number of scenes to keep in memory
                if len(self.gaussian_scenes) >= max_scenes:
                    # Drop the oldest scene if it has been serialized
                    old_scene = self.gaussian_scenes.pop(0)
                    if 'saved_path' in old_scene.get('gaussian_data', {}):
                        logger.info(f"[Generator] Dropping serialized scene {old_scene['batch_idx']}")
                    else:
                        # If not serialized, keep it by adding it back
                        self.gaussian_scenes.insert(0, old_scene)
                        self.gaussian_scenes.pop()  # Remove the last one instead
                
                # Add new scene to list
                self.gaussian_scenes.append(scene_data)
                
                # Queue for potential future use with cameras (backward compatibility)
                if not self.gaussian_queue.full():
                    self.gaussian_queue.put(scene_data)
                
                gen_time = time.time() - start_time
                self.generation_times.append(gen_time)
                logger.info(f"[Generator] Generated batch {batch_idx} in {gen_time:.2f}s")
                
                # Check if we should pause generation
                if len(self.gaussian_scenes) >= self.max_scenes_before_pause:
                    self.generating_status = False
                    logger.info(f"[Generator] Reached {self.max_scenes_before_pause} scenes, pausing generation")

                # Wait if generation is paused
                if not self.generating_status:
                    logger.info("[Generator] Generation paused, waiting for resume signal...")
                while not self.generating_status and not self.shutdown_event.is_set():
                    time.sleep(0.5)  # Check every half second
                
        except KeyboardInterrupt:
            logger.info("[Generator] Interrupted by user")
            self.shutdown_event.set()
        except Exception as e:
            logger.error(f"[Generator] Error: {e}")
            self.shutdown_event.set()
        finally:
            logger.info("[Generator] Finished generating Gaussians")
    
    def render_fn(self, 
                  camera_state: splatsviewer.CameraState,  # Will be used for actual rendering
                  render_tab_state: splatsviewer.RenderTabState) -> UInt8[np.ndarray, "H W 3"]:
        """
        Render function for the viewer.
        
        This renders the 4DGT model at the center timestamp using the interactive camera.
        """
        render_start_time = time.time()
        self.update_display()

        # Get requested render size
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        # Apply maximum size constraint
        if width > self.max_render_size:
            scale = self.max_render_size / width
            width = self.max_render_size
            height = int(height * scale)
        if height > self.max_render_size:
            scale = self.max_render_size / height
            height = self.max_render_size
            width = int(width * scale)
        
        # Check if we have Gaussians and camera data
        if self.current_gaussians is None or self.center_camera is None:
            # Return placeholder image while waiting
            img = np.ones((height, width, 3), dtype=np.uint8) * 50
            if self.current_gaussians is None:
                img[:30, :] = [255, 0, 0]  # Red bar - no Gaussians
            else:
                img[:30, :] = [255, 255, 0]  # Yellow bar - no cameras
            self.last_render_time = time.time() - render_start_time
            return img
        
        try:
            # Convert the viewer's camera_state to model camera parameters
            cameras_to_use = self.camera_state_to_camera_params(camera_state, base_camera=self.center_camera)
            
            if cameras_to_use is None:
                # Fallback to center camera if conversion fails
                cameras_to_use = self.center_camera
                
            timestamps_to_use = self._start_time + render_tab_state.time_ratio * (self._end_time - self._start_time)

            # Get additional batch data if available
            monochrome = None
            ratios = None
            if self.current_batch_data is not None:
                monochrome = self.current_batch_data.get("monochrome_output")
                ratios = self.current_batch_data.get("ratios_output")

            if self.viewer.render_modality == "rgb": 
                render_mode = "RGB"
            else: 
                render_mode = "RGB+ED"

            # Render using the demo's render function
            with torch.no_grad():
                output = self.demo.render(
                    gaussian_data=self.current_gaussians,
                    cameras_output=cameras_to_use,
                    timestamps_output=timestamps_to_use,
                    render_mode=render_mode, 
                    monochrome=monochrome,
                    ratios=ratios,
                    sequential=False,  # Single frame rendering
                    return_outputs=True  # Need the output for display
                )

            self.update_display()

            # Extract and convert the appropriate modality
            if self.viewer.render_modality == "rgb":
                image = tensor2image(output['rgb'], height, width, value_range=(-1, 1))
                
            elif self.viewer.render_modality == "depth" and 'depth' in output:
                # Depth visualization with appropriate range
                depth_tensor = output['depth']
                # Handle multi-channel depth (e.g., [B, 3, H, W])
                if depth_tensor.shape[1] == 3:
                    depth_tensor = depth_tensor[:, 0:1]  # Take first channel

                depth_min = max(0.1, depth_tensor.min())
                depth_max = min(100, depth_tensor.max()+1)
                image = tensor2image(depth_tensor, height, width, value_range=(depth_min, depth_max), colormap='viridis')

            elif self.viewer.render_modality == "normal" and 'normal' in output:
                # Normal map visualization (assuming normals are in [-1, 1])
                image = tensor2image(output['normal'], height, width, value_range=(-1, 1))

            elif self.viewer.render_modality == "motion" and 'mask' in output:
                # Binary motion mask visualization
                image = tensor2image(output['mask'], height, width, value_range=(0, 1), colormap='gray')

            elif self.viewer.render_modality == "flow" and 'flow_vis' in output:
                # Flow map visualization
                image = tensor2image(output['flow_vis'], height, width, value_range=(0, 1))

            else:
                # Fallback to RGB with warning
                if self.viewer.render_modality != "rgb":
                    logger.warning(f"[Viewer] Modality '{self.viewer.render_modality}' not available. Falling back to RGB.")
                image = tensor2image(output['rgb'], height, width, value_range=(-1, 1))

            self.last_render_time = time.time() - render_start_time
            return image


        except Exception as e:
            logger.error(f"[Viewer] Rendering error: {e}")
            # Return error image
            img = np.ones((height, width, 3), dtype=np.uint8) * 75
            img[:30, :] = [255, 0, 0]  # Red bar - error
            self.last_render_time = time.time() - render_start_time
            return img
    
    def run(self):
        """Run the interactive viewer with async Gaussian generation."""
        logger.info("=" * 60)
        logger.info("4DGT Interactive Viewer")
        logger.info("=" * 60)
        
        # Start Gaussian generation thread
        self.generation_thread = threading.Thread(
            target=self.gaussian_generator, 
            name="GaussianGenerator"
        )
        self.generation_thread.start()
        
        # Initialize the viser server and viewer
        logger.info(f"Starting viewer server on port {self.port}")
        self.server = viser.ViserServer(port=self.port, verbose=False)
        self.viewer = splatsviewer.Viewer(
            server=self.server,
            render_fn=self.render_fn,
            mode="rendering"  # Use rendering mode for interactive viewer
        )
        # Store reference to this viewer in the splatsviewer for button callbacks
        self.viewer.fourdgt_viewer = self
        
        # Make world axes visible
        self.server.scene.world_axes.visible = True
        
        logger.info(f"Viewer ready at http://localhost:{self.port}")
        logger.info("Press Ctrl+C to shutdown")
        
        try:
            # Keep the viewer running
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
                
                # Update viewer state if needed
                if self.current_gaussians is not None:
                    # We can update viewer state here if needed
                    pass
                    
        except KeyboardInterrupt:
            logger.info("\n[Viewer] Shutdown requested")
            self.shutdown_event.set()
        
        # Cleanup
        logger.info("Shutting down...")
        self.shutdown_event.set()
        
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
            if self.generation_thread.is_alive():
                logger.warning("Generation thread did not shutdown cleanly")
        
        # Cleanup demo resources
        self.demo.cleanup()
        
        logger.info("Viewer shutdown complete")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the 4DGT viewer."""
    from tlod.misc import utils
    
    # Set random seed
    utils.fix_random_seeds(cfg.seed)
    
    # Set up output directories
    cfg = setup_output_dirs(cfg)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("4DGT Viewer with Async Gaussian Generation")
    logger.info("=" * 60)
    logger.info(f"Mode: {cfg.mode}")
    logger.info(f"Config: {cfg.config}")
    logger.info(f"Output dir: {cfg.output_dir}")
    if cfg.checkpoint:
        logger.info(f"Checkpoint: {cfg.checkpoint}")
    logger.info("=" * 60)
    
    # Download checkpoint if it doesn't exist
    if cfg.checkpoint and not Path(cfg.checkpoint).exists():
        logger.info(f"Checkpoint not found at {cfg.checkpoint}, downloading...")
        cfg.checkpoint = str(download_4dgt_model(
            output_dir=Path(cfg.checkpoint).parent,
            filename=Path(cfg.checkpoint).name
        ))

    # Create dataset and dataloader
    dataloader = create_dataloader(cfg)

    # Initialize the demo class
    device = cfg.get('device', 'cuda')
    demo = FourDGTDemo(
        config_path=cfg.config,
        checkpoint_path=cfg.checkpoint,
        device=device,
        output_dir=cfg.output_dir,
        fp16=cfg.get('fp16', False),
        fp32=cfg.get('fp32', False)
    )
    
    # Create save configuration
    save_config = SaveConfig(
        gaussians=True,  # Always save Gaussians for viewer
        save_gaussians_async=True,
        gaussian_format='npz',
        use_async_save=True,
        max_workers=2
    )
    
    # Get viewer port and settings
    port = cfg.get('viewer_port', 8080)
    if hasattr(cfg, 'viewer_port'):
        port = cfg.viewer_port
    
    # Get max render size from config or use default
    max_render_size = cfg.get('max_render_size', 512)
    if hasattr(cfg, 'max_render_size'):
        max_render_size = cfg.max_render_size
    
    # Create and run viewer
    viewer = FourDGTViewer(
        demo=demo,
        dataloader=dataloader,
        save_config=save_config,
        port=port,
        max_render_size=max_render_size
    )
    
    viewer.run()


if __name__ == "__main__":
    main()