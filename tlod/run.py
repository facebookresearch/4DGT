# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import time
from typing import Dict
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data.dataloader import default_collate

from tlod.easyvolcap.utils.console_utils import logger, blue, green
from tlod.misc import utils
from tlod.misc.io_helper import mkdirs
from tlod.data_loader.mvaria_dataset import AriaDataset
from torch.utils.data import DataLoader
from os.path import join, dirname

# Import the new demo class
from tlod.demo import FourDGTDemo
from tlod.download_model import download_4dgt_model


def create_dataset(cfg: DictConfig) -> AriaDataset:
    logger.info("Creating dataset...")

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

    # Handle image resolutions - can be single int or list/tuple
    if isinstance(cfg.input_image_res, (list, tuple)):
        input_image_res = tuple(cfg.input_image_res)
    else:
        input_image_res = (cfg.input_image_res, cfg.input_image_res)

    if isinstance(cfg.output_image_res, (list, tuple)):
        output_image_res = tuple(cfg.output_image_res)
    else:
        output_image_res = (cfg.output_image_res, cfg.output_image_res)

    # Create dataset with exact same parameters as main.py
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

    # Calculate normalized dataset length (number of 3D models/sequences)
    normalized_dataset_length = int(len(dataset) / dataset.batch_image_num + 0.5)
    logger.info(f"Dataset created with {len(dataset)} samples ({normalized_dataset_length} 3D models, batch_image_num={dataset.batch_image_num})")
    return dataset


def create_dataloader(dataset: AriaDataset,
                      cfg: DictConfig):
    """
    Setup data loader for inference.
    Matches main.py's dataloader setup exactly.
    """
    logger.info("Creating dataloader...")

    # Use vanilla DataLoader for single-GPU inference
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


def save_ground_truth(batch: Dict, cfg: DictConfig, batch_idx: int) -> None:
    """Save ground truth images for comparison."""
    # Save input images
    if "rgb_input" in batch:
        input_image_out = join(
            cfg.images_dir, f"{cfg.get('eval_prefix', '')}batch_{batch_idx:06d}_inputs.png"
        )
        mkdirs(dirname(input_image_out))
        inputs = batch["rgb_input"]
        utils.save_image(inputs, input_image_out)
        logger.info(f"Saved input images to {blue(input_image_out)}")

    # Save ground truth output images
    if "rgb_output" in batch:
        gt_image_out = join(
            cfg.images_dir, f"{cfg.get('eval_prefix', '')}batch_{batch_idx:06d}_gt.png"
        )
        mkdirs(dirname(gt_image_out))
        utils.save_image(batch["rgb_output"], gt_image_out)
        logger.info(f"Saved ground truth images to {blue(gt_image_out)}")


def run_inference(demo: FourDGTDemo,
                  dataloader: DataLoader,
                  cfg: DictConfig) -> None:
    """Run inference using the FourDGTDemo class."""

    logger.info(f"Starting inference...")
    logger.info(f"Output directory: {green(cfg.output_dir)}")

    dataloader_times = []
    batch_start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):

        # Measure data loading time
        dataloader_time = time.time() - batch_start_time
        if batch_idx > 0:  # Skip first batch timing as it includes startup
            dataloader_times.append(dataloader_time)

        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        # Handle list wrapper if present (dataset is collating itself)
        if isinstance(batch, list):
            batch = batch[0]  # remove the list wrapper since dataset is collating itself

        # Determine if we should save outputs
        should_save = (
            cfg.mode == "TEST" or  # Always save in TEST mode
            (batch_idx % cfg.get('saveimg_iter_freq', 100) == 0) or  # Save at regular intervals
            (batch_idx == len(dataloader) - 1)  # Save last batch
        )

        # Create save configuration based on user settings
        if should_save:
            from tlod.demo import SaveConfig

            # Create save config based on what the user wants
            save_config = SaveConfig(
                rgb=True,  # Always save RGB
                depth=cfg.get('save_depth', False),
                normal=cfg.get('save_normal', False),
                motion_mask=cfg.get('save_motion_mask', False),
                flow=cfg.get('save_flow', False),
                gaussians=cfg.get('save_gaussians', False),
                save_raw=cfg.get('save_raw', True),
                save_visualization=cfg.get('save_visualization', True),
                save_as_video=cfg.get('save_as_video', True),  # Default to video
                video_fps=cfg.get('video_fps', 10),
                video_codec=cfg.get('video_codec', 'h264'),
                video_quality=cfg.get('video_quality', 8),
                process_in_chunks=cfg.get('process_in_chunks', True),
                chunk_size=cfg.get('chunk_size', 32)
            )
        else:
            save_config = None

        # Process batch using the demo class
        outputs = demo.process_batch(
            batch=batch,
            batch_idx=batch_idx,
            save_outputs=should_save,
            save_config=save_config
        )

        # Also save ground truth images if needed
        if should_save and cfg.get('save_ground_truth', False):
            save_ground_truth(batch, cfg, batch_idx)

        # Reset timer for next dataloader iteration
        batch_start_time = time.time()

    # Print dataloader statistics
    if dataloader_times:
        avg_load_time = sum(dataloader_times) / len(dataloader_times)
        logger.info(f"\nDataloader Statistics:")
        logger.info(f"  Average load time: {avg_load_time:.3f}s")
        logger.info(f"  Min load time: {min(dataloader_times):.3f}s")
        logger.info(f"  Max load time: {max(dataloader_times):.3f}s")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set random seed
    utils.fix_random_seeds(cfg.seed)

    # Set up output directories
    cfg = setup_output_dirs(cfg)

    # Log configuration
    logger.info("=" * 60)
    logger.info("4DGT Inference Script with Hydra")
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

    # Create dataset
    dataset = create_dataset(cfg)

    # Create dataloader
    dataloader = create_dataloader(dataset, cfg)

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

    # Optional GPU monitoring
    gpu_monitor = None
    if cfg.get('enable_gpu_monitoring', False):
        from tlod.gpu_monitor import GPUMonitor
        gpu_monitor = GPUMonitor(interval=0.5)
        gpu_monitor.start()

    # Run inference
    run_inference(demo, dataloader, cfg)

    # Stop GPU monitoring and print summary
    if gpu_monitor:
        gpu_monitor.stop()
        gpu_monitor.print_summary()

    # Clean up resources (wait for async saves to complete)
    demo.cleanup()

    logger.info("Done!")


if __name__ == "__main__":
    main()
