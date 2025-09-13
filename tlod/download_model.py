# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from tlod.easyvolcap.utils.console_utils import logger


def download_4dgt_model(
    repo_id="projectaria/4DGT",
    filename="4dgt_full.pth",
    output_dir="checkpoints",
    force_download=False
):
    """
    Download 4DGT model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: Name of the model file to download (4dgt_full.pth or 4dgt_1st_stage.pth)
        output_dir: Local directory to save the model
        force_download: Force re-download even if file exists locally

    Returns:
        Path to the downloaded model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {filename} from {repo_id}...")

    # Download using HF Hub API
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=output_dir / ".cache",
        force_download=force_download
    )

    # Create symlink or copy to expected location
    final_path = output_dir / filename
    if final_path.exists() and not force_download:
        logger.info(f"Model already exists at: {final_path}")
    else:
        # Create a symlink to the cached file
        if final_path.is_symlink() or final_path.exists():
            final_path.unlink()
        # Use absolute path for symlink to avoid relative path issues
        final_path.symlink_to(Path(downloaded_path).absolute())
        logger.info(f"Model linked to: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Download 4DGT pretrained models from Hugging Face")
    parser.add_argument(
        "--repo-id",
        default="projectaria/4DGT",
        help="Hugging Face repository ID (default: projectaria/4DGT)"
    )
    parser.add_argument(
        "--filename",
        default="4dgt_full.pth",
        choices=["4dgt_full.pth", "4dgt_1st_stage.pth"],
        help="Model filename to download (default: 4dgt_full.pth, options: 4dgt_full.pth, 4dgt_1st_stage.pth)"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Output directory for the model (default: checkpoints)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    try:
        path = download_4dgt_model(
            repo_id=args.repo_id,
            filename=args.filename,
            output_dir=args.output_dir,
            force_download=args.force
        )
        logger.info(f"✓ Model ready at: {path}")
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        logger.error("Make sure you have huggingface-hub installed:")
        logger.error("  pip install huggingface-hub")
        exit(1)


if __name__ == "__main__":
    main()