<div align="center">

<h3> 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos </h2>
<a href="https://arxiv.org/abs/2506.08015">
  <img src="https://img.shields.io/badge/2506.08015-arXiv-red" alt="arXiv">
</a>
<a href="https://4dgt.github.io/">
  <img src="https://img.shields.io/badge/4DGT-project_page-blue" alt="Project Page">
</a>

<br/>

<a href="https://zhenx.me" target="_blank">Zhen Xu<sup>1,2,*</sup></a>
<a href="https://sites.google.com/view/zhengqinli" target="_blank">Zhengqin Li<sup>1</sup></a>
<a href="https://flycooler.com/" target="_blank">Zhao Dong<sup>1</sup></a>
<a href="https://xzhou.me" target="_blank">Xiaowei Zhou<sup>2</sup></a>
<a href="https://rapiderobot.bitbucket.io/" target="_blank">Richard Newcombe<sup>1</sup></a>
<a href="https://lvzhaoyang.github.io/" target="_blank">Zhaoyang Lv<sup>1</sup></a>
</a>
<p>
    <sup>1</sup>Reality Labs Research, Meta&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Zhejiang University
    <br />
    <span style="color: #5a6268; font-size: 0.9em">
        <sup>*</sup>Work done during internship at Meta.&nbsp;&nbsp;&nbsp;&nbsp;
    </span>
</p>

[![4DGT](assets/vid/teaser.gif)](https://4dgt.github.io/)

</div>

We propose 4DGT, a 4D Gaussian-based Transformer model for dynamic scene reconstruction, trained entirely on real-world monocular posed videos. Using 4D Gaussian as an inductive bias, 4DGT unifies static and dynamic components, enabling the modeling of complex, time-varying environments with varying object lifespans. We proposed a novel density control strategy in training which enables our 4DGT to handle longer space-time input and remain efficient rendering at runtime. Our model processes 64 consecutive posed frames in a rolling-window fashion, predicting consistent 4D Gaussians in the scene. Unlike optimization-based methods, 4DGT performs purely feed-forward inference, reducing reconstruction time from hours to seconds and scaling effectively to long video sequences. Trained only on large-scale monocular posed video datasets, 4DGT can outperform prior Gaussian-based networks significantly in real-world videos and achieve on-par accuracy with optimization-based methods on cross-domain videos.

## Installation

Use the automated installation script:

```bash
bash scripts/install.sh
```

This script will interactively guide you through setting up the conda environment and installing all dependencies including PyTorch, flash-attention, and apex.

For detailed installation instructions and troubleshooting, see [docs/install.md](docs/install.md).

## Pretrained Model

Download the pretrained model from [Hugging Face](https://huggingface.co/zhaoyang-lv-meta/4DGT/tree/main):
```bash
# Create the checkpoint directory
mkdir -p logs/4DGT/checkpoints

# Download the checkpoint
wget https://huggingface.co/zhaoyang-lv-meta/4DGT/resolve/main/last.pth -O logs/4DGT/checkpoints/last.pth
```
The trained model should be available at `logs/4DGT/checkpoints/last.pth`.


### Aria Datasets

We provide two examples of converting a typical Aria recording in `.vrs` to the format recognized by *4DGT*. For details of the data format being processed, check [docs/data.md](docs/data.md).


#### Run on an Aria sequence from Aria Explorer

We use the sequence from [the Aria explorer](https://explorer.projectaria.com/aea/loc3_script3_seq1_rec1?st=%220%22) from Aria Everyday Activity as an example. This will apply to generally any sequences to be found downloaded from Aria Explorer. 

```bash
mkdir -p data/aea 
cd data/aea

# Put the download url json file into data/aea folder
# the download url file will be different if you choose a different sequence. 
aria_dataset_downloader -c loc3_script3_seq1_rec1_download_urls.json -o . -l all
```

Run the following from the sequence 
```bash
# Process the sequence "loc3_script3_seq1_rec1"
DATA_INPUT_DIR="data/aea/loc3_script3_seq1_rec1" \
DATA_PROCESSED_DIR="data/aea/loc3_script3_seq1_rec1" \
VRS_FILE="recording.vrs" \
bash shells/data/run_vrs_preprocessing.sh
```

#### Run on Aria Digital Twin sequence

```bash
# Create a directory for datasets
mkdir -p data/adt-raw
cd data/adt-raw

# Put the download link file "ADT_download_urls.json" in the current directory
# Get the file from: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download

# Download the sample data
aria_dataset_downloader --cdn_file ADT_download_urls.json --output_folder . --data_types 0 1 2 3 4 5 6 7 8 9 --sequence_names Apartment_release_multiuser_cook_seq141_M1292 Apartment_release_multiskeleton_party_seq114_M1292 Apartment_release_meal_skeleton_seq135_M1292 Apartment_release_work_skeleton_seq137_M1292
```

```bash
# Process the synthetic sequence "Apartment_release_multiuser_cook_seq141_M1292"
DATA_INPUT_DIR="data/adt-raw/Apartment_release_multiuser_cook_seq141_M1292" DATA_PROCESSED_DIR="data/adt/Apartment_release_multiuser_cook_seq141_M1292" VRS_FILE="synthetic_video.vrs" bash shells/data/run_vrs_preprocessing.sh
```

```bash
# Convert the images to videos of the RGB folder for testing
python -m tlod.scripts.video.images_to_videos --data_root data/adt --camera_dirs "synthetic_video/camera-rgb-rectified-600-h1000-factory-calib"
```

After this, you should have a `data/adt/Apartment_release_multiuser_cook_seq141_M1292/synthetic_video/camera-rgb-rectified-600-h1000-factory-calib` folder corresponding to the format discussed above.

### Custom Datasets

To convert other posed video data to the *4DGT* format, you just need to restructure the images in the previously mentioned format.

Additionally, for faster dataloading from network storage, you can convert the input to videos using `tlod.scripts.video.images_to_videos`.

Example:

```bash
python -m tlod.scripts.video.images_to_videos --data_root data/adt --camera_dirs "synthetic_video/camera-rgb-rectified-600-h1000-factory-calib"
```

We provide a few conversion scripts for some datasets in `tlod/scripts/data` for your reference when creating custom datasets:

```bash
arkit_to_tlod # for the ARKitTrack dataset # https://github.com/lawrence-cj/ARKitTrack
cop3d_to_tlod # for the COP3D dataset # https://github.com/facebookresearch/cop3d
dycheck_to_tlod # for the DyCheck dataset # https://github.com/KAIR-BAIR/dycheck
epic_to_tlod # for the EPIC-KITCHENS dataset # https://epic-kitchens.github.io/
tum_to_tlod # for the TUM-DYNAMIC dataset # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
```

## Inference

To run the inference, make sure the GPU has at least 16GB of available VRAM.

We provide three different ways to run inference, each with different trade-offs:

### Method 1: Shell Script (Original)

The original inference method using shell scripts that wraps the full training pipeline in test mode:

```bash
# Run inference on the last 128 frames of the processed sequence
# Create novel view rendering at 1.42222s and 2.84444s
# And store everything with a prefix "bowl"
# Results will be stored in "logs/4DGT/tests/bowl"
EXP_NAME="4DGT" \
data_path="data/adt" \
extra_input="\
--seq_data_root synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
--eval_prefix bowl \
--seq_list Apartment_release_multiuser_cook_seq141_M1292 \
--frame_sample -128 None 1 \
--novel_view_timestamps 1.42222 2.84444  \
" \
bash shells/mast/fdgrm.sh

# Convert the test images to a video
python -m tlod.scripts.video.test_images_to_video --data_root logs/4DGT/tests/bowl --prefix bowl --no_ignore_depth --no_ignore_normal --no_ignore_motion_mask --no_ignore_flow_vis
```

**Under the hood**: Launches the full training codebase (`main.py`) in TEST mode with distributed data parallel setup. Processes sequentially like Method 2 but includes all training infrastructure (metrics, logging, checkpointing).

### Method 2: Python Script (Simplified)

A simplified Python interface with cleaner configuration. This method processes each batch sequentially:
1. **Load batch** → 2. **Generate 4D Gaussians** → 3. **Render views** → 4. **Save outputs** → 5. **Repeat**

```bash
# Run inference using the Python script
python -m tlod.run \
    mode=TEST \
    checkpoint=logs/4DGT/checkpoints/last.pth \
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
    frame_sample="[-128, null, 1]" \
    novel_view_timestamps="[1.42222, 2.84444]"
```

**Under the hood**: Each batch waits for the previous one to complete. The model encoder generates Gaussians, then the renderer processes them, then saves to disk - all in sequence.

### Method 3: Asynchronous Pipeline (Fastest)

For improved performance through parallel Gaussian generation and rendering. This method uses a producer-consumer pattern:
- **Thread 1**: Continuously generates 4D Gaussians and queues them
- **Thread 2**: Continuously pulls Gaussians from queue and renders them
- **Thread 3**: Asynchronously saves outputs to disk

```bash
# Run async inference with parallel processing
python -m tlod.run_async \
    mode=TEST \
    checkpoint=logs/4DGT/checkpoints/last.pth \
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
    frame_sample="[-128, null, 1]" \
    novel_view_timestamps="[1.42222, 2.84444]"
```

**Under the hood**: While batch N is being rendered, batch N+1 is already generating Gaussians, and batch N-1 is being saved. This pipeline parallelism eliminates idle GPU time.

### Comparison of Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Shell Script** | • Full control<br>• All original options<br>• Battle-tested | • Complex syntax<br>• Less readable | Production runs with specific configs |
| **Python Script** | • Clean syntax<br>• Easy to modify<br>• Hydra config support | • Sequential processing<br>• Standard speed | Quick experiments and debugging |
| **Async Pipeline** | • 2-3x faster<br>• Parallel processing<br>• Saves Gaussians | • Higher memory usage<br>• New implementation | Long sequences, multiple batches |

### Advanced Options

All Python scripts support additional parameters:

```bash
# Common options for both run.py and run_async.py
save_video=true              # Save outputs as video files
save_depth=true              # Also save depth maps  
save_normal=true             # Also save normal maps
save_flow=true               # Also save optical flow
exp_name=my_experiment       # Custom experiment name
eval_prefix=my_prefix        # Prefix for output files

# Async-specific options (run_async.py only)
+max_parallel_batches=5      # Control parallelism (default: 3)
+output_dir=custom_output    # Override output directory
```

### Output Structure

By running any of the above inference scripts, you should have a folder containing the rendered results:

A composited video for the rendered novel views should be available at `logs/4DGT/tests/bowl/bowl.mp4`.


## GUI & Interactive Viewer

### Saved Gaussian Parameters

In the rendering results, we store the raw 4DGS parameters as:
- `logs/4DGT/tests/bowl/bowl1_000000_gs.npz` - Gaussian parameters
- `logs/4DGT/tests/bowl/bowl1_000000_cam.npz` - Camera parameters

These can be rendered with our interactive viewers.

### Interactive Web Viewer (New)

We provide an interactive web-based viewer with asynchronous Gaussian generation:

```bash
# Run the interactive viewer with default ADT dataset (full spatial-temporal LoD model)
python -m tlod.run_viewer \
    mode=TEST \
    checkpoint=logs/4DGT/checkpoints/last.pth \
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
    frame_sample="[-128, null, 1]" \
    +viewer_port=8080

# Alternative: Run viewer with Stage 1 checkpoint (simplified model without spatial-temporal LoD)
# This checkpoint uses a single-level architecture without hierarchical processing
# Useful for comparison or when lower complexity is needed
python -m tlod.run_viewer \
    mode=TEST \
    checkpoint=logs/4DGT_stage1/checkpoints/last.pth \  # Stage 1 model (no LoD structure)
    config=configs/models/tlod.py \                     # Override model configuration
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
    frame_sample="[-128, null, 1]"                      # Process last 128 frames

# Quick test with only 10 frames
python -m tlod.run_viewer \
    mode=TEST \
    checkpoint=logs/4DGT/checkpoints/last.pth \
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000-factory-calib \
    frame_sample="[-10, null, 1]" \
    +viewer_port=8080
```

Features:
- **Real-time rendering**: Interactive camera control in browser
- **Async generation**: Gaussians generated in background while viewing
- **Auto-save**: Gaussians automatically saved for later use
- **Web interface**: Access at `http://localhost:8080`

The viewer will:
1. Load the 4DGT model
2. Start generating Gaussians asynchronously
3. Save Gaussians to disk for reuse
4. Provide interactive rendering (implementation in progress)

### Native GUI

TODO: Add documentation for the custom native GUI.

## Citation

```
@inproceedings{xu20254dgt
    title     = {4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos},
    author    = {Xu, Zhen and Li, Zhengqin and Dong, Zhao and Zhou, Xiaowei and Newcombe, Richard and Lv, Zhaoyang},
    journal   = {arXiv preprint arXiv:2506.08015},
    year      = {2025}
}
```

## TODO 

- [ ] Clean up all the paths (in particular manifold path) within tlod/scripts/video/images_to_video.py (for Zhen)
- [ ] Streamline all the dataset preparation scripts into one command: 
    - Verify the scripts works for dataset within the AWS path. [for Zhaoyang]
    - Verify the dataset matches from public downloads and provide the download scripts. [for Zhen]
    - Streamline the dataset preparations. [for Claude]
- [ ] Clean up the running scripts in fdgrm.sh