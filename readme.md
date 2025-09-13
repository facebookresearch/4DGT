<div align="center">

<h3> 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos </h3>
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

Please cite this paper if you find this repository useful.

``` bash
@inproceedings{xu20254dgt,
    title     = {4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos},
    author    = {Xu, Zhen and Li, Zhengqin and Dong, Zhao and Zhou, Xiaowei and Newcombe, Richard and Lv, Zhaoyang},
    journal   = {arXiv preprint arXiv:2506.08015},
    year      = {2025}
}
```

## Installation

Use the automated installation script:

```bash
bash tlod/scripts/install.sh
```

This script will interactively guide you through setting up the conda environment and installing all dependencies including PyTorch, flash-attention, and apex.

For detailed installation instructions and troubleshooting, see [docs/install.md](docs/install.md).

## Pretrained Model

You can find the pretrained model from [Hugging Face](https://huggingface.co/projectaria/4DGT/) and download manually via:
```bash
# By default, the downloaded model will be saved to checkpoints/4dgt_full.pth
python tlod/download_model.py
```

You can also skip this step and it will automatically download it when executing the following commands.

## Aria Datasets

We provide two examples of converting a typical Aria recording in `.vrs` to the format recognized by *4DGT*. For details of the data format being processed, check [docs/data.md](docs/data.md).


#### Run on an Aria sequence from Aria Explorer

We use the sequence from [the Aria Explorer](https://explorer.projectaria.com/aea/loc3_script3_seq1_rec1?st=%220%22) from Aria Everyday Activity as an example. This approach applies to any sequence downloaded from Aria Explorer. 

```bash
mkdir -p data/aea 
cd data/aea

# Put the download URL JSON file into data/aea folder
# The download URL file will be different if you choose a different sequence. 
aria_dataset_downloader -c loc3_script3_seq1_rec1_download_urls.json -o . -l all
```

Run the following to process the sequence: 
```bash
# Process the sequence "loc3_script3_seq1_rec1"
DATA_INPUT_DIR="data/aea/loc3_script3_seq1_rec1" \
DATA_PROCESSED_DIR="data/aea/loc3_script3_seq1_rec1" \
VRS_FILE="recording.vrs" \
bash tlod/scripts/run_vrs_preprocessing.sh
```

#### Run on Aria Digital Twin sequence

The Aria Digital Twin uses synthetic rendering as an evaluation mechanism. Follow these steps to download an example sequence:

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
DATA_INPUT_DIR="data/adt-raw/Apartment_release_multiuser_cook_seq141_M1292" \
DATA_PROCESSED_DIR="data/adt/Apartment_release_multiuser_cook_seq141_M1292" \
VRS_FILE="synthetic_video.vrs" \
bash tlod/scripts/run_vrs_preprocessing.sh
```

After this, you should have a `data/adt/Apartment_release_multiuser_cook_seq141_M1292/synthetic_video/camera-rgb-rectified-600-h1000` folder corresponding to the format discussed above.

## Inference

We provide a simplified Python interface with cleaner configuration. During execution, each batch waits for the previous one to complete: the model encoder generates Gaussians, the renderer processes them, then saves to disk—all in sequence. Rendering all 128 frames will take some time to complete.

```bash
# Run inference on the above AEA sequence 
# It will produce the novel view spiral rendering on the chosen timestamps among all the frames.
python -m tlod.run \
    data_path=data/aea \
    seq_list=loc3_script3_seq1_rec1 \
    seq_data_root=recording/camera-rgb-rectified-600-h1000 \
    novel_view_timestamps="[1.42222, 2.84444]" 

# Run inference on the above ADT sequence 
# Frame sample indicates running only on the last 128 frames.
python -m tlod.run \
    data_path=data/adt \
    seq_list=Apartment_release_multiuser_cook_seq141_M1292 \
    seq_data_root=synthetic_video/camera-rgb-rectified-600-h1000 \
    frame_sample="[-128, null, 1]" \
    novel_view_timestamps="[1.42222, 2.84444]"
```

To run the inference, make sure the GPU has at least 16GB of available VRAM.

The full evaluation and inference on other datasets is still a work in progress. We will release them in the near future. 

We also provide the first stage model, which is trained only using EgoExo4D data following the description in our paper. It does not use level-of-detail design and will predict much denser Gaussians as output. You can use it as a reference using the following command: 
```bash
python -m tlod.run \
    checkpoint=checkpoints/4dgt_1st_stage.pth \
    exp_name=exp_4dgt_1st_stage \
    config=configs/models/tlod.py \
    data_path=data/aea \
    seq_list=loc3_script3_seq1_rec1 \
    seq_data_root=recording/camera-rgb-rectified-600-h1000 \
    novel_view_timestamps="[1.42222, 2.84444]"
```

## GUI & Interactive Viewer

We provide a simple interactive web-based viewer that renders Gaussians with asynchronous Gaussian generation:

```bash
python -m tlod.run_viewer \
    data_path=data/aea \
    seq_list=loc3_script3_seq1_rec1 \
    seq_data_root=recording/camera-rgb-rectified-600-h1000
```

It has a slider to allow you to control the space (frame) and time. Currently, the asynchronous model prediction process may slow down the interactive rendering depending on which GPU you use. We may enhance this in our future plans. 

## LICENSE 

This implementation is Creative Commons licensed, as found in the LICENSE file.

The work built in this repository benefits from the great work in the following open-source projects:
* [Project Aria tool](https://github.com/facebookresearch/projectaria_tools): Apache 2.0
* [Egocentric Splats](https://github.com/facebookresearch/egocentric_splats): CC-NC, Meta. 
* [EasyVolcap](https://github.com/zju3dv/EasyVolcap): MIT
* [Nerfview](https://github.com/nerfstudio-project/nerfview): Apache 2.0