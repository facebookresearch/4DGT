# 4DGT Model Card

## Model Details

4DGT (4D Gaussian Transformer) is a neural network model that learns dynamic 3D Gaussian representations from monocular videos. It uses a transformer-based architecture to predict 4D Gaussians for novel view synthesis in dynamic scenes.

- **Paper:** [4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](https://arxiv.org/abs/2506.08015)
- **Project Page:** [https://4dgt.github.io/](https://4dgt.github.io/)
- **Github:** [GitHub repository](https://github.com/facebookresearch/4dgt)

Please refer to the project page and github for more details of the model. 

## Citation

```bibtex
@inproceedings{xu20254dgt,
    title     = {4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos},
    author    = {Xu, Zhen and Li, Zhengqin and Dong, Zhao and Zhou, Xiaowei and Newcombe, Richard and Lv, Zhaoyang},
    journal   = {arXiv preprint arXiv:2506.08015},
    year      = {2025}
}
```

## Model Files

### Checkpoint: `4dgt_full.pth`
- **Size:** ~14.5 GB
- **Format:** PyTorch state dict
- **Contents:**
  - The full model trained as described in the paper.
  - Encoder weights (DINOv2 backbone)
  - Level of Details Transformer 
  - 4D Gaussian Decoder 

### Checkpoint: `4dgt_1st_stage.pth`
- **Size:** ~4.85 GB
- **Format:** PyTorch state dict
- **Contents:**
  - The first stage model trained only using Egoexo4D dataset as described in the paper. 
  - Encoder weights (DINOv2 backbone)
  - Vanilla Transformer, no level of details. 
  - 4D Gaussian Decoder

## Quick Start
Please refer to [4DGT GitHub repository](https://github.com/facebookresearch/4dgt) for the full set up. 

## Contact
For questions and issues, please open an issue on the [GitHub repository](https://github.com/facebookresearch/4dgt).