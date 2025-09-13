
# Data Convention

## Aria Video Preprocessing

For all Aria videos, we use a similar data preprocessing and data structure as [Egocentric Splats](https://github.com/facebookresearch/egocentric_splats), with the full description available at their [documentation](https://github.com/facebookresearch/egocentric_splats/blob/main/docs/preprocess_aria_video.md#preprocessing-explained).

### Key Differences in 4DGT Processing:

1. **RGB Camera Only**: We exclusively use the RGB camera stream and do not process SLAM cameras.

2. **Simplified Rolling Shutter Handling**:
   - We do not model the rolling shutter effect for RGB cameras
   - We use only the center row pose to represent the full image motion
   - The `transform_matrix` in our JSON corresponds to the camera position at the middle of the exposure time

3. **Single Pose Representation**: Unlike Egocentric Splats which provides start/center/end row poses, we use a single `transform_matrix` field that treats the entire frame as captured instantaneously at the center timestamp.

### Aria Data Structure:

The preprocessing generates:
- **RGB images**: Rectified and lens-shading corrected (in `camera-rgb-rectified-*/images/`)
- **transforms.json**: Contains camera intrinsics and per-frame extrinsics
- **vignette.png**: Lens shading correction model
- **videos.mp4**: Compressed video file for efficient loading

### Minimal content of `transforms.json`: 

```json
{
    "frames": [
        {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 499.5,
            "cy": 499.5,
            "w": 1000,
            "h": 1000,
            "image_path": "images/xxxxx1.png",
            "transform_matrix": [
                [
                    -0.20143383390898673,
                    -0.7511175205260028,
                    0.6286866317296691,
                    -0.14306097204890356
                ],
                [
                    -0.9555694270570241,
                    0.009676747745910097,
                    -0.2946072480896778,
                    1.5051132788382027
                ],
                [
                    0.21520102376763378,
                    -0.66009759196041,
                    -0.7196941631397538,
                    3.127329889389685
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "timestamp": 3898243023000.0,
        },
        {
            ... # content of xxxxx2.png, names can be arbitrary, doesn't have to be numbered
        }
    ]
}
```

## Dataloader Output Format

The dataloader returns batches with the following tensor format for TEST mode:

### Required Keys and Shapes

| Key | Shape | Description |
|-----|-------|-------------|
| `c2w_avg` | `[B, 3, 4]` | Average camera-to-world transformation matrix |
| `cameras_input` | `[B, N_in, 20]` | Input camera parameters (16 for 4x4 matrix + 4 for FOV and principal) |
| `cameras_output` | `[B, N_out, 20]` | Output camera parameters |
| `rays_t_un_input` | `[B, N_in]` | Input time values (unnormalized) |
| `rays_t_un_output` | `[B, N_out]` | Output time values (unnormalized) |
| `rgb_input` | `[B, N_in, 3, H, W]` | Input RGB images |
| `rgb_output` | `[B, N_out, 3, H, W]` | Output RGB images |

### Optional Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `monochrome_output` | `[B, N_out]` | Monochrome camera flags |
| `ratios_output` | `[B, N_out]` | Aspect ratios |
| `img_name_output` | `list` | Image filenames |

### Dimension Meanings

- **B**: Batch size (typically 1 for inference)
- **N_in**: Number of input views (dynamically determined by dataset, e.g., 2 or 3)
- **N_out**: Number of output views (default: 6)
- **H, W**: Image height and width (default: 512x512)

### Example Shapes

For a typical test configuration with batch_size=1:

```
c2w_avg:          [1, 3, 4]
cameras_input:    [1, 3, 20]        # 3 input views
cameras_output:   [1, 6, 20]        # 6 output views
rays_t_un_input:  [1, 3]
rays_t_un_output: [1, 6]
rgb_input:        [1, 3, 3, 512, 512]  # [B, N_in, C, H, W]
rgb_output:       [1, 6, 3, 512, 512]  # [B, N_out, C, H, W]
```

### Camera Parameter Format

The 20-dimensional camera parameter contains:
- **[0:16]**: Flattened 4x4 camera-to-world matrix (OpenGL convention)
- **[16]**: FOV_x (field of view in x direction)
- **[17]**: FOV_y (field of view in y direction)
- **[18]**: Principal_x (normalized principal point x coordinate)
- **[19]**: Principal_y (normalized principal point y coordinate)

### Coordinate System Conversion

The dataset returns camera matrices in OpenGL convention. During processing:
1. Y and Z axes are negated to convert to OpenCV convention
2. Camera-to-world (c2w) matrices are inverted to get world-to-camera (w2c) for rendering
