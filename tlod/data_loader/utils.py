# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import gzip
import json

from collections.abc import Sequence
from typing import List

import cv2
import numpy as np
import torch

from os.path import isfile, splitext
from ..easyvolcap.utils.console_utils import dirname, join, red, run
from ..easyvolcap.utils.data_utils import (  # noqa: F401
    as_numpy_func
)
from ..easyvolcap.utils.math_utils import affine_inverse, affine_padding
from ..easyvolcap.utils.parallel_utils import parallel_execution

from ..misc.io_helper import pathmgr


def prepare_images(
    data_root,
    key,
    seq_data_root,
    video_file,
    frame_sample: List[int] = (0, None, 1),
    ims=None,
    clean_up: bool = True,
    resize=256,
    force_reload=False,
):
    disk_file = join(data_root, key, seq_data_root, video_file)
    img_root = join(data_root, key, seq_data_root, splitext(video_file)[0])
    
    # Check if images directory exists directly
    images_dir = "images" if ims is None else dirname(ims[0])
    disk_dir = join(data_root, key, seq_data_root, images_dir) 

    # Prioritize loading images if they exist instead of decoding from videos
    # Only extract images from videos if they exist
    if pathmgr.isdir(disk_dir) and not force_reload:
        img_root = pathmgr.get_local_path(disk_dir, recursive=True)  # local path        
        # List images in the directory
        ims = sorted(pathmgr.ls(img_root))
        ims = [join(images_dir, im) for im in ims if im.endswith(('.png', '.jpg', '.jpeg'))]
        ims = np.asarray(ims)

        # Apply frame sampling
        b, e, s = frame_sample
        ims = ims[b:e:s]
    else:
        # FIXME: Possibly overwriting images if they exist
        ims = prepare_video(
            disk_file,
            img_root,
            frame_sample,
            clean_up=clean_up,
            resize=resize,
        )
    return ims


def load_aria_images(
    ims: List[str],  # path related
    prefix,
    xs,  # target location
    ys,  # target location
    Ws,  # target shape
    Hs,  # target shape
    hs,  # original shape
    ws,  # original shape
    rotate: bool = True,
    sequential: bool = True,
):
    ims = [join(prefix, i) for i in ims]
    ims = np.asarray(ims)
    imgs = parallel_execution(
        ims.tolist(), action=load_image, sequential=sequential
    )  # avoid overloading the CPU

    for i in range(len(ims)):
        Hi, Wi = imgs[i].shape[:2]
        if Hi != hs[i] or Wi != ws[i]:
            imgs[i] = cv2.resize(imgs[i], dsize=(ws[i], hs[i]))
        imgs[i] = imgs[i][ys[i] : ys[i] + Hs[i], xs[i] : xs[i] + Ws[i]]  # Cropping
        if len(imgs[i].shape) == 2:
            # Sometimes we'll be missing a loaded last dimension
            imgs[i] = imgs[i][..., None]
    imgs = np.stack(imgs)
    imgs = imgs.astype(np.float32)

    if rotate:
        imgs = rotate_images(imgs)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def load_aria_npys(
    ims: List[str],
    prefix: str,
    xs,  # target location
    ys,  # target location
    Ws,  # target shape
    Hs,  # target shape
    hs,  # original shape
    ws,  # original shape
    rotate: bool = True,
    sequential: bool = True,
):
    ims = [join(prefix, i) for i in ims]
    ims = np.asarray(ims)
    imgs = parallel_execution(
        ims.tolist(), action=load_npy, sequential=sequential
    )  # avoid overloading the CPU
    # N, H, W, 1
    # imgs = [img["dpt"] for img in imgs if not isinstance(img, np.ndarray)]
    # if not isinstance(imgs[0], np.ndarray):
    #     imgs = [img["dpt"] for img in imgs]

    for i in range(len(ims)):
        Hi, Wi = imgs[i].shape[:2]
        if Hi != hs[i] or Wi != ws[i]:
            imgs[i] = cv2.resize(imgs[i], dsize=(ws[i], hs[i]))
        imgs[i] = imgs[i][ys[i] : ys[i] + Hs[i], xs[i] : xs[i] + Ws[i]]  # Cropping
        if len(imgs[i].shape) == 2:
            # Sometimes we'll be missing a loaded last dimension
            imgs[i] = imgs[i][..., None]

    imgs = np.stack(imgs)
    imgs = imgs.astype(np.float32)

    if rotate:
        imgs = rotate_images(imgs)
    return imgs


def load_npy(file):
    """Load a numpy array from .npy or .npz file.
    
    Args:
        file: Path to the .npy or .npz file
        
    Returns:
        numpy array with the loaded data
    """
    data = np.load(file)
    
    # Handle .npz files that might contain depth data under different keys
    if not isinstance(data, np.ndarray):
        if "dpt" in data:
            data = data["dpt"]
        elif "depth" in data:
            data = data["depth"]
        else:
            # If neither key exists, try to get the first array
            keys = list(data.keys())
            if keys:
                data = data[keys[0]]
    
    return data


ffmpeg_bin = "/usr/bin/ffmpeg"


def prepare_video(
    disk_file: str,
    img_root: str,
    frame_sample: List[int] = (0, None, 1),
    resize: int = 256,
    fps=-1,
    clean_up: bool = True,
):
    if not pathmgr.exists(img_root) or len(pathmgr.ls(img_root)) == 0:
        # Get local path for the disk file
        local_path = pathmgr.get_local_path(disk_file)

        # Decode videos to align with the paths
        ffmpeg_bin = ensure_ffmpeg()
        pathmgr.mkdirs(img_root)

        # Decode the sequences to img_root and rename them to match the image names
        vf = f'-vf "scale={resize}:-1"' if resize > 0 else ""
        fps = f"-r {fps}" if fps > 0 else ""
        # Try hardware acceleration first, but don't specify hevc_cuvid since it's not available
        cmd = f'{ffmpeg_bin} -hwaccel cuda -loglevel error -hide_banner -i {local_path} {vf} {fps} -q:v 1 -qmin 1 -compression_level 100 -start_number 0  {join(img_root, "%06d.jpg")} -y '
        try:
            run(cmd, quite=True, skip_failed=True)  # sometimes the images already exist
        except Exception as e:
            print(red(f"Failed to run ffmpeg with hardware acceleration: {e}"))
            print(red(f"Command: {cmd}"))
            # Rerun without hw acceleration
            cmd = f'{ffmpeg_bin} -loglevel error -hide_banner -i {local_path} {vf} {fps} -q:v 1 -qmin 1 -compression_level 100 -start_number 0  {join(img_root, "%06d.jpg")} -y '
            run(cmd, quite=True, skip_failed=True)  # sometimes the images already exist

        # Clean up local path if it's different from disk file (temporary copy)
        if clean_up and local_path != disk_file:
            pathmgr.rm(local_path)

    ims = sorted(pathmgr.ls(img_root))
    ims = [join(img_root.split("/")[-1], im) for im in ims]
    ims = np.asarray(ims)

    b, e, s = frame_sample
    ims = ims[b:e:s]
    return ims


def ensure_ffmpeg():
    """
    Check if ffmpeg is available in the system.
    Returns the ffmpeg binary path if found, otherwise raises an error.
    """
    global ffmpeg_bin
    
    # Check if the default ffmpeg exists
    if not isfile(ffmpeg_bin):
        # Try to find ffmpeg in PATH
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        
        if ffmpeg_path:
            ffmpeg_bin = ffmpeg_path
        else:
            raise RuntimeError(
                "ffmpeg not found! Please install ffmpeg on your system.\n"
                "On Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "On macOS: brew install ffmpeg\n"
                "On other systems, please refer to: https://ffmpeg.org/download.html"
            )
    
    return ffmpeg_bin


def segment_times_vectorized(durations, k):
    """
    Load-balance sequences across k workers for distributed training.
    
    Uses a greedy algorithm to distribute sequences based on their durations,
    ensuring each worker gets approximately equal total processing time.
    This prevents imbalanced workloads when sequence lengths vary significantly.
    
    Args:
        durations: Array of sequence durations (in seconds)
        k: Number of workers/partitions to distribute across
        
    Returns:
        List of k partitions, each containing indices of assigned sequences
    """
    # Sort durations in descending order (longest first for better bin packing)
    sorted_indices = np.argsort(durations)[::-1]
    sorted_durations = durations[sorted_indices]
    # Initialize partitions and their properties
    partitions = [[] for _ in range(k)]
    partition_sums = np.zeros(k)
    partition_counts = np.zeros(k, dtype=int)
    # Assign each duration to the partition with the smallest sum and fewest sequences
    for index, duration in zip(sorted_indices, sorted_durations):
        # Find the partition with the smallest sum (with count as tie-breaker)
        min_index = np.argmin(partition_sums + partition_counts * 1e-6)
        partitions[min_index].append(index)
        partition_sums[min_index] += duration
        partition_counts[min_index] += 1
    return partitions




def pack_c2ws_to_cameras(
    c2ws: np.ndarray, Ks: np.ndarray, Hs: np.ndarray, Ws: np.ndarray
):
    c2ws = as_numpy_func(affine_padding)(c2ws)  # avoid inplace modification
    c2ws_gl = c2ws.copy()
    c2ws_gl[..., :3, 1:3] *= -1  # OpenCV to OpenGL to accommodate hubert's impl
    fovx = np.arctan(Ws / Ks[..., 0, 0] / 2) * 2
    fovy = np.arctan(Hs / Ks[..., 1, 1] / 2) * 2
    ppx = Ks[..., 0, 2] / Ws
    ppy = Ks[..., 1, 2] / Hs
    cameras = c2ws_gl.reshape(-1, 16)
    cameras = np.concatenate(
        [cameras, np.stack([fovx, fovy, ppx, ppy], axis=-1)], axis=-1
    )
    return cameras


def rotate_cameras(RTs: np.ndarray, Ks: np.ndarray, Ws: np.ndarray, Hs: np.ndarray):
    Ks_copy = Ks.copy()
    # RTs_copy = RTs.copy()
    Ks[..., 0, 0] = Ks_copy[..., 1, 1]
    Ks[..., 0, 2] = Ks_copy[..., 1, 2]
    Ks[..., 1, 1] = Ks_copy[..., 0, 0]
    Ks[..., 1, 2] = Ks_copy[..., 0, 2]
    Ks[..., 0, 2] = Hs - Ks[..., 0, 2]  # wtf...
    c2ws = as_numpy_func(affine_inverse)(RTs)
    c2ws_copy = c2ws.copy()
    c2ws[..., :3, 0:1] = -c2ws_copy[..., :3, 1:2]
    c2ws[..., :3, 1:2] = c2ws_copy[..., :3, 0:1]
    RTs[:] = as_numpy_func(affine_inverse)(c2ws)
    Hs_val = Hs.copy()
    Ws_val = Ws.copy()
    Hs[:], Ws[:] = Ws_val[:], Hs_val[:]


def rotate_images(imgs: np.ndarray):
    imgs = np.transpose(imgs, (0, 2, 1, 3))  # flip images around
    imgs = np.flip(imgs, 2)  # flip around the transposed x axis
    imgs = np.ascontiguousarray(imgs)
    return imgs


def load_camera_poses(
    json_file: str,
    frame_sample: List[int] = (0, None, 1),
    loaded_to_seconds: float = 1e9,
    loaded_to_meters: float = 1.0,
):
    """
    Load camera poses and metadata from a JSON file (NeRF-style format).
    
    This function reads camera calibration data typically stored in transforms.json
    files used in NeRF-based reconstructions. It handles both regular and gzipped
    JSON files and applies necessary transformations for coordinate systems.
    
    Assume the input file transforms matrix is camera-to-world.

    Args:
        json_file: Path to the JSON file containing camera poses
        frame_sample: Tuple (begin, end, step) for subsampling frames
        loaded_to_seconds: Scale factor to convert timestamps to seconds
                          (e.g., 1e9 for nanoseconds to seconds)
        loaded_to_meters: Scale factor to convert translations to meters
                         (e.g., 1000.0 for millimeters to meters)
    
    Returns:
        Tuple of 6 arrays:
        - ims: Image file paths (np.ndarray of strings, shape [F])
               Paths to the image files for each frame
        - Hs: Image heights in pixels (np.ndarray of int32, shape [F])
              Height of each image frame
        - Ws: Image widths in pixels (np.ndarray of int32, shape [F])
              Width of each image frame
        - Ks: Camera intrinsic matrices (np.ndarray of float32, shape [F, 3, 3])
              Contains focal lengths (fx, fy) and principal points (cx, cy)
        - RTs: Camera extrinsic matrices in world-to-camera (w2c) format
               (np.ndarray of float32, shape [F, 4, 4])
               Transformed from camera-to-world (c2w) format in the JSON
        - ts: Timestamps in seconds (np.ndarray of float64, shape [F])
              Time of each frame capture, normalized to seconds
    """
    # Load JSON data from file (supports both plain and gzipped)
    with pathmgr.open(json_file, "r") as fIn:
        if json_file.endswith(".json"):
            data = json.load(fIn)
        else:
            with gzip.open(fIn, "rt", encoding="utf-8") as f:
                return json.load(f)

    # Extract frames array from the JSON structure
    if isinstance(data, dict):
        frames = data["frames"]
    else:
        frames = data
    
    # Apply frame subsampling
    b, e, s = frame_sample
    frames = frames[b:e:s]

    # Extract image paths
    ims = [f["image_path"] for f in frames]
    ims = np.asarray(ims)
    
    # Extract image dimensions
    Hs = [np.asarray(f["h"], dtype=np.int32) for f in frames]
    Hs = np.stack(Hs)  # F,
    Ws = [np.asarray(f["w"], dtype=np.int32) for f in frames]
    Ws = np.stack(Ws)  # F,
    
    # Build camera intrinsic matrices
    # Supports both "fx/fy" and "fl_x/fl_y" naming conventions
    Ks = [
        np.asarray(
            [
                [f.get("fx", f.get("fl_x", None)), 0, f["cx"]],
                [0, f.get("fy", f.get("fl_y", None)), f["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        for f in frames
    ]
    Ks = np.stack(Ks)  # F, 3, 3
    RTs = [np.asarray(f["transform_matrix"], dtype=np.float32) for f in frames]  # c2w
    RTs = np.stack(RTs)  # F, 4, 4
    RTs[..., :3, 3] /= loaded_to_meters
    RTs = as_numpy_func(affine_inverse)(RTs)  # w2c
    ts = [np.asarray(f["timestamp"], dtype=np.float64) for f in frames]
    ts = np.stack(ts) / loaded_to_seconds  # F, 4, 4
    return ims, Hs, Ws, Ks, RTs, ts


def save_camera_poses(
    json_file: str,
    ims: np.ndarray,
    Hs: np.ndarray,
    Ws: np.ndarray,
    Ks: np.ndarray,
    RTs: np.ndarray,
    ts: np.ndarray,
    loaded_to_seconds: float = 1e9,
    loaded_to_meters: float = 1.0,
):
    """
    Save camera poses to a JSON file, reversing the process of load_camera_poses.

    Args:
        json_file: Path to the output JSON file
        ims: Array of image paths
        Hs: Array of image heights
        Ws: Array of image widths
        Ks: Array of camera intrinsic matrices (F, 3, 3)
        RTs: Array of camera extrinsic matrices (F, 4, 4) in w2c format
        ts: Array of timestamps
        loaded_to_seconds: Scale factor to convert timestamps
        loaded_to_meters: Scale factor to convert translations
    """
    if isinstance(RTs, torch.Tensor):
        RTs = RTs.cpu().numpy()
    if isinstance(Ks, torch.Tensor):
        Ks = Ks.cpu().numpy()
    if isinstance(ts, torch.Tensor):
        ts = ts.cpu().numpy()

    # Convert w2c to c2w
    RTs_c2w = as_numpy_func(affine_inverse)(RTs)

    # Scale translations back
    RTs_c2w[..., :3, 3] *= loaded_to_meters

    # Scale timestamps back
    ts_scaled = ts * loaded_to_seconds

    # Create frames data
    frames = []
    for i in range(len(ims)):
        frame = {
            "image_path": ims[i],
            "h": int(Hs[i]),
            "w": int(Ws[i]),
            "fx": float(Ks[i, 0, 0]),
            "fy": float(Ks[i, 1, 1]),
            "cx": float(Ks[i, 0, 2]),
            "cy": float(Ks[i, 1, 2]),
            "transform_matrix": RTs_c2w[i].tolist(),
            "timestamp": float(ts_scaled[i]),
        }
        frames.append(frame)

    # Create the data dictionary
    data = {"frames": frames}

    # Write to file
    with pathmgr.open(json_file, "w") as fOut:
        if json_file.endswith(".json"):
            json.dump(data, fOut, indent=4)
        else:
            with gzip.open(fOut, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=4)


def load_image(path: str, normalize: bool = True):
    """Load an image using PIL.
    
    Args:
        path: Path to the image file
        normalize: If True, normalize pixel values to [0, 1] range
        
    Returns:
        numpy array with shape (H, W, C) for RGB or (H, W) for grayscale
    """
    from PIL import Image
    
    # Open image using PIL
    with Image.open(path) as img:
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize if requested
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array


def compute_rays(fov, matrix, res):
    """
    Screen space convention:
    x = right
    y = up
    z = into camera

    fov: Image space (h, w) tuple or int for square image.
         Note that (h, w) is in image space, which is (y,x) in screen space.
    """
    res_tuple = res if isinstance(res, Sequence) else (res, res)
    matrix = np.array(matrix)
    rays_o = np.zeros((res_tuple[0], res_tuple[1], 3), dtype=np.float32) + matrix[
        0:3, 3
    ].reshape(1, 1, 3)
    rays_o = rays_o

    # h_axis, w_axis is the 2D image space axis in height/width direction
    h_axis = np.linspace(0.5, res_tuple[0] - 0.5, res_tuple[0]) / res_tuple[0]
    w_axis = np.linspace(0.5, res_tuple[1] - 0.5, res_tuple[1]) / res_tuple[1]
    h_axis = 2 * h_axis - 1
    w_axis = 2 * w_axis - 1
    x, y = np.meshgrid(w_axis, h_axis)  # Default indexing="xy" behavior
    if isinstance(fov, Sequence):
        x = x * np.tan(fov[1] / 2.0)  # fov order is (w, h)
        y = y * np.tan(fov[0] / 2.0)
    else:
        x = x * np.tan(fov / 2.0)
        y = y * np.tan(fov / 2.0)

    # At this point
    #   x is im_width  axis = right
    #   y is im_height axis = down

    y = y * -1
    z = -np.ones(res_tuple)
    rays_d_un = np.stack([x, y, z], axis=-1)
    rot = matrix[0:3, 0:3][None, None, :, :]
    rays_d_un = np.sum(rot * rays_d_un[:, :, None, :], axis=-1)
    rays_d = rays_d_un / np.linalg.norm(rays_d_un, axis=-1)[:, :, None]

    return rays_o, rays_d, rays_d_un
