# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2

import numpy as np
import projectaria_tools.core.mps as mps
from PIL import Image

from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import interpolate, SE3

@dataclass
class AriaFrame:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_params: np.ndarray
    w: int
    h: int
    file_path: str
    camera_modality: str
    transform_matrix: np.ndarray
    camera2device: np.ndarray
    timestamp: int
    timestamp_read_start: int
    timestamp_read_end: int
    exposure_duration_s: float
    gain: float
    camera_name: str
    transform_matrix_read_start: np.ndarray = None
    transform_matrix_read_end: np.ndarray = None


@dataclass
class PoseSE3:
    transform_world_device: SE3
    linear_velocity: np.array
    angular_velocity: np.array
    quality_score: float
    time_ns: int


def interpolate_closeloop_trajectory_pose(
    start_pose: mps.ClosedLoopTrajectoryPose,
    end_pose: mps.ClosedLoopTrajectoryPose,
    time_ns: int,
):
    start_time = start_pose.tracking_timestamp.total_seconds() * 1e9
    end_time = end_pose.tracking_timestamp.total_seconds() * 1e9
    ratio = (time_ns - start_time) / (end_time - start_time)

    assert (
        ratio >= 0.0 and ratio <= 1.0
    ), f"interpolation ratio {ratio} is not within [0.0, 1.0]"
    interp_pose_SE3 = interpolate(
        start_pose.transform_world_device, end_pose.transform_world_device, ratio
    )
    interp_linear_velocity = (
        start_pose.device_linear_velocity_device * (1 - ratio)
        + end_pose.device_linear_velocity_device * ratio
    )
    interp_angular_velocity = (
        start_pose.angular_velocity_device * (1 - ratio)
        + end_pose.angular_velocity_device * ratio
    )
    interp_pose_score = (
        start_pose.quality_score * (1 - ratio) + end_pose.quality_score * ratio
    )

    return PoseSE3(
        transform_world_device=interp_pose_SE3,
        linear_velocity=interp_linear_velocity,
        angular_velocity=interp_angular_velocity,
        quality_score=interp_pose_score,
        time_ns=int(time_ns),
    )


def interpolate_aria_pose(closed_loop_traj, capture_time_ns) -> PoseSE3:

    start_idx = mps.utils.bisection_timestamp_search(closed_loop_traj, capture_time_ns)
    if start_idx is None:
        return None

    if (
        closed_loop_traj[start_idx].tracking_timestamp.total_seconds() * 1e9
        > capture_time_ns
    ):
        start_idx -= 1

    start_pose = closed_loop_traj[start_idx]

    if start_idx + 1 >= len(closed_loop_traj):
        return closed_loop_traj[start_pose]

    end_pose = closed_loop_traj[start_idx + 1]

    interp_pose = interpolate_closeloop_trajectory_pose(
        start_pose, end_pose, time_ns=capture_time_ns
    )

    return interp_pose


def seek_nearest_timestamp(timestamps_ns, capture_time_ns, tolerance_diff=1e6):
    """
    assume the input timestamps are sorted already
    """

    if capture_time_ns < timestamps_ns[0] - tolerance_diff:
        return None

    if capture_time_ns > timestamps_ns[-1] + tolerance_diff:
        return None

    nearest_pose_idx = np.searchsorted(timestamps_ns, capture_time_ns)

    if nearest_pose_idx == 0 or nearest_pose_idx >= len(timestamps_ns) - 1:
        return min(nearest_pose_idx, len(timestamps_ns) - 1)

    # choose the nearest_idx
    timestamp_lower = timestamps_ns[nearest_pose_idx - 1]
    timestamp_upper = timestamps_ns[nearest_pose_idx]

    if (capture_time_ns - timestamp_lower) < (timestamp_upper - capture_time_ns):
        return nearest_pose_idx - 1
    else:
        return nearest_pose_idx


def project(
    point3d: np.ndarray,
    T_w2c: np.ndarray,
    calibK: np.ndarray,
    frame_h: int,
    frame_w: int,
):
    rot = T_w2c[:3, :3]
    t = T_w2c[:3, 3:]
    point3d_cam = rot @ point3d + t
    point3d_proj = calibK @ point3d_cam

    u_proj = point3d_proj[0] / point3d_proj[2]
    v_proj = point3d_proj[1] / point3d_proj[2]
    z = point3d_proj[2]

    if point3d.shape[-1] > 1:
        mask = (
            (u_proj > 0)
            * (u_proj < frame_w)
            * (v_proj > 0)
            * (v_proj < frame_h)
            * (z > 0)
        )
        return u_proj[mask > 0], v_proj[mask > 0], z[mask > 0], mask
    else:
        if (
            u_proj < 0
            or u_proj >= frame_w - 1
            or v_proj < 0
            or v_proj >= frame_h - 1
            and z > 0
        ):
            return None, None, None
        return u_proj, v_proj, z


def read_frames_from_metadata(transforms_json: str = "transforms.json"):
    metadata_path = transforms_json
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    frames_raw = metadata["frames"]

    frames = []
    for f in frames_raw:
        file_path = f["file_path"]

        # get intrinsic calibration parameters
        # The fisheye624 calibration parameter vector has the following interpretation:
        # params = [f c_u c_v k_0 k_1 k_2 k_3 k_4 k_5 p_0 p_1 s_0 s_1 s_2 s_3]
        frames.append(
            AriaFrame(
                fx=f["fl_x"],
                fy=f["fl_y"],
                cx=f["cx"],
                cy=f["cy"],
                distortion_params=np.asarray(f["distortion_params"]),
                w=f["w"],
                h=f["h"],
                file_path=str(file_path),
                camera_modality=f["camera_modality"],
                camera2device=np.asarray(f["camera2device"]),
                transform_matrix=np.asarray(f["transform_matrix"]),
                transform_matrix_read_start=np.asarray(
                    f["transform_matrix_read_start"]
                ),
                transform_matrix_read_end=np.asarray(f["transform_matrix_read_end"]),
                timestamp=f["timestamp"],
                timestamp_read_start=f["timestamp_read_start"],
                timestamp_read_end=f["timestamp_read_end"],
                exposure_duration_s=f["exposure_duration_s"],
                gain=f["gain"],
                camera_name=f["camera_name"],
            )
        )

    print(
        f"There are {len(frames)} valid frames out of {len(frames_raw)} frames in total."
    )

    return frames


def undistort_image(
    frame: AriaFrame,
    image_raw: np.ndarray,
    camera_model: str,
    ow: int,
    oh: int,
    f: float,
):
    """
    replace it by the data provider API
    """
    # we assume fx and fy are the same
    proj_params = np.asarray([frame.fx, frame.cx, frame.cy])
    proj_params = np.concatenate((proj_params, frame.distortion_params))

    transform_matrix = SE3()

    frame_calib = calibration.CameraCalibration(
        frame.camera_name,
        calibration.CameraModelType.FISHEYE624,
        proj_params,
        transform_matrix,
        frame.w,
        frame.h,
        None,
        90.0,
        "",
    )

    if camera_model == "linear":
        out_calib = calibration.get_linear_camera_calibration(ow, oh, f)
    elif camera_model == "spherical":
        out_calib = calibration.get_spherical_camera_calibration(ow, oh, f)
    else:
        raise NotImplementedError(
            f"cannot recognized camera model type {camera_model} !"
        )

    image = calibration.distort_by_calibration(image_raw, out_calib, frame_calib)

    return image


def center_segment(image, boundary=400, thres=40):
    """
    Segment the image with central crop.
    @todo: this function has not been tested for a while. May need to deprecate.
    """
    from rembg import remove

    bg_canvas = np.zeros_like(image[..., 0])

    center_image = image[boundary:-boundary, boundary:-boundary]
    output = remove(center_image)

    bg_canvas[boundary:-boundary, boundary:-boundary] = output[..., 3]

    bg_canvas[bg_canvas < thres] = 0
    bg_canvas[bg_canvas > thres] = 255
    return bg_canvas


# the multiprocessor version of the code
def process_frame(
    frame: AriaFrame,
    data_root: Path,
    output_folder: Path,
    camera_model: str,
    rectified_focal: float,
    rectified_h: int,
    rectified_w: int = None,
):

    filename = os.path.basename(frame.file_path)

    # Force the output images to use png format
    filename = f"{filename}"[:-3] + "png"
    image_output_subpath = f"images/{filename}"
    image_output_path = output_folder / image_output_subpath
    image_output_path.parent.mkdir(exist_ok=True)

    image_raw = np.array(Image.open((str(data_root / frame.file_path)))) 

    if rectified_w is None:
        rectified_w = int((rectified_h / frame.h) * frame.w)

    image = undistort_image(
        frame, image_raw, camera_model, rectified_w, rectified_h, rectified_focal
    )

    # Reference: https://github.com/facebookresearch/projectaria_tools/blob/main/core/calibration/CameraCalibration.cpp#L153
    output_fx = rectified_focal
    output_fy = rectified_focal
    output_cx = (rectified_w - 1) / 2.0
    output_cy = (rectified_h - 1) / 2.0

    Image.fromarray(image).save(image_output_path)

    # Original Surreal coordinate system is the same as the coordinate system used in COLMAP as well as nerfstudio.
    # "The local camera coordinate system of an image is defined in a way that the X axis points to the right,
    # the Y axis to the bottom, and the Z axis to the front as seen from the image,
    transform = frame.transform_matrix

    # get the image mask over rectified frame
    return {
        "fx": output_fx,
        "fy": output_fy,
        "cx": output_cx,
        "cy": output_cy,
        "w": rectified_w,
        "h": rectified_h,
        "image_width_raw": frame.w,
        "image_height_raw": frame.h,
        "image_path": image_output_subpath,
        "image_modality": frame.camera_modality,
        "mask_path": "",
        "camera2device": frame.camera2device.tolist(),
        "transform_matrix": transform.tolist(),
        # "transform_matrix_read_start": frame.transform_matrix_read_start.tolist(),
        # "transform_matrix_read_end": frame.transform_matrix_read_end.tolist(),
        "timestamp": frame.timestamp,
        "timestamp_read_start": frame.timestamp_read_start,
        "timestamp_read_end": frame.timestamp_read_end,
        "exposure_duration_s": frame.exposure_duration_s,
        "gain": frame.gain,
    }


def calculate_timestamp_to_data3d(
    semidense_points_data,
    semidense_observations,
    transform_json_file,
):
    """
    Calculate the mapping of visible 3D data points according to each timestamp
    """
    with open(str(transform_json_file)) as json_file:
        transforms = json.loads(json_file.read())

    timestamps = []
    for frame in transforms["frames"]:
        timestamps.append(frame["timestamp"])
    timestamps = np.array(timestamps)

    time_uid_map = {}
    visible_uids = set()
    for data2d in semidense_observations:
        uid = data2d.point_uid
        timestamp_ns = int(data2d.frame_capture_timestamp.total_seconds() * 1e9)

        nearest_idx = seek_nearest_timestamp(timestamps, timestamp_ns)
        if nearest_idx is None:
            continue

        camera_time_ns = str(int(timestamps[nearest_idx]))

        if camera_time_ns in time_uid_map.keys():
            time_uid_map[camera_time_ns].append(uid)
        else:
            time_uid_map[camera_time_ns] = [uid]

        visible_uids.add(uid)

    uid_point3d_map = {}
    for point3d in semidense_points_data:
        if point3d.uid not in visible_uids:
            continue
        uid_point3d_map[point3d.uid] = point3d.position_world.tolist()

    data = {
        "time_uid_map": time_uid_map,
        "visible_uids": list(visible_uids),
        "uid_points3d_map": uid_point3d_map,
    }

    return data


def get_rectified_row_index(
    frame: AriaFrame,
    camera_model: str,
    output_focal: int,
    output_h: int,
    output_w: int = None,
):
    """
    We will generate a row index for the original image from (0-127).
    It can be used to track the row index of the rolling shutter for a RGB camera.

    For details of Aria rolling shutter camera, check:
    https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/temporal_alignment_of_sensor_data#images-formation-temporal-model-rolling-shutter-and-pls-artifact
    """
    H, W = frame.h, frame.w
    array = np.linspace(0, 1, H, dtype=np.float32).reshape(H, 1)
    image_index = np.tile(array, (1, W))

    if output_w is None:
        output_w = int(output_h / H * W)

    image_index_rectified = undistort_image(
        frame, image_index, camera_model, output_w, output_h, output_focal
    )

    return (image_index_rectified * 255).astype(np.uint8)


def get_rectified_vignette_image(
    frame: AriaFrame,
    input_root: Path,
    camera_label: str,
    camera_model: str,
    output_focal: int,
    output_h: int,
    output_w: int = None,
):
    """
    The output image will be a rectified image at the same resolution of input streaming frames.
    """

    if camera_label == "camera-rgb":
        vignette_path = (
            input_root / "vignette_imx577_16bit.png"
        )
    else:
        vignette_path = input_root / "vignette_ov7251.png"

    vignette_raw = np.array(Image.open(os.path.expanduser(vignette_path)))

    vignette_raw = vignette_raw[:, :, :3]

    input_h = frame.h
    input_w = frame.w

    if camera_label == "camera-rgb" and input_w != 2880:
        # apply the correct downscaling operation. Crop the 2880x2880 image to 2816x2816 and then downsample (e.g. 1408x1408)
        assert vignette_raw.shape[0] == 2880
        vignette_raw = vignette_raw[32:-32, 32:-32]
        vignette_raw = cv2.resize(vignette_raw, (input_w, input_h))

    if output_w is None:
        output_w = int(output_h / input_h * input_w)

    # for some unknown reasons, the vignette image in uint8 was given strange grid-like results.
    vignette = undistort_image(
        frame,
        vignette_raw.astype(float),
        camera_model,
        output_w,
        output_h,
        output_focal,
    )

    return vignette.astype(np.uint8)


def get_rectified_mask(
    frame: AriaFrame,
    camera_model: str,
    output_focal: int,
    output_height: int = None,
    input_root: Path = None,
):
    """
    if input_root is not None, it will read the mask.png inside, otherwise it will a full frame mask for the original frame.
    """

    input_h = frame.h
    input_w = frame.w

    if input_root is None:
        mask_raw = np.ones((input_h, input_w)).astype(np.float32)
    else:
        mask_path = input_root / "mask_imx577.png"
        mask_raw = Image.open(os.path.expanduser(mask_path))
        mask_raw = np.array(mask_raw) / 255.0
        mask_raw = cv2.resize(mask_raw, (input_w, input_h))

    if output_height:
        output_w = int(output_height / input_h * input_w)
        output_h = output_height
    else:
        output_w = input_w
        output_h = input_h
    mask = undistort_image(
        frame,
        mask_raw.astype(np.float32),
        camera_model,
        output_w,
        output_h,
        output_focal,
    )
    return (mask * 255.0).astype(np.uint8)


def storePly(path, xyz, rgb, normals=None):
    from plyfile import PlyData, PlyElement

    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
