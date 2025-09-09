# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas

import projectaria_tools.core.sophus as sophus
import rerun as rr

from aria_utils import (
    get_rectified_mask,
    get_rectified_row_index,
    get_rectified_vignette_image,
    interpolate_aria_pose,
    process_frame,
    project,
    read_frames_from_metadata,
)
from PIL import Image

from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps.utils import (
    bisection_timestamp_search,
    filter_points_from_confidence,
)
from tqdm import tqdm


@dataclass
class AriaImageFrame:
    camera: calibration.CameraCalibration  # Camera calibration
    file_path: str  # file path
    image_size: np.array  # (H, W)
    t_world_camera: (
        sophus.SE3
    )  # The RGB camera to world transformation in SE3 (Default Center of Captured Frames)
    t_world_camera_read_start: (
        sophus.SE3
    )  # The RGB camera to world transformation at the start of rolling shutter capture
    t_world_camera_read_end: (
        sophus.SE3
    )  # The RGB camera to world transformation at the end of rolling shutter capture
    timestamp: float  # Timestamp (Center of Readout time)
    timestamp_read_start: float  # Timestamp (start of the readout time)
    timestamp_read_end: float  # Timestamp (end of the readout time)
    exposure_duration_s: float  # The exposure duration in second
    gain: float  # The (analog) gain from device


def to_aria_image_frame(
    provider,
    online_camera_calibs: List[mps.OnlineCalibration],
    closed_loop_traj: mps.ClosedLoopTrajectoryPose,
    out_dir: Path,
    camera_label: str = "camera-rgb",
    use_factory_calib: bool = False,
    visualize: bool = True,
):
    assert camera_label in ["camera-rgb", "camera-slam-left", "camera-slam-right"]

    def process_raw_data(frame_i: int, camera_label: str):
        stream_id = provider.get_stream_id_from_label(camera_label)

        sensor_data = provider.get_sensor_data_by_index(stream_id, frame_i)
        image_data, image_record = sensor_data.image_data_and_record()

        # https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/temporal_alignment_of_sensor_data#images-formation-temporal-model-rolling-shutter-and-pls-artifact
        if (
            image_data.get_height() == 2880
        ):  # RGB full resolution readout time is 16.26ms
            capture_time_offset = 8.13 * 1e6
        elif (
            image_data.get_height() == 1408
        ):  # RGB half resolution readout time is 5 ms
            capture_time_offset = 2.5 * 1e6
        elif (
            image_data.get_height() == 480
        ):  # slam camera is global shutter. readout is close to 0.
            capture_time_offset = 0
        else:
            raise RuntimeError(f"Unknown image data size! {image_data.get_height()}")

        # rgb camera timestamp during readout (start, middle, end)
        capture_time_start = image_record.capture_timestamp_ns
        capture_time_middle = capture_time_start + capture_time_offset
        capture_time_end = capture_time_start + 2 * capture_time_offset

        exposure_duration_s = image_record.exposure_duration
        gain = image_record.gain

        # replace this the interpolation function.
        pose_info = interpolate_aria_pose(closed_loop_traj, capture_time_middle)
        if pose_info is None:
            return None
        # pose_info = get_nearest_pose(closed_loop_traj, capture_time_middle)
        pose_read_start_info = interpolate_aria_pose(
            closed_loop_traj, capture_time_start
        )
        if pose_read_start_info is None:
            return None

        pose_read_end_info = interpolate_aria_pose(closed_loop_traj, capture_time_end)
        if pose_read_end_info is None:
            return None

        if pose_info.quality_score < 0.9:
            print(f"pose quality score below 1.0: {pose_info.quality_score}!")

        # Get the device calibration
        if use_factory_calib:
            device_calib = provider.get_device_calibration()
            camera_calib = device_calib.get_camera_calib(camera_label)
        else:
            nearest_calib_idx = bisection_timestamp_search(
                online_camera_calibs, capture_time_middle
            )
            if nearest_calib_idx is None:
                return None
            camera_calibration = online_camera_calibs[nearest_calib_idx]

            # find the one that is RGB camera
            camera_calib = None
            for calib in camera_calibration.camera_calibs:
                if calib.get_label() == camera_label:
                    if calib.get_image_size()[0] != image_data.get_height():
                        # the online calibration was not exporting the right image size. We will use the factory calibration instead.
                        device_calib = provider.get_device_calibration()
                        camera_calib = device_calib.get_camera_calib(camera_label)
                    else:
                        camera_calib = calib

                    break

        assert (
            camera_calib is not None
        ), f"Did not find {camera_label} calibration in online calibrations!"

        # Gaussian Splatting (COLMAP) use the same coordinate system as Aria
        # Therefore there is no coordinate system conversion needed.
        t_world_camera_read_center = (
            pose_info.transform_world_device
            @ camera_calib.get_transform_device_camera()
        )
        t_world_camera_read_start = (
            pose_read_start_info.transform_world_device
            @ camera_calib.get_transform_device_camera()
        )
        t_world_camera_read_end = (
            pose_read_end_info.transform_world_device
            @ camera_calib.get_transform_device_camera()
        )

        image = image_data.to_numpy_array()
        img_out_dir = out_dir / f"{camera_label}-images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        # always store the images using png file.
        image_rel_path = (
            f"{camera_label}-images/{camera_label}_{capture_time_middle}.jpg"
        )
        img_file_path = out_dir / image_rel_path

        if not img_file_path.exists():
            Image.fromarray(image).save(img_file_path)

        return AriaImageFrame(
            camera=camera_calib,
            file_path=image_rel_path,
            image_size=image.shape[:2],
            t_world_camera=t_world_camera_read_center,
            t_world_camera_read_start=t_world_camera_read_start,
            t_world_camera_read_end=t_world_camera_read_end,
            timestamp=capture_time_middle,
            timestamp_read_start=capture_time_start,
            timestamp_read_end=capture_time_end,
            exposure_duration_s=exposure_duration_s,
            gain=gain,
        )

    frames = []
    stream_id = provider.get_stream_id_from_label(camera_label)

    num_process = 1
    total_frames = provider.get_num_data(stream_id)
    for frame_i in tqdm(range(0, total_frames, num_process)):
        img_frame = process_raw_data(frame_i, camera_label=camera_label)
        if img_frame is None:
            continue
        frames.append(to_frame_json(img_frame, out_dir, visualize=visualize))

        # The VRS file reader is not thread-safe. Will cause some issues to run in parallel
        # num_process_to_launch = min(total_frames - frame_i, num_process)
        # with ThreadPoolExecutor(max_workers=num_process) as e:
        #     futures = [e.submit(process_raw_data, frame_i+i, camera_label) for i in range(num_process_to_launch)]
        #     results = [future.result() for future in futures if future.result() is not None]
        #     sorted_results = sorted(results, key=lambda x: x.timestamp)

        #     # if img_frame is None: continue
        #     # aria_image_frames.append(img_frame)
        #     for img_frame in sorted_results:
        #         frames.append(to_frame_json(img_frame, visualize=visualize))

    ns_frames = {
        "camera_model": "FISHEYE624",
        "camera_label": camera_label,
        "frames": frames,
    }
    print(f"{camera_label}: a total of {len(frames)} number of frames.")

    # with multiprocessing.Pool(24) as pool:
    #     pool_args = [(frame_i) for frame_i in range(0, provider.get_num_data(rgb_stream_id))]
    #     aria_image_frames = pool.starmap(process_raw_data, pool_args)
    # aria_image_frames = [x for x in aria_image_frames if x is not None]

    return ns_frames


def to_frame_json(
    frame: AriaImageFrame, out_dir: Path, visualize: bool = True, scale: float = 1000
):
    """
    transform Aria frames to a json file
    """

    fx, fy = frame.camera.get_focal_lengths()
    cx, cy = frame.camera.get_principal_point()
    h, w = (
        frame.image_size
    )  # the calibration image size might be incorrect due to API issue.

    camera_name = frame.camera.get_label()

    camera2device = frame.camera.get_transform_device_camera()

    if visualize:
        rr.log(
            "world/device",
            rr.Transform3D(
                translation=frame.t_world_camera.translation() * scale,
                mat3x3=frame.t_world_camera.rotation().to_matrix(),
            ),
        )
        # rotate image just for visualization
        image = np.array(Image.open(out_dir / frame.file_path).rotate(270))
        rr.log(
            f"world/device/{camera_name}/image",
            rr.Image(image).compress(jpeg_quality=75),
        )
        rr.log(
            f"world/device/{camera_name}",
            rr.Pinhole(resolution=[w, h], focal_length=[fx, fy]),
        )
        rr.log(
            f"world/device/{camera_name}_exposure_ms",
            rr.Scalars(frame.exposure_duration_s * 1000),
        )
        rr.log(f"world/device/{camera_name}_gain", rr.Scalars(frame.gain))

    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "distortion_params": frame.camera.get_projection_params()[3:15].tolist(),
        "w": int(w),
        "h": int(h),
        "file_path": frame.file_path,
        "camera_modality": "rgb",
        "camera2device": camera2device.to_matrix().tolist(),
        "transform_matrix": frame.t_world_camera.to_matrix().tolist(),  # This is supposed to be the center of readout time, used as default pose.
        "transform_matrix_read_start": frame.t_world_camera_read_start.to_matrix().tolist(),
        "transform_matrix_read_end": frame.t_world_camera_read_end.to_matrix().tolist(),
        "timestamp": frame.timestamp,  # this is the supposed to be the center of readout time
        "timestamp_read_start": frame.timestamp_read_start,
        "timestamp_read_end": frame.timestamp_read_end,
        "exposure_duration_s": frame.exposure_duration_s,
        "gain": frame.gain,
        "camera_name": frame.camera.get_label(),
    }


def visualize_frames(folder: Path, scale=100.0):

    rr.init("Visualize all frames rectified.", spawn=True)

    # VIsualize the transformed output path
    transform_output_path = folder / "transforms.json"
    semidense_points3d_path = folder / "semidense_points.csv.gz"

    semidense_points_data = mps.read_global_point_cloud(str(semidense_points3d_path))

    filtered_point_positions = []
    for point in semidense_points_data:
        if point.inverse_distance_std < 0.001 and point.distance_std < 0.15:
            filtered_point_positions.append(point.position_world * scale)
    rr.log(
        "world/points_3D",
        rr.Points3D(filtered_point_positions, colors=[200, 200, 200], radii=0.01),
    )

    with open(transform_output_path, "r") as f:
        ns_frames = json.load(f)
        frames = ns_frames["frames"]

    for _, frame in enumerate(frames):

        rr.set_time_seconds("sensor_time", frame.timestamp / 1e9)

        transform_matrix = np.array(frame["transform_matrix"])
        rr.log(
            "world/device",
            rr.Transform3D(
                translation=transform_matrix[:3, 3] * scale,
                mat3x3=transform_matrix[:3, :3],
            ),
        )

        image = np.array(Image.open(folder / frame["image_path"]))
        rr.log(
            "world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            "world/device/rgb",
            rr.Pinhole(
                resolution=[frame["w"], frame["h"]],
                focal_length=float(frame["fx"]),
            ),
        )

        if frame["mask_path"] != "":
            mask = np.array(Image.open(folder / frame["mask_path"]))
            rr.log("world/device/rgb/mask", rr.SegmentationImage(mask))

        rr.log(
            "world/device/exposure_ms", rr.Scalar(frame["exposure_duration_s"] * 1000)
        )
        rr.log("world/device/gain", rr.Scalar(frame["gain"]))


def run_single_sequence(
    recording_folder: str,
    vrs_file: Path,
    trajectory_file: Path,
    online_calibration_file: Path,
    semi_dense_points_file: Path,
    semi_dense_observation_file: Path,
    output_path: Path,
    rectified_rgb_focal: float,
    rectified_rgb_size: int,
    rectified_monochrome_focal: float,
    rectified_monochrome_height: int,
    options,  # all other options in configs
):
    if vrs_file == "":
        input_vrs = glob(str(recording_folder / "*.vrs"))
        assert len(input_vrs) == 1, "the target folder should only have 1 vrs file."
        input_vrs = Path(input_vrs[0])
        print(f"Find VRS file: {input_vrs}")
    else:
        input_vrs = recording_folder / vrs_file

    assert input_vrs.exists(), f"cannot find input vrs file {input_vrs}"

    if options.visualize:
        if options.aws_cluster:
            rr.init(
                f"Extract VRS file from {vrs_file}", recording_id="extract vrs file"
            )
            rr.serve(web_port=options.web_port, ws_port=options.ws_port)
        else:
            rr.init(f"Extract VRS file from {vrs_file}", spawn=True)

    print("Getting poses from closed loop trajectory CSV...")

    assert trajectory_file.exists(), f"cannot find trajectory file {trajectory_file}"
    closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_file))

    ##############################################################################
    # Process the raw semi-dense point cloud and show it as the base map
    ##############################################################################
    print("Get semi-dense point cloud")
    semidense_points_data = mps.read_global_point_cloud(str(semi_dense_points_file))
    inverse_distance_std_threshold = 0.005
    distance_std_threshold = 0.01
    filtered_semidense_points = filter_points_from_confidence(
        semidense_points_data, inverse_distance_std_threshold, distance_std_threshold
    )
    scale = 1000.0  # we will use this hard-coded parameter for all.
    point_positions = [it.position_world * scale for it in filtered_semidense_points]
    if options.visualize:
        rr.log(
            "world/points_3D",
            rr.Points3D(point_positions, colors=[200, 200, 200], radii=0.01),
        )
    semidense_map = {point.uid: point for point in semidense_points_data}

    vrs_provider = data_provider.create_vrs_data_provider(str(input_vrs))

    device_factory_calib = vrs_provider.get_device_calibration()

    # create an AriaImageFrame for each image in the VRS.
    camera_process_list = []
    camera_serial_map = {}
    if rectified_rgb_focal > 0:
        camera_process_list.append("camera-rgb")
        camera_serial_map["camera-rgb"] = device_factory_calib.get_camera_calib(
            "camera-rgb"
        ).get_serial_number()
    if rectified_monochrome_focal > 0:
        camera_process_list.append("camera-slam-left")
        camera_process_list.append("camera-slam-right")
        camera_serial_map["camera-slam-left"] = device_factory_calib.get_camera_calib(
            "camera-slam-left"
        ).get_serial_number()
        camera_serial_map["camera-slam-right"] = device_factory_calib.get_camera_calib(
            "camera-slam-right"
        ).get_serial_number()

    assert (
        online_calibration_file.exists()
    ), f"cannot find online calibration file {online_calibration_file}"
    camera_calibs = mps.read_online_calibration(str(online_calibration_file))

    ######################################################
    # Process a transform json file for raw camera stream
    ######################################################
    for camera_label in camera_process_list:
        print(f"Creating Aria frames for {camera_label}")

        if options.use_factory_calib:
            json_path = output_path / f"{camera_label}-transforms-factory-calib.json"
        else:
            json_path = output_path / f"{camera_label}-transforms.json"

        if json_path.exists() and not options.overwrite:
            print(f"{json_path} existed. Skip preprocessing.")
            continue

        ns_frames = to_aria_image_frame(
            provider=vrs_provider,
            online_camera_calibs=camera_calibs,
            closed_loop_traj=closed_loop_traj,
            out_dir=output_path,
            camera_label=camera_label,
            visualize=options.visualize,
            use_factory_calib=options.use_factory_calib,
        )

        ns_frames["camera_label"] = camera_label
        ns_frames["transform_cpf"] = (
            device_factory_calib.get_transform_cpf_sensor(camera_label)
            .to_matrix()
            .tolist()
        )

        print(f"Write camera information for {camera_label} to {json_path}")
        with open(json_path, "w", encoding="UTF-8") as file:
            json.dump(ns_frames, file, indent=4)

    ########################################################################################
    # Rectify images &
    # Generate the per-frame depth map information
    #########################################################################################

    # We will read them directly csv file which makes things easier
    # semidense_observations = mps.read_point_observations(str(semi_dense_observation_file))
    if semi_dense_observation_file is not None:
        df_semidense_observations = pandas.read_csv(str(semi_dense_observation_file))
    else:
        df_semidense_observations = None

    for camera_label in camera_process_list:
        print(f"Creating rectified frames for {camera_label}")

        if camera_label == "camera-rgb":
            rectified_image_folder = f"{camera_label}-rectified-{int(rectified_rgb_focal)}-h{rectified_rgb_size}"
        else:
            rectified_image_folder = f"{camera_label}-rectified-{int(rectified_monochrome_focal)}-h{rectified_monochrome_height}"

        if options.extract_fisheye:
            rectified_image_folder = str(rectified_image_folder) + "-spherical"
            rectified_camera_model = "spherical"
            parallel_process = False  # somehow the spherical calibration will throw segfault when parallel process
        else:
            rectified_camera_model = "linear"
            parallel_process = True

        rectified_image_folder = output_path / rectified_image_folder

        if options.use_factory_calib:
            input_json_path = (
                output_path / f"{camera_label}-transforms-factory-calib.json"
            )
            rectified_image_folder = Path(
                str(rectified_image_folder) + "-factory-calib"
            )
        else:
            input_json_path = output_path / f"{camera_label}-transforms.json"

        rectified_image_folder.mkdir(exist_ok=True)
        transform_json_path = rectified_image_folder / "transforms.json"

        # set up a symbolic link for the semi-dense point cloud
        semidense_points_path_in_rectified = (
            rectified_image_folder / "semidense_points.csv.gz"
        )
        print(
            f"add symbolic link of point cloud for {semidense_points_path_in_rectified}"
        )
        if not semidense_points_path_in_rectified.is_symlink():
            semidense_points_path_in_rectified.symlink_to(
                semi_dense_points_file.absolute()
            )

        # set up a symbolic link for the closed-loop trajectory
        closed_loop_traj_path_in_rectified = (
            rectified_image_folder / "closed_loop_trajectory.csv"
        )
        if not closed_loop_traj_path_in_rectified.is_symlink():
            closed_loop_traj_path_in_rectified.symlink_to(trajectory_file.absolute())

        frames = read_frames_from_metadata(transforms_json=input_json_path)

        # Read the vignette image, rectify and save the vignette image.
        if camera_label == "camera-rgb":
            vignette = get_rectified_vignette_image(
                frame=frames[0],
                input_root=Path("data"),
                camera_label=camera_label,
                camera_model=rectified_camera_model,
                output_focal=rectified_rgb_focal,
                output_h=rectified_rgb_size,
                output_w=rectified_rgb_size,
            )
            Image.fromarray(vignette).save(rectified_image_folder / "vignette.png")
            mask = get_rectified_mask(
                frame=frames[0],
                input_root=Path("data"),
                camera_model=rectified_camera_model,
                output_focal=rectified_rgb_focal,
                output_height=rectified_rgb_size,
            )
            Image.fromarray(mask).save(rectified_image_folder / "mask.png")

            image_index = get_rectified_row_index(
                frame=frames[0],
                camera_model=rectified_camera_model,
                output_focal=rectified_rgb_focal,
                output_h=rectified_rgb_size,
                output_w=rectified_rgb_size,
            )
            Image.fromarray(image_index).save(rectified_image_folder / "image_index.png")

            rec_focal = rectified_rgb_focal
            rec_height = rectified_rgb_size
        else:
            vignette = get_rectified_vignette_image(
                frame=frames[0],
                input_root=Path("data"),
                camera_label=camera_label,
                camera_model=rectified_camera_model,
                output_focal=rectified_monochrome_focal,
                output_h=rectified_monochrome_height,
            )
            Image.fromarray(vignette).save(rectified_image_folder / "vignette.png")

            rec_focal = rectified_monochrome_focal
            rec_height = rectified_monochrome_height

            mask = get_rectified_mask(
                frame=frames[0],
                camera_model=rectified_camera_model,
                output_focal=rectified_monochrome_focal,
                output_height=rectified_monochrome_height,
            )
            Image.fromarray(mask).save(rectified_image_folder / "mask.png")

        if transform_json_path.exists() and not options.overwrite:
            print(f"{transform_json_path} existed. Skip rectification.")
            continue

        rectified_frames = []
        num_process = 24
        if parallel_process:
            print("Parallel process all the frames")
            for frame_i in tqdm(range(0, len(frames), num_process)):
                num_process_to_launch = min(len(frames) - frame_i, num_process)
                with ThreadPoolExecutor(max_workers=num_process) as e:
                    futures = [
                        e.submit(
                            process_frame,
                            frames[frame_i + i],
                            output_path,
                            rectified_image_folder,
                            rectified_camera_model,
                            rec_focal,
                            rec_height,
                        )
                        for i in range(num_process_to_launch)
                    ]
                    results = [future.result() for future in futures]
                    sorted_results = sorted(results, key=lambda x: x["timestamp"])
                    rectified_frames += sorted_results
        else:
            for frame in tqdm(frames):
                rec_frame = process_frame(
                    frame,
                    output_path,
                    rectified_image_folder,
                    rectified_camera_model,
                    rec_focal,
                    rec_height,
                )
                rectified_frames.append(rec_frame)

        # Generate sparse depth map for SLAM camera stream using semi-dense point cloud and tracker information
        if (
            camera_label.startswith("camera-slam")
            and df_semidense_observations is not None
            and rectified_camera_model == "linear"
        ):
            device_serial_num = camera_serial_map[camera_label]
            df_cam_observations2d = df_semidense_observations[
                df_semidense_observations["camera_serial"] == device_serial_num
            ]

            print(
                f"==> There are a total of {len(df_cam_observations2d)} 2d tracked points"
            )

            rectified_frames = create_visible_depth_map(
                df_observations2d=df_cam_observations2d,
                points3d=semidense_map,
                frames=rectified_frames,
                camera_label=camera_label,
                image_folder=rectified_image_folder,
                options=options,
            )

        with open(transform_json_path, "w") as f:
            json.dump(
                {
                    "camera_model": rectified_camera_model,
                    "camera_label": camera_label,
                    "transform_cpf": device_factory_calib.get_transform_cpf_sensor(
                        camera_label
                    )
                    .to_matrix()
                    .tolist(),
                    "frames": rectified_frames,
                },
                f,
                indent=4,
            )

    #######################################################################################
    # Create sparse depth for the image
    # For slam camera view, we use the points observed in the corresponding image
    # For RGB camera view, we use the points in nearest SLAM camera view that is also within the frustum of RGB image.
    slam_transform_json_path = {}
    for camera_label in camera_process_list:
        if camera_label.startswith("camera-slam"):

            rectified_image_folder = (
                output_path
                / f"{camera_label}-rectified-{int(rectified_monochrome_focal)}-h{rectified_monochrome_height}"
            )
            transform_json_with_depth_path = (
                rectified_image_folder / "transforms_with_sparse_depth.json"
            )

            slam_transform_json_path[camera_label] = transform_json_with_depth_path
            if transform_json_with_depth_path.exists() and not options.overwrite:
                print(f"{transform_json_with_depth_path} has been generated. Skip!")
                continue
            elif df_semidense_observations is None:
                print("There are no semi-dense observation file. Skip generating it.")
                continue

            transform_json_path = rectified_image_folder / "transforms.json"

            device_serial_num = camera_serial_map[camera_label]
            df_cam_observations2d = df_semidense_observations[
                df_semidense_observations["camera_serial"] == device_serial_num
            ]

            print(
                f"==> There are a total of {len(df_cam_observations2d)} 2d tracked points"
            )

            with open(transform_json_path, "r") as f:
                camera_info = json.load(f)
                frames = camera_info["frames"]

            frames_with_depth = create_visible_depth_map(
                df_observations2d=df_cam_observations2d,
                points3d=semidense_map,
                frames=frames,
                camera_label=camera_label,
                image_folder=rectified_image_folder,
                options=options,
            )

            with open(transform_json_with_depth_path, "w") as f:
                camera_info["frames"] = frames_with_depth
                json.dump(camera_info, f, indent=4)

    # Find the points in nearest SLAM camera view and filter those in frustum
    rgb_rectified_image_folder = (
        output_path
        / f"camera-rgb-rectified-{int(rectified_rgb_focal)}-h{rectified_rgb_size}"
    )
    if options.use_factory_calib:
        rgb_rectified_image_folder = Path(
            str(rgb_rectified_image_folder) + "-factory-calib"
        )

    rgb_transform_json_path = rgb_rectified_image_folder / "transforms.json"
    rgb_transform_json_with_depth_path = (
        rgb_rectified_image_folder / "transforms_with_sparse_depth.json"
    )
    if rgb_transform_json_with_depth_path.exists() and not options.overwrite:
        print(f"{rgb_transform_json_with_depth_path} has been generated. Skip!")
    elif df_semidense_observations is None:
        print("There are no semi-dense observation file. Skip generating it.")
    else:
        frames_with_depth = fetch_visible_depth_map_for_RGB(
            rgb_transform_json_path,
            slam_transform_json_path["camera-slam-left"],
            slam_transform_json_path["camera-slam-right"],
            rgb_rectified_image_folder,
            options,
        )

        with open(rgb_transform_json_with_depth_path, "w") as f:
            json.dump(frames_with_depth, f, indent=4)


def fetch_visible_depth_map_for_RGB(
    rgb_transform_json_path,
    slam_left_transform_json_path,
    slam_right_transform_json_path,
    image_folder,
    options,
):
    with open(rgb_transform_json_path, "r") as f:
        rgb_camera_info = json.load(f)

    with open(slam_left_transform_json_path, "r") as f:
        slam_left_camera_info = json.load(f)

    with open(slam_right_transform_json_path, "r") as f:
        slam_right_camera_info = json.load(f)

    slam_left_timestamps = [
        frame["timestamp"] for frame in slam_left_camera_info["frames"]
    ]
    slam_left_timestamps = np.asarray(slam_left_timestamps)

    slam_right_timestamps = [
        frame["timestamp"] for frame in slam_right_camera_info["frames"]
    ]
    slam_right_timestamps = np.asarray(slam_right_timestamps)

    sparse_depth_folder = image_folder / "sparse_depth"
    sparse_depth_folder.mkdir(exist_ok=True)

    # find the nearest neighbor in the slam camera info
    for frame in rgb_camera_info["frames"]:
        rgb_timestamp = frame["timestamp"]

        rgb_c2w = np.asarray(frame["transform_matrix"])
        rgb_w2c = np.linalg.inv(rgb_c2w)
        rgb_calibK = np.array(
            [[frame["fx"], 0, frame["cx"]], [0, frame["fy"], frame["cy"]], [0, 0, 1]]
        )

        # could be more precise checking closest time within the two, but may not be necessary here
        slam_left_index = np.searchsorted(slam_left_timestamps, rgb_timestamp)
        # slam_left_timestamp = slam_left_timestamps[slam_left_index]
        if slam_left_index < len(slam_left_camera_info["frames"]):
            slam_left_frame = slam_left_camera_info["frames"][slam_left_index]
        else:
            slam_left_frame = slam_left_camera_info["frames"][-1]
        with open(
            slam_left_transform_json_path.parent / slam_left_frame["sparse_depth"], "r"
        ) as f:
            slam_left_sparse_depth = json.load(f)

            u = np.asarray(slam_left_sparse_depth["u"])
            v = np.asarray(slam_left_sparse_depth["v"])
            z = np.asarray(slam_left_sparse_depth["z"])
            inverse_distance_std = np.asarray(
                slam_left_sparse_depth["inverseDistanceStd"]
            )
            distance_std = np.asarray(slam_left_sparse_depth["distanceStd"])

            if len(u) > 0:
                fx = slam_left_frame["fx"]
                fy = slam_left_frame["fy"]
                cx = slam_left_frame["cx"]
                cy = slam_left_frame["cy"]

                x = z * (u - cx) / fx
                y = z * (v - cy) / fy
                z = z

                pt3d_slamleft = np.stack([x, y, z])
                slam_left_c2w = np.asarray(slam_left_frame["transform_matrix"])
                pt3d_left_world = (
                    slam_left_c2w[:3, :3] @ pt3d_slamleft + slam_left_c2w[:3, 3:]
                )

                u_from_slam_left, v_from_slam_left, z_from_slam_left, mask = project(
                    pt3d_left_world, rgb_w2c, rgb_calibK, frame["h"], frame["w"]
                )
                inv_dist_std_from_slam_left = inverse_distance_std[mask]
                dist_std_from_slam_left = distance_std[mask]
            else:
                u_from_slam_left = np.asarray([])
                v_from_slam_left = np.asarray([])
                z_from_slam_left = np.asarray([])
                inv_dist_std_from_slam_left = np.asarray([])
                dist_std_from_slam_left = np.asarray([])

        slam_right_index = np.searchsorted(slam_right_timestamps, rgb_timestamp)
        if slam_right_index < len(slam_right_camera_info["frames"]):
            slam_right_frame = slam_right_camera_info["frames"][slam_right_index]
        else:
            slam_right_frame = slam_right_camera_info["frames"][-1]
        with open(
            slam_right_transform_json_path.parent / slam_right_frame["sparse_depth"],
            "r",
        ) as f:
            slam_right_sparse_depth = json.load(f)

            u = np.asarray(slam_right_sparse_depth["u"])
            v = np.asarray(slam_right_sparse_depth["v"])
            z = np.asarray(slam_right_sparse_depth["z"])
            inverse_distance_std = np.asarray(
                slam_right_sparse_depth["inverseDistanceStd"]
            )
            distance_std = np.asarray(slam_right_sparse_depth["distanceStd"])

            if len(u) > 0:  # no visible point.
                fx = slam_right_frame["fx"]
                fy = slam_right_frame["fy"]
                cx = slam_right_frame["cx"]
                cy = slam_right_frame["cy"]

                x = z * (u - cx) / fx
                y = z * (v - cy) / fy
                z = z

                pt3d_slamright = np.stack([x, y, z])
                slam_right_c2w = np.asarray(slam_right_frame["transform_matrix"])
                pt3d_right_world = (
                    slam_right_c2w[:3, :3] @ pt3d_slamright + slam_right_c2w[:3, 3:]
                )

                u_from_slam_right, v_from_slam_right, z_from_slam_right, mask = project(
                    pt3d_right_world, rgb_w2c, rgb_calibK, frame["h"], frame["w"]
                )
                inv_dist_std_from_slam_right = inverse_distance_std[mask]
                dist_std_from_slam_right = distance_std[mask]
            else:
                u_from_slam_right = np.asarray([])
                v_from_slam_right = np.asarray([])
                z_from_slam_right = np.asarray([])
                inv_dist_std_from_slam_right = np.asarray([])
                dist_std_from_slam_right = np.asarray([])

        u_rgb = np.concatenate([u_from_slam_left, u_from_slam_right], axis=0)
        v_rgb = np.concatenate([v_from_slam_left, v_from_slam_right], axis=0)
        z_rgb = np.concatenate([z_from_slam_left, z_from_slam_right], axis=0)
        inv_dist_rgb = np.concatenate(
            [inv_dist_std_from_slam_left, inv_dist_std_from_slam_right], axis=0
        )
        dist_std_rgb = np.concatenate(
            [dist_std_from_slam_left, dist_std_from_slam_right], axis=0
        )

        if options.visualize:
            image = np.array(Image.open(image_folder / frame["image_path"]))
            rr.log(
                f"camera-rgb/image",
                rr.Image(image).compress(jpeg_quality=70),
            )

            points2d = np.stack([u_rgb, v_rgb], axis=-1)
            rr.log(
                f"camera-rgb/image/points_2D",
                rr.Points2D(points2d, colors=[0, 200, 0], radii=2),
            )

        # save it as json file as well
        depth_filename = f"sparse_depth/camera-rgb_{frame['timestamp']}.json"
        with open(image_folder / depth_filename, "w") as f:
            frame_pts3d = {
                "u": u_rgb.tolist(),
                "v": v_rgb.tolist(),
                "z": z_rgb.tolist(),
                "inverseDistanceStd": inv_dist_rgb.tolist(),
                "distanceStd": dist_std_rgb.tolist(),
            }
            json.dump(frame_pts3d, f, indent=4)

        frame["sparse_depth"] = depth_filename

    return rgb_camera_info


def create_visible_depth_map(
    df_observations2d,
    points3d,
    frames,
    camera_label,
    image_folder,
    options,
):
    if options.visualize:
        if options.aws_cluster:
            rr.init(
                "Visualize the visible depth map for each slam frame",
                recording_id="SLAM frame visible depth map",
            )
            rr.serve(web_port=options.web_port, ws_port=options.ws_port)
        else:
            rr.init("Visualize the visible depth map for each slam frame", spawn=True)

    sparse_depth_folder = image_folder / "sparse_depth"
    sparse_depth_folder.mkdir(exist_ok=True)

    for frame in frames:

        frame_data2d = df_observations2d[
            df_observations2d["frame_tracking_timestamp_us"]
            == int(frame["timestamp"] / 1e3)
        ]

        # get the corresponding point cloud in a device local frame
        uids = frame_data2d["uid"].tolist()

        c2w = np.array(frame["transform_matrix"])
        w2c = np.linalg.inv(c2w)
        calibK = np.array(
            [[frame["fx"], 0, frame["cx"]], [0, frame["fy"], frame["cy"]], [0, 0, 1]]
        )
        frame_h = frame["h"]
        frame_w = frame["w"]

        frame_pts3d = {
            "u": [],
            "v": [],
            "z": [],
            "inverseDistanceStd": [],
            "distanceStd": [],
            "uid": [],
        }
        for uid in uids:
            pt3d = points3d[uid].position_world

            u, v, z = project(pt3d[:, None], w2c, calibK, frame_h, frame_w)

            if u is not None:
                frame_pts3d["u"].append(u[0])
                frame_pts3d["v"].append(v[0])
                frame_pts3d["z"].append(z[0])
                frame_pts3d["inverseDistanceStd"].append(
                    points3d[uid].inverse_distance_std
                )
                frame_pts3d["distanceStd"].append(points3d[uid].distance_std)

        if options.visualize:
            image = np.array(Image.open(image_folder / frame["image_path"]))
            rr.log(
                f"{camera_label}/image",
                rr.Image(image).compress(jpeg_quality=100),
            )

            points2d = np.stack(
                [np.array(frame_pts3d["u"]), np.array(frame_pts3d["v"])], axis=-1
            )
            rr.log(
                f"{camera_label}/image/points_2D",
                rr.Points2D(points2d, colors=[0, 200, 0], radii=2),
            )

        # save the sparse points as json file
        depth_filename = f"sparse_depth/{camera_label}_{frame['timestamp']}.json"
        with open(image_folder / depth_filename, "w") as f:
            json.dump(frame_pts3d, f, indent=4)

        frame["sparse_depth"] = depth_filename

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Aria VRS data with MPS output to a json file format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input files
    parser.add_argument(
        "--input_root",
        help="input root folder which contain all gaia recordings folders",
        type=Path,
        default="",
    )
    parser.add_argument(
        "--vrs_file", type=Path, default="", help="The VRS file to be executed"
    )
    parser.add_argument(
        "--trajectory_file",
        help="path to timestamped world-to-device trajectory",
        type=str,
        default="",
    )
    parser.add_argument(
        "--online_calib_file",
        help="path to an online calibration file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--semi_dense_points_file",
        help="path to the semi-dense point cloud file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--semi_dense_observation_file",
        help="path to the semi-dense observation file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_root",
        help="the output files folder. Will use the same input folder if empty",
        type=Path,
        default="",
    )
    parser.add_argument(
        "--json_path", help="output json path", type=str, default="transforms.json"
    )
    parser.add_argument(
        "--image_path", help="output images path", type=str, default="images"
    )
    parser.add_argument(
        "--rectified_folder",
        help="The rectified images folder",
        type=str,
        default="rectified",
    )
    parser.add_argument(
        "--rectified_rgb_focal",
        help="The rectified RGB image focal length. If set to <0, it will skip rectifying RGB images.\
            For a 2880x2880 image, 1200 can be a default choice close to its original focal length. \
            For a 1408x1408 image, 600 can be a default choice close to its original focal length.",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--rectified_rgb_size",
        help="The rectified RGB image size. It is square image of size^2",
        type=int,
        default=2880,
    )
    parser.add_argument(
        "--rectified_monochrome_focal",
        help="The rectified SLAM image focal length. If set to <0, it will skip rectifying SLAM images.",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--rectified_monochrome_height",
        help="The rectified monochrome image size in height. The width will be adjusted accordingly.",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the extracted intermediate files if they have been generated.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the processed files"
    )
    parser.add_argument(
        "--use_factory_calib", action="store_true", help="Use factory calibration"
    )
    parser.add_argument(
        "--extract_fisheye",
        action="store_true",
        help="Extract fisheye of camera streams",
    )
    parser.add_argument(
        "--aws_cluster", action="store_true", help="run on aws cluster if True."
    )
    parser.add_argument(
        "--web_port",
        type=int,
        default=8080,
        help="aws cluster web port if used. (for aws server only)",
    )
    parser.add_argument(
        "--ws_port",
        type=int,
        default=9877,
        help="aws cluster ws port if used. (for aws server only)",
    )

    args = parser.parse_args()

    print("Extract files from a single VRS recording and its MPS")
    recording_folder = args.input_root

    name = str(args.vrs_file)[:-4]

    if args.output_root == Path(""):
        output_folder = recording_folder / name
    else:
        output_folder = args.output_root / name

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Ouput will be saved to {output_folder}")

    trajectory_file = Path(args.trajectory_file)
    assert (
        trajectory_file.exists()
    ), f"Trajectory is required. Provided trajectory path does not exists! {trajectory_file}"
    print(f"Will load trajectory file: {trajectory_file}")

    online_calib_file = Path(args.online_calib_file)
    assert (
        online_calib_file.exists()
    ), f"Online calibration file is required. Provided online calibration file does not exists! {online_calib_file}"
    print(f"Will load calibration file: {online_calib_file}")

    semidense_points_path = Path(args.semi_dense_points_file)
    print(f"Will load semi-dense point cloud file: {semidense_points_path}")
    assert (
        semidense_points_path.exists()
    ), f"Semi-dense point cloud file is required. Provided semi-dense point file does not exists! {semidense_points_path}"

    semidense_observation_path = Path(args.semi_dense_observation_file)
    if semidense_observation_path.exists() and args.semi_dense_observation_file != "":
        # assert semidense_observation_path.suffix == ".gz", f"provided semi dense observation file {semidense_observation_path} is not .gz format!"
        print(f"Will load semi-dense observation file: {semidense_observation_path}")
    else:
        print(
            f"No valid semi-dense observation file is provided. Will skip load {semidense_observation_path}"
        )
        semidense_observation_path = None

    run_single_sequence(
        recording_folder=recording_folder,
        vrs_file=args.vrs_file,
        trajectory_file=trajectory_file,
        online_calibration_file=online_calib_file,
        semi_dense_points_file=semidense_points_path,
        semi_dense_observation_file=semidense_observation_path,
        output_path=output_folder,
        rectified_rgb_focal=args.rectified_rgb_focal,
        rectified_rgb_size=args.rectified_rgb_size,
        rectified_monochrome_focal=args.rectified_monochrome_focal,
        rectified_monochrome_height=args.rectified_monochrome_height,
        options=args,
    )


if __name__ == "__main__":
    main()
