"""
Script for converting images to videos
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np  # noqa: F401

from ...easyvolcap.utils.console_utils import (
    blue,
    build_parser,
    catch_throw,
    dirname,
    dotdict,
    exists,
    join,
    json,
    log,
    red,
    relpath,
    split,
    # std_tqdm as tqdm,
    tqdm,
    yellow,
)
from ...easyvolcap.utils.data_utils import generate_video
from ...easyvolcap.utils.parallel_utils import parallel_execution
from ...misc.io_helper import pathmgr


ffmpeg_bin = "ffmpeg"


def node(
    images_dirs: List[str] = [],  # noqa: B006
    output_files: List[str] = [],
    ffmpeg_args: Dict[str, Union[int, str, bool]] = dict(  # noqa: B006, C408
        fps=60,
        crf=23,
        preset="veryfast",
        vcodec="libopenh264",  # Use libopenh264 which is available in ffmpeg-free
        verbose=True,
    ),
    ext: str = ".png",
    sequential: bool = False,
    skip_existing: bool = True,
    trust_local_images_sorting: bool = False,
    **kwargs,
):
    """
    On one node, launch a bunch of workers to process images
    """
    parallel_execution(
        images_dirs,
        output_files,
        action=worker,
        ffmpeg_args=ffmpeg_args,
        skip_existing=skip_existing,
        ext=ext,
        # print_progress=True,
        use_process=True,
        sequential=sequential,
        trust_local_images_sorting=trust_local_images_sorting,
    )


def worker(
    images_dir: str = "",
    output_file: str = "",
    ffmpeg_args: Dict = {},  # noqa: B006, C408
    ext: str = ".png",
    skip_existing: bool = True,
    trust_local_images_sorting: bool = False,
    **kwargs,
) -> None:
    """
    Download all the images in this folder, concatenate them into a video, and upload the video.
    Since we're only able to use ffmpeg on a command line efficiently here.
    """


    if trust_local_images_sorting:
        local_images_dir = pathmgr.get_local_path(
            images_dir, recursive=True
        )  # will download all images to the current node

        local_output_file = split(local_images_dir)[0] + "/" + split(output_file)[-1]
        generate_video(
            f"{local_images_dir}/*{ext}",
            local_output_file,
            ffmpeg=ffmpeg_bin,
            **ffmpeg_args,
        )

        pathmgr.copy_from_local(local_output_file, output_file, overwrite=True)
        # pathmgr.rm(local_output_file)
        return

    output_json = output_file.replace("videos.mp4", "transforms.json")
    with pathmgr.open(output_json, "r") as f:
        data = json.load(f)
        frames = data["frames"] if isinstance(data, dict) else data
        names = [d["image_path"] for d in frames]

    should_regenerate = False
    # lens = [len(name) for name in names]
    # lens = np.asarray(lens)

    # # Find the timestamp of the misalignment, the length of the image name changes
    # # Assume only changing once and only upping one digit
    # # Find the first instance where the length changes
    # i = len(lens) - 1
    # for i in range(len(lens) - 1):
    #     if lens[i] != lens[i + 1]:
    #         break
    # should_regenerate = i != len(lens) - 1
    # if should_regenerate:
    #     log(
    #         yellow(
    #             f"Found the timestamp of the misalignment at {i} for {blue(output_file)}, the length of the image name changes, will have to regenerate video for this particular sequence using the new method"
    #         )
    #     )
    # # first_segment_last_index = len(lens) - i - 1 - 1

    if pathmgr.isfile(output_file) and not should_regenerate and skip_existing:
        log(f"{blue(output_file)} already exists, skipping processing")
        return

    local_images_dir = pathmgr.get_local_path(
        images_dir, recursive=True
    )  # will download all images to the current node
    local_output_file = split(local_images_dir)[0] + "/" + split(output_file)[-1]

    # Sort the files by the actual timestamp
    # files = pathmgr.ls(local_images_dir)
    # files = [f for f in files if f.endswith(ext)]
    # files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[-1]))
    # Find the name of the image sorted by the transforms.json file
    files = [split(n)[-1] for n in names]

    # Create a symlink to the images in the local folder
    # This is to avoid downloading the images again and again
    # The symlink will be removed after the video is generated

    local_temp_dir = join(dirname(local_images_dir), "temp")
    pathmgr.mkdirs(local_temp_dir, exist_ok=True)
    for idx, file in enumerate(tqdm(files)):
        local_temp_file = join(local_temp_dir, f"{idx:08d}{ext}")
        src = join(local_images_dir, file)
        tar = local_temp_file
        if exists(tar):
            pathmgr.rm(tar)
        pathmgr.symlink(relpath(src, dirname(tar)), tar)

    generate_video(
        f"{local_temp_dir}/*{ext}",
        local_output_file,
        ffmpeg=ffmpeg_bin,
        **ffmpeg_args,
    )

    pathmgr.copy_from_local(local_output_file, output_file, overwrite=True)
    pathmgr.rm(local_temp_dir)
    # pathmgr.rm(local_output_file)


def launcher(
    data_root: str = "data/aea",
    images_dir: str = "images",
    video_file: str = "videos.mp4",
    camera_dirs: List[str] = [  # noqa: B006
        "camera-rgb-rectified-600-h1000-factory-calib-color_transformed",
        "camera-slam-left-rectified-180-h480-factory-calib",
        "camera-slam-right-rectified-180-h480-factory-calib",
    ],
    ok_list: str = "ok_list.txt",
    bad_list: str = "bad_list.txt",
    empty_list: str = "empty_list.txt",
    path_sample: List[Optional[int]] = [0, None, 1],  # noqa: B006
    num_jobs: int = 1000,
    ffmpeg_args: Dict[str, Union[int, str, bool]] = dict(  # noqa: B006, C408
        fps=20,
        crf=23,
        preset="veryfast",
        vcodec="libopenh264",  # Use libopenh264 which is available in ffmpeg-free
        verbose=True,
    ),
    ext: str = ".png",
    find_subseqs: bool = True,
    skip_existing: bool = True,
    trust_local_images_sorting: bool = False,
    **kwargs,
) -> Dict:
    """
    Process Aria sequences from egoexo datasets, store the processed images as videos.
    """

    ok_list = join(data_root, ok_list)
    bad_list = join(data_root, bad_list)
    empty_list = join(data_root, empty_list)

    # Scan the dataset directory for existing images folder for further processing
    if pathmgr.isfile(ok_list):
        log(f"{blue(ok_list)} already exists, skipping scanning")
        with pathmgr.open(ok_list, "r") as f:
            paths = f.read().splitlines()
    else:
        log(yellow(f'Scanning direcotry "{blue(data_root)}"...'))

        paths = []
        bads = []
        emptys = []
        seqs = sorted(pathmgr.ls(data_root))
        total = len(seqs)
        for i, seq in tqdm(enumerate(seqs), total=total):
            if find_subseqs:
                subseqs = sorted(pathmgr.ls(join(data_root, seq)))
            else:
                subseqs = ["."]
            for subseq in subseqs:
                for camera_dir in camera_dirs:
                    images = join(
                        data_root,
                        seq,
                        subseq,
                        camera_dir,
                        images_dir,
                    )
                    if pathmgr.exists(images):  # could be a symlink
                        paths.append(images)
                    else:
                        log(
                            f"{red(seq)} doesn't contain an images folder in {red(subseq)} at {red(camera_dir)}"
                        )
                        bads.append(images)

            if not len(subseqs):
                log(f"{red(seq)} does not contain any subfolders")
                emptys.append(seq)

            if i % math.ceil(total / 20) == 0 or i == total - 1:
                log(f"Scanned {i+1}/{total} ({(i+1)/total*100:.2f}%) sequences...")

        log(
            yellow(
                f"Cachine scanning results to {blue(ok_list)}, {blue(bad_list)} and {blue(empty_list)}"
            )
        )
        with pathmgr.open(ok_list, "w") as f:
            f.write("\n".join(paths))
        with pathmgr.open(bad_list, "w") as f:
            f.write("\n".join(bads))
        with pathmgr.open(empty_list, "w") as f:
            f.write("\n".join(emptys))

    # Given the number of jobs to launch, distribute the workload across nodes
    b, e, s = path_sample
    paths = paths[b:e:s]
    log(f"Found {len(paths)} paths for processing")
    job_per_node = math.ceil(len(paths) / num_jobs)
    log(f"Each node will process {job_per_node} paths")
    split_paths = [
        paths[i * job_per_node : (i + 1) * job_per_node] for i in range(num_jobs)
    ]

    for paths in split_paths:
        # log(f"Launching node {i} with {len(paths)} paths")
        node(
            images_dirs=paths,
            output_files=[p.replace(images_dir, video_file) for p in paths],
            ffmpeg_args=ffmpeg_args,
            ext=ext,
            skip_existing=skip_existing,
            trust_local_images_sorting=trust_local_images_sorting,
        )

    return {}


@catch_throw
def main():
    args = dotdict()
    args.data_root = "data/adt"
    args.camera_dirs = dotdict(
        default=(".",),
        nargs="+",
        type=str,
    )
    args.num_jobs = 1
    args.sequential = True
    args.skip_existing = True
    args.find_subseqs = False
    args.fps = 30
    args.ext = ".png"
    args.trust_local_images_sorting = True
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    ffmpeg_args = dict(  # noqa: B006, C408
        fps=args.fps,
        crf=23,
        preset="veryfast",
        vcodec="libopenh264",  # Use libopenh264 which is available in ffmpeg-free
        verbose=True,
    )

    launcher(
        data_root=args.data_root,
        camera_dirs=args.camera_dirs,
        num_jobs=args.num_jobs,
        sequential=args.sequential,
        skip_existing=args.skip_existing,
        find_subseqs=args.find_subseqs,
        ffmpeg_args=ffmpeg_args,
        ext=args.ext,
        trust_local_images_sorting=args.trust_local_images_sorting,
    )


if __name__ == "__main__":
    main()
