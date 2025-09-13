# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Collect the duration information for all the datasets present in the folder
"""

import os

from easyvolcap.utils.console_utils import (
    build_parser,
    catch_throw,
    dirname,
    dotdict,
    exists,
    isdir,
    join,
    json,
    log,
    tqdm,
    yellow,
)
from easyvolcap.utils.data_utils import get_video_length


@catch_throw
def main():
    args = dotdict(
        data_root="/source/dendenxu/datasets",
        ignore=dotdict(
            default=["Panda-70M", "backlog", "DAVIS-final", "dynpose-100k"],
            type=str,
            nargs="*",
        ),
        only=dotdict(default=["ritw", "tum"], type=str, nargs="*"),
        max_search_level=3,  # search into the folder for the videos.mp4 file for calculating the duration
        fps=30,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    datasets = sorted(os.listdir(args.data_root))
    for dataset in datasets:
        if not isdir(join(args.data_root, dataset)):
            continue
        if dataset in args.ignore:
            continue
        if len(args.only) > 0 and dataset not in args.only:
            continue
        durations = dotdict()
        seqs = sorted(os.listdir(join(args.data_root, dataset)))
        for seq in tqdm(seqs, desc=f"Processing {dataset}"):
            if not isdir(join(args.data_root, dataset, seq)):
                continue
            vid = join(args.data_root, dataset, seq, "videos.mp4")
            for _ in range(args.max_search_level):
                if not exists(vid):
                    dirs = os.listdir(dirname(vid))
                    if len(dirs):
                        first_dir = sorted(dirs)[0]
                        vid = join(dirname(vid), first_dir, "videos.mp4")
                else:
                    break
            if not exists(vid):
                log(yellow(f"Video not found for {dataset} {seq}, skipping..."))
                continue
            durations[seq] = get_video_length(vid) / args.fps
        duration_file = f"{dirname(__file__)}/../../../assets/{dataset}-durations.json"
        with open(duration_file, "w") as f:
            json.dump(durations, f)


if __name__ == "__main__":
    main()
