#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# n_nodes=${n_nodes:-"1"}

echo "Data input directory: " $DATA_INPUT_DIR
echo "Preprocessed data directory:" $DATA_PROCESSED_DIR
echo "VRS filename: " $VRS_FILE

# MPS folder
MPS_FOLDER=${MPS_FOLDER="$DATA_INPUT_DIR/mps/slam"}
# Ensures the MPS folder contains the following structure
# $MPS_FOLDER
# - closed_loop_trajectory.csv
# - semidense_points.csv.gz
# - semidense_observations.csv.gz
# - online_calibration.jsonl

# With "--visualize" flag on, the script will stream the processed output in each stage to a rerun visualizer.
# With "--extract_fisheye" flag on, the script will rectify images into the equidistant fisheye images. 

# You can adjust this value (with the focal) that fits best for your applications. 
# For half-resolution Aria (1408x1408) recordings, using focal ~600 and rgb_size as ~1000 is a recommended range.
python tlod/scripts/extract_aria_vrs.py \
    --input_root $DATA_INPUT_DIR \
    --output_root $DATA_PROCESSED_DIR \
    --vrs_file $VRS_FILE \
    --rectified_rgb_focal 600 \
    --rectified_rgb_size 1000 \
    --rectified_monochrome_focal -1 \
    --online_calib_file $MPS_FOLDER/online_calibration.jsonl \
    --trajectory_file $MPS_FOLDER/closed_loop_trajectory.csv \
    --semi_dense_points_file $MPS_FOLDER/semidense_points.csv.gz \
    --semi_dense_observation_file $MPS_FOLDER/semidense_observations.csv.gz \
    # --use_factory_calib
    # --visualize
    # --extract_fisheye

