#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# n_nodes=${n_nodes:-"1"}

DATA_INPUT_DIR=${DATA_INPUT_DIR="data/adt-raw/Apartment_release_multiuser_cook_seq141_M1292"}
DATA_PROCESSED_DIR=${DATA_PROCESSED_DIR="data/adt/Apartment_release_multiuser_cook_seq141_M1292"}
VRS_FILE=${VRS_FILE="video.vrs"}
# VRS_FILE=${VRS_FILE="synthetic_video.vrs"}

# MPS folder
MPS_FOLDER=${MPS_FOLDER="$DATA_INPUT_DIR/mps/slam"}
# Ensures the MPS folder contains the following structure
# $MPS_FOLDER
# - closed_loop_trajectory.csv
# - semidense_points.csv.gz
# - semidense_observations.csv.gz
# - online_calibration.jsonl

# Disable --use_factory_lib if we used online calibration for better precision when VIBA is enabled.
# With "--visualize" flag on, the script will stream the processed output in each stage to a rerun visualizer.
# With "--extract_fisheye" flag on, the script will rectify images into the equidistant fisheye images. 

# The rectified RGB size 2400x2400 is the default we value we benchmark all reconstuction in the paper. 
# You can adjust this value (with the focal) that fits best for your applications. 
# For half-resolution (1408x1408) recordings, using focal ~600 and rgb_size as ~1000 is a recommended range.
python tlod/scripts/extract_aria_vrs.py \
    --input_root $DATA_INPUT_DIR \
    --output_root $DATA_PROCESSED_DIR \
    --vrs_file $VRS_FILE \
    --rectified_rgb_focal 600 \
    --rectified_rgb_size 1000 \
    --rectified_monochrome_focal 180 --rectified_monochrome_height 480 \
    --online_calib_file $MPS_FOLDER/online_calibration.jsonl \
    --trajectory_file $MPS_FOLDER/closed_loop_trajectory.csv \
    --semi_dense_points_file $MPS_FOLDER/semidense_points.csv.gz \
    --semi_dense_observation_file $MPS_FOLDER/semidense_observations.csv.gz \
    # --use_factory_calib
    # --visualize
    # --extract_fisheye
    
