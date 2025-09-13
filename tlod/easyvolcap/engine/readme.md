# EasyVolcap Engine

## Overview
This directory contains the configuration and registry system used in the 4DGT project. The code in this directory is adapted from [EasyVolcap](https://github.com/zju3dv/EasyVolcap/tree/main/easyvolcap/engine), which itself is a modified version of the configuration and registry system from [XRNeRF](https://github.com/openxrlab/xrnerf).

## Code Lineage
1. **Original Source**: The configuration and registry system was originally developed by the OpenMMLab team as part of [MMCV](https://github.com/open-mmlab/mmcv)
2. **XRNeRF Adaptation**: [XRNeRF](https://github.com/openxrlab/xrnerf) adapted this system for NeRF-related projects
3. **EasyVolcap Version**: [EasyVolcap](https://github.com/zju3dv/EasyVolcap) further modified the system for volumetric capture applications
4. **Current Implementation**: This version is directly adapted from EasyVolcap's implementation with additional modifications for the 4DGT project

## Key Components
- **config.py**: Configuration file parsing and management system
- **registry.py**: Module registration system for dynamic component instantiation
- **file_client.py**: File I/O abstraction layer
- **io.py**: I/O utilities for various file formats
- **misc.py**: Miscellaneous utility functions
- **parse.py**: Configuration parsing utilities
- **path.py**: Path manipulation utilities

## License
- **EasyVolcap**: Licensed under the MIT License
- **XRNeRF**: Licensed under the Apache License 2.0
- **MMCV**: Licensed under the Apache License 2.0

This part of the code is subject to the terms of the Apache License 2.0, following XRNeRF's licensing. Please refer to the respective repositories for detailed license information.
