# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import shutil

# Simple PathManager using standard Python I/O

class PathManager:
    @staticmethod
    def ls(path):
        return os.listdir(path)

    @staticmethod
    def rm(path):
        if not os.path.exists(path):
            return  # File doesn't exist, nothing to remove
        if os.path.isdir(path):
            return shutil.rmtree(path)
        else:
            return os.remove(path)

    @staticmethod
    def isfile(path):
        return os.path.isfile(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def copy_from_local(src, tgt, overwrite=True, **kwargs):
        if src != tgt:
            try:
                return shutil.copy(src, tgt, **kwargs)
            except OSError:
                pass

    @staticmethod
    def mv(src, tgt, **kwargs):
        return shutil.move(src, tgt)

    @staticmethod
    def copy(src, tgt, overwrite=True, recursive=False, **kwargs):
        if src != tgt:
            try:
                if recursive:
                    return shutil.copytree(src, tgt, **kwargs)
                else:
                    return shutil.copy(src, tgt, **kwargs)
            except OSError:
                return tgt
        return src

    @staticmethod
    def open(*args, **kwargs):
        return open(*args, **kwargs)

    @staticmethod
    def mkdirs(path, exist_ok=True):
        return os.makedirs(path, exist_ok=exist_ok)

    @staticmethod
    def get_local_path(path, cache_dir=None, recursive=False, **kwargs):
        if cache_dir is None:
            return path
        else:
            if recursive:
                cache_dir = os.path.join(cache_dir, os.path.basename(path))
            return PathManager.copy(path, cache_dir, recursive=recursive, **kwargs)

    @staticmethod
    def symlink(src, tar, **kwargs):
        os.symlink(src, tar, **kwargs)

    @staticmethod
    def get_modified_time(path):
        return os.path.getmtime(path)


# Create a single instance to use
pathmgr = PathManager()


# Simplified helper functions - no special handling needed for standard file paths


def mkdirs(dirpath):
    from .dist_helper import get_rank

    if get_rank() == 0 and (not pathmgr.isdir(dirpath)):
        pathmgr.mkdirs(dirpath)
    return dirpath
