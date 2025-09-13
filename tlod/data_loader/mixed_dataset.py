# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
from torch.utils.data import Dataset

from ..easyvolcap.utils.console_utils import logger  # noqa: F401


class MixedDataset(Dataset):
    def __init__(
        self,
        datasets,
        ratios,
    ):
        super().__init__()
        ratios = np.asarray(ratios)
        self.datasets = datasets
        self.ratios = ratios
        # For estimation of epoch length
        mean_len = np.sum([len(d) * r for d, r in zip(datasets, ratios)])
        # Make sure everyone has a ratio > 1
        self.virt_lengths = (ratios / ratios.min()).astype(int) * mean_len
        self.cumsums = [0] + self.virt_lengths.cumsum(-1).astype(int).tolist()
        self.virt_total_len = self.cumsums[-1]
        self.batch_image_num = self.datasets[0].batch_image_num

    def __len__(self):
        return self.virt_total_len

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumsums, idx, side="right").item() - 1
        sample_idx = (idx - self.cumsums[dataset_idx]) % len(self.datasets[dataset_idx])
        logger.info(f"idx={idx}, dataset_idx={dataset_idx}, sample_idx={sample_idx}")
        return self.datasets[dataset_idx][sample_idx]
