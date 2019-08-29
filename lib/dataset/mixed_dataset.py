# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.config import config
from dataset.joints_dataset import JointsDataset
from dataset.multiview_h36m import MultiViewH36M
from dataset.mpii import MPIIDataset


class MixedDataset(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        h36m = MultiViewH36M(cfg, image_set, is_train, transform)
        mpii = MPIIDataset(cfg, image_set, is_train, transform)
        self.h36m_size = len(h36m.db)
        self.db = h36m.db + mpii.db

        self.grouping = h36m.grouping + self.mpii_grouping(
            mpii.db, start_frame=len(h36m.db))

        self.group_size = len(self.grouping)

    def __len__(self):
        return self.group_size

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def mpii_grouping(self, db, start_frame=1):
        mpii_grouping = []
        mpii_length = len(db)
        for i in range(mpii_length // 4):
            mini_group = []
            for j in range(4):
                index = i * 4 + j
                mini_group.append(index + start_frame)
            mpii_grouping.append(mini_group)
        return mpii_grouping
