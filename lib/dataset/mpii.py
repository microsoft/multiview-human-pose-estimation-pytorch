# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json_tricks as json

from dataset.joints_dataset import JointsDataset


class MPIIDataset(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'rank',
            1: 'rkne',
            2: 'rhip',
            3: 'lhip',
            4: 'lkne',
            5: 'lank',
            6: 'root',
            7: 'thorax',
            8: 'upper neck',
            9: 'head top',
            10: 'rwri',
            11: 'relb',
            12: 'rsho',
            13: 'lsho',
            14: 'lelb',
            15: 'lwri'
        }
        self.db = self._get_db()

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

    def _get_db(self):
        file_name = os.path.join(self.root, 'mpii', 'annot',
                                 self.subset + '.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_vis = np.zeros((16, 3), dtype=np.float)
            if self.subset != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                vis = np.array(a['joints_vis'])

                joints_vis[:, 0] = vis[:]
                joints_vis[:, 1] = vis[:]

            gt_db.append({
                'image': image_name,
                'center': c,
                'scale': s,
                'joints_2d': joints,
                'joints_3d': np.zeros((16, 3)),
                'joints_vis': joints_vis,
                'source': 'mpii'
            })

        return gt_db
