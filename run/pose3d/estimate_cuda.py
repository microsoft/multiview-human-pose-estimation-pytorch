# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle
import argparse
import numpy as np

import torch

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger
from multiviews.pictorial_cuda import rpsm
from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Pose Estimation')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length


def main():

    device = torch.device('cuda:0')
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test3d')

    prediction_path = os.path.join(final_output_dir,
                                   config.TEST.HEATMAP_LOCATION_FILE)
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)

    all_heatmaps = h5py.File(prediction_path)['heatmaps']
    pairwise_file = config.PICT_STRUCT.PAIRWISE_FILE
    with open(pairwise_file, 'rb') as f:
        pairwise = pickle.load(f)['pairwise_constrain']
    for k, v in pairwise.items():
        pairwise[k] = torch.as_tensor(
            v.todense().astype(np.float), device=device, dtype=torch.float)

    cnt = 0
    grouping = test_dataset.grouping
    mpjpes = []
    for items in grouping:
        heatmaps = []
        boxes = []
        poses = []
        cameras = []

        for idx in items:
            datum = test_dataset.db[idx]
            camera = datum['camera']
            cameras.append(camera)

            poses.append(
                camera_to_world_frame(datum['joints_3d_camera'], camera['R'],
                                      camera['T']))

            box = {}
            box['scale'] = np.array(datum['scale'])
            box['center'] = np.array(datum['center'])
            boxes.append(box)

            heatmaps.append(all_heatmaps[cnt])
            cnt += 1
        heatmaps = np.array(heatmaps)

        # This demo uses GT root locations and limb length; but can be replaced by statistics
        grid_center = poses[0][0]
        body = HumanBody()
        limb_length = compute_limb_length(body, poses[0])

        heatmaps = torch.as_tensor(heatmaps, device=device)
        kw = {
            'body': body,
            'boxes': boxes,
            'center': grid_center,
            'pairwise': pairwise,
            'limb_length': limb_length
        }
        prediction = rpsm(cameras, heatmaps, config, kw)
        prediction = prediction.cpu().numpy()
        mpjpe = np.mean(np.sqrt(np.sum((prediction - poses[0])**2, axis=1)))
        mpjpes.append(mpjpe)
    print(np.mean(mpjpes))


if __name__ == '__main__':
    main()
