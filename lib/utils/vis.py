# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import os
import matplotlib.pyplot as plt

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 file_name,
                                 nrow=8,
                                 padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                               [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])), 1,
                       [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(input, meta['joints_2d_transformed'],
                                     meta['joints_vis'],
                                     '{}_gt.jpg'.format(prefix1))
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(input, joints_pred, meta['joints_vis'],
                                     '{}_pred.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(input, target, '{}_hm_gt.jpg'.format(prefix2))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(input, output, '{}_hm_pred.jpg'.format(prefix2))


def visualize_aggre_weights(aggre_layer, input, output, meta):
    batch_index = 0
    images = []
    heatmaps = []
    pose = []

    for _input, _output, _meta in zip(input, output, meta):
        img = _input[batch_index].clone()
        # normalize
        min, max = float(img.min()), float(img.max())
        img.add_(-min).div_(max - min + 1e-5)

        images.append(
            img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        heatmaps.append(_output[batch_index].clone().cpu().numpy())
        pose.append(_meta['joints'][batch_index].clone().cpu().numpy())

    view1, view2 = cv2.resize(images[0], (80, 80)), cv2.resize(
        images[1], (80, 80))
    heat1, heat2 = heatmaps[0], heatmaps[1]
    joint2 = pose[1]

    weights = aggre_layer.aggre[0].weight.data.cpu().numpy()

    for joint_index in range(joint2.shape[0]):
        h2 = heat2[joint_index]

        plt.figure()
        plt.subplot(121)
        plt.imshow(view2[:, :, ::-1])
        plt.imshow(h2, alpha=0.5)

        point = joint2[joint_index][:2] / 4
        idx = int(h2.shape[0] * point[1] + point[0])
        value = h2[int(point[1]), int(point[0])]
        print(value)

        new_map = value * weights[idx]
        new_map = new_map.reshape(h2.shape)
        new_map[new_map < 0] = 0

        plt.subplot(122)
        plt.imshow(view1[:, :, ::-1])
        plt.imshow(new_map, alpha=0.5)
        plt.show()


def vis_mid_heatmaps(images, heatmaps, finals, tar_idx, idx):
    batch_index = 0
    imgs = []
    hmaps = []

    for _img, _hmap in zip(images, heatmaps):
        img = _img[batch_index].clone()
        min, max = float(img.min()), float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
        imgs.append(
            img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        hmaps.append(_hmap[batch_index].clone().mul(255).clamp(
            0, 255).byte().cpu().numpy())

    heat1_final = finals[batch_index].clone().mul(255).clamp(
        0, 255).byte().cpu().numpy()
    heat1 = hmaps[0]
    heat1_transfer = hmaps[1] + hmaps[2] + hmaps[3]

    h, w = heat1.shape[1:]

    view = cv2.resize(imgs[tar_idx], (h, w))

    heats_set = [heat1_final, heat1, heat1_transfer]

    num_joints = heat1.shape[0]
    grid_image = np.zeros((num_joints * h, 4 * w, 3), dtype=np.uint8)

    for i in range(len(heats_set)):
        map = heats_set[i]

        for j in range(num_joints):

            h = map[j]
            colored_h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
            masked_img = colored_h * 0.7 + view * 0.3

            height_begin = h * j
            height_end = h * (j + 1)

            width_begin = w * (i + 1)
            width_end = w * (i + 2)

            grid_image[height_begin:height_end, width_begin:
                       width_end, :] = masked_img
            grid_image[height_begin:height_end, 0:w, :] = view

    save_root = '/data/extra/haibo/testing_samples'
    imgname = 'sample{:0>5}_view{}.jpg'.format(idx, tar_idx + 1)
    file_name = os.path.join(save_root, imgname)
    cv2.imwrite(file_name, grid_image)
