# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiviews.cameras_cuda as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform


def infer(unary, pairwise, body, config):
    """
    Args:
        unary: [list] unary terms of all joints
        pairwise: [list] pairwise terms of all edges
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """
    root_idx = config.DATASET.ROOTIDX

    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level

    states_of_all_joints = {}
    for node in skeleton_sorted_by_level:
        children_state = []
        u = unary[node['idx']].clone()
        if len(node['children']) == 0:
            energy = u
            children_state = [[-1]] * energy.numel()
        else:
            for child in node['children']:
                pw = pairwise[(node['idx'], child)]
                ce = states_of_all_joints[child]['Energy']
                ce = ce.expand_as(pw)
                pwce = torch.mul(pw, ce)
                max_v, max_i = torch.max(pwce, dim=1)
                u = torch.mul(u, max_v)
                children_state.append(max_i.detach().cpu().numpy())

            children_state = np.array(children_state).T

        res = {'Energy': u, 'State': children_state}
        states_of_all_joints[node['idx']] = res

    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy'].detach().cpu().numpy()
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])

    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]

        children_index = skeleton[joint_idx]['children']
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])

    pose3d_as_cube_idx.sort()
    return pose3d_as_cube_idx


def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = torch.zeros(njoints, 3, device=grid[0].device)
    single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d


def compute_grid(boxSize, boxCenter, nBins, device=None):
    grid1D = torch.linspace(-boxSize / 2, boxSize / 2, nBins, device=device)
    gridx, gridy, gridz = torch.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],
    )
    gridx = gridx.contiguous().view(-1, 1)
    gridy = gridy.contiguous().view(-1, 1)
    gridz = gridz.contiguous().view(-1, 1)
    grid = torch.cat([gridx, gridy, gridz], dim=1)
    return grid


def pdist2(x, y):
    """
    Compute distance between each pair of row vectors in x and y

    Args:
        x: tensor of shape n*p
        y: tensor of shape m*p
    Returns:
        dist: tensor of shape n*m
    """
    p = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    xtile = torch.cat([x] * m, dim=1).view(-1, p)
    ytile = torch.cat([y] * n, dim=0)
    dist = torch.pairwise_distance(xtile, ytile)
    return dist.view(n, m)


def compute_pairwise(skeleton, limb_length, grid, tolerance):

    pairwise = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            expect_length = limb_length[(current, child)]
            distance = pdist2(grid[current], grid[child]) + 1e-9
            pairwise[(current, child)] = (torch.abs(distance - expect_length) <
                                          tolerance).float()
    return pairwise


def compute_unary_term(heatmap, grid, bbox2D, cam, imgSize):
    """
    Args:
        heatmap: array of size (n * k * h * w)
                -n: views,      -k: joints
                -h: height,     -w: width

        grid: k lists of ndarrays of size (nbins * 3)
                -k: joints; 1 when the grid is shared in PSM
                -nbins: bins in the grid

        bbox2D: bounding box on which heatmap is computed

    Returns:
        unary_of_all_joints: a list of ndarray of size nbins
    """
    device = heatmap.device
    share_grid = len(grid) == 1

    n, k = heatmap.shape[0], heatmap.shape[1]
    h, w = heatmap.shape[2], heatmap.shape[3]

    all_unary = {}
    for v in range(n):
        center = bbox2D[v]['center']
        scale = bbox2D[v]['scale']
        trans = torch.as_tensor(
            get_transform(center, scale, 0, imgSize),
            dtype=torch.float,
            device=device)

        for j in range(k):
            grid_id = 0 if len(grid) == 1 else j
            nbins = grid[grid_id].shape[0]

            if (share_grid and j == 0) or not share_grid:
                xy = cameras.project_pose(grid[grid_id], cam[v])
                xy = do_transform(xy, trans) * torch.tensor(
                    [w, h], dtype=torch.float, device=device) / torch.tensor(
                        imgSize, dtype=torch.float, device=device)

                sample_grid = xy / torch.tensor(
                    [h - 1, w - 1], dtype=torch.float,
                    device=device) * 2.0 - 1.0
                sample_grid = sample_grid.view(1, 1, nbins, 2)

            unary_per_view_joint = F.grid_sample(
                heatmap[v:v + 1, j:j + 1, :, :], sample_grid)

            if j in all_unary:
                all_unary[j] += unary_per_view_joint
            else:
                all_unary[j] = unary_per_view_joint

    all_unary_list = []
    for j in range(k):
        all_unary_list.append(all_unary[j].view(1, -1))
    return all_unary_list


def recursive_infer(initpose, cams, heatmaps, boxes, img_size, heatmap_size,
                    body, limb_length, grid_size, nbins, tolerance, config):

    device = heatmaps.device
    njoints = initpose.shape[0]
    grids = []
    for i in range(njoints):
        grids.append(compute_grid(grid_size, initpose[i], nbins, device=device))

    unary = compute_unary_term(heatmaps, grids, boxes, cams, img_size)

    skeleton = body.skeleton
    pairwise = compute_pairwise(skeleton, limb_length, grids, tolerance)

    pose3d_cube = infer(unary, pairwise, body, config)
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)

    return pose3d


def rpsm(cams, heatmaps, config, kw):
    """
    Args:
        cams : camera parameters for each view
        heatmaps: 2d pose heatmaps (n, k, h, w)
    Returns:
        pose3d: 3d pose
    """

    # all in this device
    device = heatmaps.device
    img_size = config.NETWORK.IMAGE_SIZE
    map_size = config.NETWORK.HEATMAP_SIZE
    grd_size = config.PICT_STRUCT.GRID_SIZE
    fst_nbins = config.PICT_STRUCT.FIRST_NBINS
    rec_nbins = config.PICT_STRUCT.RECUR_NBINS
    rec_depth = config.PICT_STRUCT.RECUR_DEPTH
    tolerance = config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE

    grid = compute_grid(grd_size, kw['center'], fst_nbins, device=device)
    unary = compute_unary_term(heatmaps, [grid], kw['boxes'], cams, img_size)

    pose3d_as_cube_idx = infer(unary, kw['pairwise'], kw['body'], config)
    pose3d = get_loc_from_cube_idx([grid], pose3d_as_cube_idx)

    cur_grd_size = grd_size / fst_nbins
    for i in range(rec_depth):
        pose3d = recursive_infer(pose3d, cams, heatmaps, kw['boxes'], img_size,
                                 map_size, kw['body'], kw['limb_length'],
                                 cur_grd_size, rec_nbins, tolerance, config)
        cur_grd_size = cur_grd_size / rec_nbins

    return pose3d
