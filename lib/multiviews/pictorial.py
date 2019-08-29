# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import scipy
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import multiviews.cameras as cameras
from multiviews.body import HumanBody
from utils.transforms import get_affine_transform, affine_transform, affine_transform_pts

import numexpr as ne


def infer(unary, pairwise, body, config):
    """
    Args:
        unary: a list of unary terms for all JOINTS
        pairwise: a list of pairwise terms of all EDGES
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """

    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level
    root_idx = config.DATASET.ROOTIDX
    nbins = len(unary[root_idx])
    states_of_all_joints = {}

    # zhe 20190104 replace torch with np
    for node in skeleton_sorted_by_level:
        # energy = []
        children_state = []
        unary_current = unary[node['idx']]
        # unary_current = torch.tensor(unary_current, dtype=torch.float32).to(dev)
        if len(node['children']) == 0:
            energy = unary[node['idx']].squeeze()
            children_state = [[-1]] * len(energy)
        else:
            children = node['children']
            for child in children:
                child_energy = states_of_all_joints[child][
                    'Energy'].squeeze()
                pairwise_mat = pairwise[(node['idx'], child)]
                if type(pairwise_mat) == scipy.sparse.csr.csr_matrix:
                    pairwise_mat = pairwise_mat.toarray()
                unary_child = child_energy
                unary_child_with_pairwise = np.multiply(pairwise_mat, unary_child)
                # unary_child_with_pairwise = ne.evaluate('pairwise_mat*unary_child')
                max_i = np.argmax(unary_child_with_pairwise, axis=1)
                max_v = np.max(unary_child_with_pairwise, axis=1)
                unary_current = np.multiply(unary_current, max_v)
                children_state.append(max_i)

            # rearrange children_state
            children_state = np.array(children_state).T.tolist()

        res = {'Energy': np.array(unary_current), 'State': children_state}
        states_of_all_joints[node['idx']] = res
    # end here 20181225

    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy']
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
    pose3d = np.zeros(shape=[njoints, 3])
    is_single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if is_single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d


def compute_grid(boxSize, boxCenter, nBins):
    grid1D = np.linspace(-boxSize / 2, boxSize / 2, nBins)
    gridx, gridy, gridz = np.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],
    )
    dimensions = gridx.shape[0] * gridx.shape[1] * gridx.shape[2]
    gridx, gridy, gridz = np.reshape(gridx, (dimensions, -1)), np.reshape(
        gridy, (dimensions, -1)), np.reshape(gridz, (dimensions, -1))
    grid = np.concatenate((gridx, gridy, gridz), axis=1)
    return grid


def compute_pairwise_constrain(skeleton, limb_length, grid, tolerance):
    pairwise_constrain = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']

        for child in children:
            expect_length = limb_length[(current, child)]
            nbin_current = len(grid[current])
            nbin_child = len(grid[child])
            constrain_array = np.zeros((nbin_current, nbin_child))

            for i in range(nbin_current):
                for j in range(nbin_child):
                    actual_length = np.linalg.norm(grid[current][i] -
                                                   grid[child][j])
                    offset = np.abs(actual_length - expect_length)
                    if offset <= tolerance:
                        constrain_array[i, j] = 1
            pairwise_constrain[(current, child)] = constrain_array

    return pairwise_constrain


def compute_unary_term(heatmap, grid, bbox2D, cam, imgSize):
    """
    Args:
        heatmap: array of size (n * k * h * w)
                -n: number of views,  -k: number of joints
                -h: heatmap height,   -w: heatmap width
        grid: list of k ndarrays of size (nbins * 3)
                    -k: number of joints; 1 when the grid is shared in PSM
                    -nbins: number of bins in the grid
        bbox2D: bounding box on which heatmap is computed
    Returns:
        unary_of_all_joints: a list of ndarray of size nbins
    """

    n, k = heatmap.shape[0], heatmap.shape[1]
    h, w = heatmap.shape[2], heatmap.shape[3]
    nbins = grid[0].shape[0]

    unary_of_all_joints = []
    for j in range(k):
        unary = np.zeros(nbins)
        for c in range(n):

            grid_id = 0 if len(grid) == 1 else j
            xy = cameras.project_pose(grid[grid_id], cam[c])
            trans = get_affine_transform(bbox2D[c]['center'],
                                         bbox2D[c]['scale'], 0, imgSize)

            xy = affine_transform_pts(xy, trans) * np.array([w, h]) / imgSize
            # for i in range(nbins):
            #     xy[i] = affine_transform(xy[i], trans) * np.array([w, h]) / imgSize

            hmap = heatmap[c, j, :, :]
            point_x, point_y = np.arange(hmap.shape[0]), np.arange(
                hmap.shape[1])
            rgi = RegularGridInterpolator(
                points=[point_x, point_y],
                values=hmap.transpose(),
                bounds_error=False,
                fill_value=0)
            score = rgi(xy)
            unary = unary + np.reshape(score, newshape=unary.shape)
        unary_of_all_joints.append(unary)

    return unary_of_all_joints


def recursive_infer(initpose, cams, heatmaps, boxes, img_size, heatmap_size,
                    body, limb_length, grid_size, nbins, tolerance, config):

    k = initpose.shape[0]
    grids = []
    for i in range(k):
        point = initpose[i]
        grid = compute_grid(grid_size, point, nbins)
        grids.append(grid)

    unary = compute_unary_term(heatmaps, grids, boxes, cams, img_size)

    skeleton = body.skeleton
    pairwise_constrain = compute_pairwise_constrain(skeleton, limb_length,
                                                    grids, tolerance)
    pose3d_cube = infer(unary, pairwise_constrain, body, config)
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)

    return pose3d


def rpsm(cams, heatmaps, boxes, grid_center, limb_length, pairwise_constraint,
         config):
    """
    Args:
        cams : camera parameters for each view
        heatmaps: 2d pose heatmaps (n, k, h, w)
        boxes: on which the heatmaps are computed; n dictionaries
        grid_center: 3d location of the root
        limb_length: template limb length
        pairwise_constrain: pre-computed pairwise terms (iteration 1)
    Returns:
        pose3d: 3d pose
    """
    image_size = config.NETWORK.IMAGE_SIZE
    heatmap_size = config.NETWORK.HEATMAP_SIZE
    first_nbins = config.PICT_STRUCT.FIRST_NBINS
    recur_nbins = config.PICT_STRUCT.RECUR_NBINS
    recur_depth = config.PICT_STRUCT.RECUR_DEPTH
    grid_size = config.PICT_STRUCT.GRID_SIZE
    tolerance = config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE

    # Iteration 1: discretizing 3d space
    body = HumanBody()
    grid = compute_grid(grid_size, grid_center, first_nbins)
    unary = compute_unary_term(heatmaps, [grid], boxes, cams, image_size)
    pose3d_as_cube_idx = infer(unary, pairwise_constraint, body, config)
    pose3d = get_loc_from_cube_idx([grid], pose3d_as_cube_idx)

    cur_grid_size = grid_size / first_nbins
    for i in range(recur_depth):
        pose3d = recursive_infer(pose3d, cams, heatmaps, boxes, image_size,
                                 heatmap_size, body, limb_length, cur_grid_size,
                                 recur_nbins, tolerance, config)
        cur_grid_size = cur_grid_size / recur_nbins

    return pose3d
