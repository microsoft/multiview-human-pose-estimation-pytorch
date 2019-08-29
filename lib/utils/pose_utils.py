# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import normalize


class PoseUtils:

    def estimate_camera(self, pose_2d, pose_3d, indices=None):
        """Estimate camera parameters given 2d-3d pose pair.
        Args:
            pose_2d: [n_joint, 2] 2d pose array
            pose_3d: Correspondent [n_joint, 3] 3d pose array
            indices: Indices used to do least square. At least 4 independent points should provided. 
                    All of the points will be used if not specified.
        Returns:
            A [2, 3] projection matrix and a [2] translation matrix.
        """
        if indices is None:
            indices = np.arange(pose_2d.shape[0])
        pose_2d = pose_2d.reshape([-1, 2])
        pose_3d = pose_3d.reshape([-1, 3])
        pose_2d_mean = np.mean(pose_2d, axis=0)
        pose_3d_mean = np.mean(pose_3d, axis=0)
        pose_2d = pose_2d - pose_2d_mean
        pose_3d = pose_3d - pose_3d_mean

        M = np.matmul(pose_2d[indices].T, np.linalg.pinv(pose_3d.T))
        U, s, Vt = np.linalg.svd(M)
        R = np.matmul(np.matmul(U, np.array([[1, 0, 0], [0, 1, 0]])), Vt)
        M = np.matmul(np.diag(s), R)
        t = pose_2d_mean - np.matmul(M, pose_3d_mean)

        R3 = np.cross(R[0, :], R[1, :])
        R3 = np.reshape(R3, (1, 3))
        R3 = normalize(R3)

        camera = {'R': np.concatenate((R, R3), axis=0), 's': s, 't': t}
        return camera

    def align_3d_to_2d(self, pose_2d, pose_3d, camera, rootIdx):
        """ Given the 2d and 3d poses, we align 3D pose to the 2D image frame, z of root is zero
        Args:
            pose_2d: [n_joint, 2] 2d pose array
            pose_3d: Correspondent [n_joint, 3] 3d pose array
        Returns:
            aligned3d: Correspondent [n_joint, 3] 3d pose array 
        """
        R = camera['R']
        s = np.mean(camera['s'])
        t = np.reshape(camera['t'], (2, 1))
        translation = np.dot(inv(R), np.vstack((t / s, s)))
        aligned3d = s * np.dot(R, (pose_3d + translation.T).T).T
        return aligned3d - np.array([0, 0, aligned3d[rootIdx, 2]])

    def procrustes(self, A, B, scaling=True, reflection='best'):
        """ A port of MATLAB's `procrustes` function to Numpy.
        $$ \min_{R, T, S} \sum_i^N || A_i - R B_i + T ||^2. $$
        Use notation from [course note]
        (https://fling.seas.upenn.edu/~cis390/dynamic/slides/CIS390_Lecture11.pdf).
        Args:
            A: Matrices of target coordinates.
            B: Matrices of input coordinates. Must have equal numbers of  points
                (rows), but B may have fewer dimensions (columns) than A.
            scaling: if False, the scaling component of the transformation is forced
                to 1
            reflection:
                if 'best' (default), the transformation solution may or may not
                include a reflection component, depending on which fits the data
                best. setting reflection to True or False forces a solution with
                reflection or no reflection respectively.
        Returns:
            d: The residual sum of squared errors, normalized according to a measure
                of the scale of A, ((A - A.mean(0))**2).sum().
            Z: The matrix of transformed B-values.
            tform: A dict specifying the rotation, translation and scaling that
                maps A --> B.
        """
        assert A.shape[0] == B.shape[0]
        n, dim_x = A.shape
        _, dim_y = B.shape

        # remove translation
        A_bar = A.mean(0)
        B_bar = B.mean(0)
        A0 = A - A_bar
        B0 = B - B_bar

        # remove scale
        ssX = (A0**2).sum()
        ssY = (B0**2).sum()
        A_norm = np.sqrt(ssX)
        B_norm = np.sqrt(ssY)
        A0 /= A_norm
        B0 /= B_norm

        if dim_y < dim_x:
            B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

        # optimum rotation matrix of B
        A = np.dot(A0.T, B0)
        U, s, Vt = np.linalg.svd(A)
        V = Vt.T
        R = np.dot(V, U.T)

        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(R) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                R = np.dot(V, U.T)

        S_trace = s.sum()
        if scaling:
            # optimum scaling of B
            scale = S_trace * A_norm / B_norm

            # standarised distance between A and scale*B*R + c
            d = 1 - S_trace**2

            # transformed coords
            Z = A_norm * S_trace * np.dot(B0, R) + A_bar
        else:
            scale = 1
            d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
            Z = B_norm * np.dot(B0, R) + A_bar

        # transformation matrix
        if dim_y < dim_x:
            R = R[:dim_y, :]
        translation = A_bar - scale * np.dot(B_bar, R)

        # transformation values
        tform = {'rotation': R, 'scale': scale, 'translation': translation}
        return d, Z, tform


if __name__ == '__main__':
    pose2d = np.array(
        [[115.42669678, 102.42271423], [99.8081665, 100.83456421], [
            97.40727234, 154.66975403
        ], [93.27631378, 198.52540588], [130.71669006, 103.99852753], [
            127.00919342, 156.07492065
        ], [116.97068024, 199.52674866], [116.74355316, 72.27806854],
         [117.79602051, 41.93487549], [119.92079926, 31.99210548], [
             119.03995514, 17.96786118
         ], [135.2973175, 41.59934235], [165.03352356, 48.2557373],
         [193.47923279, 49.47089005], [98.16778564, 41.83195496],
         [66.04647827, 52.59766006], [39.56548309, 56.02058792]])

    pose3d = np.array([[-81.77583313, -552.1887207, 4291.38916016], [
        -211.30630493, -559.60925293, 4243.83203125
    ], [-236.88369751, -112.57989502,
        4357.40820312], [-287.40042114, 277.21643066, 4596.80908203], [
            47.75523758, -544.76800537, 4338.94726562
        ], [16.53305054, -103.14489746,
            4470.79199219], [-74.98716736, 291.76489258, 4688.68212891], [
                -69.65709686, -795.61425781, 4218.54589844
            ], [-60.29067993, -1038.8338623,
                4161.70996094], [-41.86167908, -1097.89172363, 4069.44775391], [
                    -48.99368286, -1212.49609375, 4063.14331055
                ], [85.64931488, -1060.91186523,
                    4238.390625], [344.31176758, -1031.31384277, 4349.76513672],
                       [592.30114746, -1025.36315918, 4360.91748047], [
                           -219.86462402, -1028.68896484, 4115.89404297
                       ], [-473.94354248, -924.9197998, 4046.16748047],
                       [-662.23358154, -865.71044922, 3895.49780273]])

    box = np.array([234, 125, 736, 627])
    utils = PoseUtils()
    camera = utils.estimate_camera(pose2d, pose3d)
    print(camera['R'])
    aligned3d = utils.align_3d_to_2d(pose2d, pose3d, camera, 0)
    print(aligned3d)


    import cv2
    image = cv2.imread('./s_11_act_16_subact_02_ca_04_000001.jpg')
    print(image.shape)
    image = image[125:627, 234:736, :]
    npoints = aligned3d.shape[0]
    for i in range(npoints):
        cv2.circle(image, (int(aligned3d[i, 0]*502.0/224.0), int(aligned3d[i, 1]*502.0/224.0)), radius=14, color=(0,255,0))

    cv2.imshow('image', image)
    cv2.waitKey(0)


