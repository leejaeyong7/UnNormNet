# -*- coding: utf-8 -*-
""" Base class for all homography based datsets. """

import torch
import numpy as np
import cv2
import random
import math

from .base_dataset import BaseDataset
class HomographyDataset(BaseDataset):
    '''
    Base class for Homography based datasets.


    '''
    def __init__(self, aug_config):
        return super(HomographyDataset, self).__init__(aug_config)

    def __len__(self) -> int:
        '''
        Returns size of the dataset. Because it is a virtual function,
        it will raise NotImplementedError
        '''
        raise NotImplementedError

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        '''
        Returns item by index. Because it is a virtual function, 
        it will raise NotImplementedError
        '''
        raise NotImplementedError
        

    def sample_images(self, cv_image: np.ndarray) -> (np.ndarray, np.ndarray):
        '''
        Given opencv image, generates N augmenations and N homographies. Note that
        it will return N + 1 images since first image is original image and first
        homography is identity matrix.

        Arguments:
            - cv_image(np.ndarray): original image in HWC np.uin8 format
            - num_augmentations(int): number of total augmentations to make

        Returns:
            - (np.ndarray, np.ndarray): augmented images, and its respective 
                homographies.
        '''
        N = self.num_augmentations
        W = self.target_width
        H = self.target_height

        # clone images

        # augment images
        augmenteds = self.augment_images(cv_image, N)

        # obtain random homographies
        homographies, angles = self.random_sample_homographies(N, shape=(W, H))

        # warp images
        warped = self.warp_images(augmenteds, homographies)

        # prepend original image and identity homography
        return warped, homographies, angles

    def warp_images(self,
                    cv_images: np.ndarray,
                    homographies: np.ndarray,
                    mode: str = 'bilinear') -> np.ndarray:
        '''
        Given images and corresponding homographies, warp them

        Arguments:
          - cv_images(np.ndarray): N x H x W x C shaped arrays to be warped
          - homographies(np.ndarray): N x 3 x 3 homographies

        Returns:
          - np.ndarray: N x H x W x C shaped warped array
        '''
        N, H, W, C  = cv_images.shape
        flag = cv2.INTER_LINEAR
        if(mode == 'nearest'):
            flag = cv2.INTER_NEAREST
        return np.stack([
            cv2.warpPerspective(cv_image,
                                homographies[i],
                                dsize=(W, H),
                                flags=flag)
            for i, cv_image in enumerate(cv_images)
        ])

    def rotmat_from_angles(self, angles):
        return np.stack([
            np.cos(-angles), np.sin(-angles), np.zeros_like(angles),
            -np.sin(-angles), np.cos(-angles),np.zeros_like(angles),
            np.zeros_like(angles),np.zeros_like(angles),np.ones_like(angles),
        ], 2).reshape(-1, 3, 3)


    def compute_dense_correspondence(self, 
                                     images: np.ndarray, 
                                     homographies: np.ndarray) -> np.ndarray:
        '''
        Arguments:
            images(np.ndarray): N x H x W x C np.uint8 images. First image
                denote reference image
            homographies(np.ndarray): N x 3 x 3 homographies. First one should
                be identity matrix

        Returns:
            (np.ndarray): N x H x W x 2 dense correspondences from reference 
                to source pixels, in y-x ordering. correspondences are set to
                float('nan') if it is outside W or H

        '''
        N, H, W, C = images.shape

        # create meshgrid of image coordinates
        Ws = np.arange(W)
        Hs = np.arange(H)
        
        # mesh grid creates x major grids
        # xs = [[0, 1, 2, 3..]]
        # ys = [[0, 0, 0, 0 ..]]
        xs, ys = np.meshgrid(Ws, Hs)
        flat_xs = xs.ravel()
        flat_ys = ys.ravel()
        ones = np.ones_like(flat_xs)

        # homogeneous coordinates are 3 x (H x W) sized array
        h_coords = np.stack([flat_xs, flat_ys, ones]).astype(np.float32)

        # simply multiply homographies with H coords
        #  N x 3 x 3 . 1 x 3 x (HW) => N x 3 x HW
        warped_coords = np.matmul(homographies, h_coords[np.newaxis])

        yx_coords = warped_coords[:, [1, 0]] / warped_coords[:, 2:]
        valid_masks = np.zeros_like(yx_coords)

        # check coordinates
        # valid_masks = N x 2 x HW
        valid_masks[:, 0] = np.logical_and(yx_coords[:, 0] >= 0, yx_coords[:, 0] < H)
        valid_masks[:, 1] = np.logical_and(yx_coords[:, 1] >= 0, yx_coords[:, 1] < W)

        # set masked coords to nan and return
        masked_coords = np.where(valid_masks, yx_coords, float('nan'))
        # masked coords = N x 2 x HW => N x H x W x 2
        return masked_coords.transpose((0, 2, 1)).reshape(N, H, W, 2).astype(np.float32)

    def random_sample_homographies(self,
                                   num_homographies: int,
                                   shape:(int, int),
                                   perspective: bool = True,
                                   scaling: bool = True,
                                   rotation: bool = True,
                                   translation: bool = True,
                                   n_scales: int = 5,
                                   n_angles: int = 25,
                                   scaling_amplitude: float = 0.2,
                                   perspective_amplitude_x: float = 0.2,
                                   perspective_amplitude_y: float = 0.2,
                                   patch_ratio: float = 0.85,
                                   max_angle: float = math.pi/2,
                                   allow_artifacts: bool = True,
                                   trns_overflow: float = 0.) -> np.ndarray:
        '''
        Randomly samples homographies

        Args:
            num_homographies (int): number of homographies to create
            shape (tuple): shape of the image
            perspective (bool): perform perspective transform
            scaling (bool): perform scaling
            rotation (bool): perform in-plane rotation
            translation (bool): perform translation
            n_scales (int): number of scales to try
            n_angles (int): number of andles to try
            scaling_amplitude (float): amplitude of scaling (1 - a) ~ (1 + a)
            perspective_amplitude_x (float): perspective amplitude in x dir
            perspective_amplitude_y (float): perspective amplitude in y dir
            patch_ratio (float): ratio of patch to use for cropping
            max_angle (float): maximum angle in in plane rotation
            allow_artifacts (bool): allow empty regions
            trns_overflow (float): add overflow offset to translation
        Returns:
            (np.ndarray): array of of shape N x 3 x 3 representing N homographies.
        '''
        N = num_homographies
        # Corners of the output image
        #
        # ref_corners are of shape N x 4 x 2
        ref_corners = np.array([
            [0, 0], # top left
            [0, 1], # bottom left
            [1, 1], # bottom right
            [1, 0], # top left
        ], dtype=np.float32).reshape((1, 4, 2)).repeat(N, 0)

        # Corners of the input patch
        margin = (1 - patch_ratio) / 2

        # ref_corners are of shape N x 4 x 2
        src_corners = np.array([
            [0, 0],
            [0, patch_ratio],
            [patch_ratio, patch_ratio],
            [patch_ratio, 0]
        ], dtype=np.float32).reshape((1, 4, 2)).repeat(N, 0) + margin

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scale_samples = np.random.normal(1, scaling_amplitude / 2, N * n_scales)

            # scales = N_s,
            scales = np.clip(scale_samples, 1-scaling_amplitude, 1 + scaling_amplitude)

            # src_corners = N x 4 x 2
            # centers = N x 1 x 2
            centers = np.expand_dims(src_corners.mean(1), 1)

            # scaled = N x N_s x 4 x 2
            centereds = np.expand_dims((src_corners - centers), 1)
            scaleds = centereds * scales.reshape((N, -1, 1, 1))

            if allow_artifacts:
                valid_range = list(range(n_scales))
                idxs = np.random.choice(valid_range, N)
                src_corners = scaleds[range(N), idxs]
            else:
                # N_s
                idxs = []
                for scaled in scaleds:
                    valid_scales = ((scaled < 0) * (scaled >= 1)).sum(1).sum(1) == 0
                    valid_range = valid_scales.nonzero()[0].tolist()
                    idx = random.choice(valid_range)
                    idxs.append(idx)
                src_corners = scaleds[range(N), idxs]

        # Random translation
        if translation:
            # t_min = N x 4 x 2
            t_min = src_corners.min(1)
            t_max = (1 - src_corners).min(1)
            if allow_artifacts:
                t_min += trns_overflow
                t_max += trns_overflow

            # translation_x = N,
            #

            translation_x = np.random.uniform(-t_min[:, 0], t_max[:, 0])
            translation_y = np.random.uniform(-t_min[:, 1], t_max[:, 1])
            trans = np.stack([translation_x, translation_y], 1).reshape(N, 1, 2)

            src_corners += trans

        # Random rotation
        # sample several rotations, check collision with borders, randomly pick a valid one
        if rotation:
            # angles  N x N_a,
            angles = np.linspace(-max_angle, max_angle, N * n_angles).reshape(N, -1)
            # N x N-a + 1
            angles = np.concatenate((angles, np.array([0] * N, dtype=np.float32)[:, np.newaxis]), axis=1)

            # center = N x 1 x 2
            centers = np.expand_dims(src_corners.mean(1), 1)

            # rotmat = N x (N_a + 1) x 2 x 2
            rot_mat = np.stack([
                np.cos(angles),
                np.sin(angles),
                -np.sin(angles),
                np.cos(angles),
            ], 2).reshape(N, -1, 2, 2)

            # (N x 4 x 2) . (N x (N_a + 1) x 2 x 2) = N x N_a x 4 x 2
            # centereds = N x 4 x 2  => N x 1 x 4 x 2
            centereds = np.expand_dims((src_corners - centers), 1)
            # rotateds = N x 1 x 4 x 2  @ N x (N_a + 1) x 2 x 2 =>
            # N x (N_a + 1) x 4 x 2 + N x 1 x 1 x 2
            # N x (N_a + 1)
            rotateds = (centereds @ rot_mat) + centers[:, np.newaxis]

            if allow_artifacts:
                valid_range = list(range(n_angles))
                idxs = np.random.choice(valid_range, N)
                src_corners = rotateds[range(N), idxs]
            else:
                # N_s
                idxs = []
                for rotated in rotateds:
                    valid_ranged = ((rotated < 0) * (rotated >= 1)).sum(1).sum(1) == 0
                    valid_range = valid_ranged.nonzero()[0].tolist()
                    idx = random.choice(valid_range)
                    idxs.append(idx)
                src_corners = rotateds[range(N), idxs]

            angles = -angles[:, idxs]

        # Rescale to actual size
        # shape = (H, W)
        t_shape = np.array(list(shape)).reshape(1, 1, 2)
        ref_corners *= t_shape
        src_corners *= t_shape

        #
        homographies = [ np.eye(3, dtype=np.float32) ]
        for i, ref_corner in enumerate(ref_corners):
            homography = cv2.findHomography(ref_corners, src_corners[i], 0)[0]
            homographies.append(homography)


        return np.stack(homographies).astype(np.float32), angles

    def _resize_homography(self, homography, original_size, target_size):
        # build intrinsic matrix
        orig_to_target = np.array([
            [target_size[0] / original_size[0], 0, 0],
            [0, target_size[1] / original_size[1], 0],
            [0, 0, 1],
        ])
        target_to_orig = np.linalg.inv(orig_to_target)
        return orig_to_target.dot(homography).dot(target_to_orig)
