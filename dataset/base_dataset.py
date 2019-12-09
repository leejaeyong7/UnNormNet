# -*- coding: utf-8 -*-
""" Base class for all datsets. """

import cv2
import torch
import numpy as np
import imgaug
from torch.utils.data import Dataset

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmentables import Keypoint, KeypointsOnImage

class BaseDataset(Dataset):
    '''
    Base class for Homography based datasets.
    '''
    def __init__(self, aug_config):
        super(BaseDataset).__init__()
        self.aug_config = aug_config

        self.num_augmentations = aug_config['num-augmentations']
        target_width = aug_config['image-width']
        target_height = aug_config['image-height']

        valid_aug_keys = [
            key
            for key in aug_config.keys()
            if key not in ['image-width', 'image-height', 'num-augmentations']
        ]

        aug_functions = {
            'jpeg-compression': iaa.JpegCompression,
            'grayscale': iaa.Grayscale,
            'motion-blur': iaa.MotionBlur,
            'color-perturb': iaa.AddToHueAndSaturation,
            'gaussian-blur': iaa.GaussianBlur,
            'gaussian-noise': iaa.AdditiveGaussianNoise
        }

        augmentations = [iaa.Noop()]
        for valid_aug_key in valid_aug_keys:
            aug_obj = aug_config[valid_aug_key]
            aug_func = aug_functions[valid_aug_key]
            aug_prob = aug_obj['probability']
            aug_dist = aug_obj['distribution']
            dist_type = aug_dist['type']
            if dist_type == 'uniform':
                dist_start = aug_dist['start']
                dist_end = aug_dist['end']
                aug_arg = (dist_start, dist_end)
            else:
                dist_mean = aug_dist['mean']
                dist_std = aug_dist['std']

                if dist_type == 'normal':
                    aug_arg = iap.Normal(dist_mean, dist_std)
                elif dist_type == 'truncated-normal':
                    dist_start = aug_dist['start']
                    dist_end = aug_dist['end']
                    aug_arg = iap.TruncatedNormal(
                        dist_mean,
                        dist_std,
                        dist_start,
                        dist_end)
            augmentations.append(iaa.Sometimes(aug_prob, aug_func(aug_arg)))
        self.augment_image = iaa.Sequential(augmentations, True)
        self.resize_image = iaa.Resize({'height': target_height, 'width': target_width})
        # self.resize_crop_image = iaa.CropToFixedSize(height=target_height, width= target_width)
        self.resize_crop_image = iaa.Sequential([
            iaa.CropToFixedSize(height=target_height, width= target_width),
            iaa.PadToFixedSize(height=target_height, width= target_width)
        ])
        self.target_width = target_width
        self.target_height = target_height

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

    def augment_images(self, cv_image: np.ndarray, num_augmentations: int) -> np.ndarray:
        '''
        Takes a single OpenCV image (H x W x C) and number of augmentations, to augment
        image N times.
        Args:
            cv_image:
            num_augmentations:

        Returns:
            N x H x W x C
        '''

        # simply call augment_image function
        resized = self.resize_crop_image(image=cv_image)
        resizeds = np.tile(resized, (num_augmentations, 1, 1, 1))
        augmenteds = self.augment_image(images=resizeds)

        return np.concatenate((resized[np.newaxis], augmenteds))

    def cv_to_tensor(self, cv_image: np.ndarray) -> torch.Tensor:
        num_axis = len(cv_image.shape)
        if(num_axis == 3):
            order = [2, 0, 1]
        elif(num_axis == 4):
            order = [0, 3, 1, 2]
        return (torch.from_numpy(cv_image).float() * 255).permute(*order)

    def tensor_to_cv(self, torch_image: torch.Tensor) -> np.ndarray:
        num_axis = len(torch_image.shape)
        if(num_axis == 3):
            order = [1, 2, 0]
        elif(num_axis == 4):
            order = [0, 2, 3, 1]
        return (torch_image.permute(*order).numpy() / 255).astype(np.uint8)

    def _resize_image(self, image, W, H):
        new_img = cv2.resize(image, dsize=(W, H),
                            interpolation=cv2.INTER_LINEAR)
        return new_img
