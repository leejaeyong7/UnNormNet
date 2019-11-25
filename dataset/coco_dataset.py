import os
from os import path
from PIL import Image
import logging

import torch
import numpy as np
from .homography_dataset import HomographyDataset
import time


class COCODataset(HomographyDataset):
    """
    MS COCO Dataset

    Arguments:


    Returns:
    """
    def __init__(self, dataset_dir, aug_config, mode='train'):
        super(COCODataset, self).__init__(aug_config)
        if mode == 'train':
            sub_path = 'train2017'
        elif mode == 'val':
            sub_path = 'val2017'
        else:
            sub_path = 'test2017'

        self.dataset_dir = path.join(dataset_dir, sub_path)
        all_images = os.listdir(self.dataset_dir)
        self.samples = [
            path.join(self.dataset_dir, image_name)
            for image_name in all_images
        ]

    def __len__(self) -> int:
        ''' Returns number of samples in this dataset '''
        return len(self.samples)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        sample = Image.open(self.samples[index])

        cv_image = np.array(sample)
        if(len(cv_image.shape) != 3):
            cv_image = np.tile(np.expand_dims(cv_image, 2), (1, 1, 3))
        augmented, homographies = self.sample_images(cv_image)
        dense_corrs = self.compute_dense_correspondence(augmented, homographies)

        return augmented, dense_corrs
