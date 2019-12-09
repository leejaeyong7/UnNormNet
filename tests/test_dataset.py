# path imports
import sys
sys.path.append('.')
#sys.path.append('..')

# system imports
import argparse
import os
from os import path
import logging
import shutil
import random

# pytorch imports
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from torch import optim
from torch.nn import utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# third party imports
import json
from tqdm import tqdm
import numpy as np
import yaml

# local imports
from dataset.coco_dataset import COCODataset



def test(args):
    dataset_dir = args.dataset_dir

    with open(args.config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Invalid YAML")
            logging.error(exc)


    if(args.dataset_type == 0):
        coco_dataset_dir = path.join(dataset_dir, 'coco')
        train_dataset = COCODataset(coco_dataset_dir, config, mode='train')
        val_dataset = COCODataset(coco_dataset_dir, config, mode='validation')
    else:
        raise NotImplementedError

    images, corrs = train_dataset[0]
    print(images.shape, corrs.shape)
    print('training image loading working properly')

    images, corrs = val_dataset[0]
    print(images.shape, corrs.shape)
    print('validation image loading working properly')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-type', type=int, required=True, help='type of dataset: 0=coco, ...')
    parser.add_argument('--config-file', type=str, required=True, help='path to augmentation config file')
    parser.add_argument('--dataset-dir', type=str, default='data', help='Path of Dataset. Defaults to ./data')
    args = parser.parse_args()
    test(args)