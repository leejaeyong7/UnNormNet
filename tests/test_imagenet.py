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
from dataset import ImageNetDataset



def test(args):
    dataset_dir = args.dataset_dir

    with open(args.config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Invalid YAML")
            logging.error(exc)

    imagenet_dataset_dir = path.join(dataset_dir, 'imagenet100')
    train_dataset = ImageNetDataset(imagenet_dataset_dir, config, mode='train')
    val_dataset = ImageNetDataset(imagenet_dataset_dir, config, mode='val')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=4)
    for i, b in enumerate(train_dataloader):
        print(i)
        print(len(b))
        for s in b:
            print(s.shape)
        break

    images, corrs, rotmat = train_dataset[0]
    print(images.shape, corrs.shape, rotmat.shape)
    print('training image loading working properly')

    images, corrs, rotmat = val_dataset[0]
    print(images.shape, corrs.shape, rotmat.shape)
    print('validation image loading working properly')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-type', type=int, required=True, help='type of dataset: 0=coco, ...')
    parser.add_argument('--config-file', type=str, required=True, help='path to augmentation config file')
    parser.add_argument('--dataset-dir', type=str, default='data', help='Path of Dataset. Defaults to ./data')
    args = parser.parse_args()
    test(args)
