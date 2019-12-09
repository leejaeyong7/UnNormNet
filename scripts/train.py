# path imports
import sys
sys.path.append('.')
sys.path.append('..')

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
from dataset import COCODataset
from dataset import ImageNetDataset
from torch.utils.data import Dataset, DataLoader
from models.resnet50 import ED
from utils.checkpoint import CheckPoint
from utils.loss import batch_surface_normal_loss


logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def main(args):
    # setup
    writer = SummaryWriter(log_dir='./logs/{}'.format(args.run_name))
    dev = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
    device = torch.device(dev)

    with open(args.config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Invalid YAML")
            logging.error(exc)

    # setup models
    model = ED()
    model = model.to(device)

    # setup training optimizers
    learning_rate = args.learning_rate

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # setup colmap runner
    dataset_dir = args.dataset_dir
    imagenet_dataset_dir = path.join(dataset_dir, 'imagenet100')
    train_dataset = ImageNetDataset(imagenet_dataset_dir, config, mode='train')
    val_dataset = ImageNetDataset(imagenet_dataset_dir, config, mode='val')

    epoch_start = 0
    epoch_end = args.epochs
    total_step = 0

    # load from checkpoint iff exists
    latest_checkpoint_name = '{}-latest.ckpt'.format(args.run_name)
    latest_checkpoint_path = path.join(args.checkpoint_dir, latest_checkpoint_name)

    if((path.exists(latest_checkpoint_path)) and args.resume):
        checkpoint = CheckPoint.load(latest_checkpoint_path, device)
        model_weight = checkpoint['model']
        optimizer_weight = checkpoint['optimizer']
        scheduler_weight = checkpoint['scheduler']
        epoch_start = checkpoint['epoch']
        total_step = checkpoint['total_step']

        model.load_state_dict(model_weight)
        optimizer.load_state_dict(optimizer_weight)
        scheduler.load_state_dict(scheduler_weight)


    # setup dataset and actually run training
    for epoch in range(epoch_start, epoch_end):
        ####################
        # run training
        model.train()
        num_samples = len(train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        sample_it = tqdm(enumerate(dataloader), total=num_samples // args.batch_size)
        for i_batch, data in sample_it:
            # image
            images, corrs, rotmat = data
            images = images.to(device)
            corrs = corrs.to(device)
            rotmat = rotmat.to(device)
            B, _, _, H, W = images.shape

            cat_images = images.view(-1, 3, H, W)

            # compute loss and update
            # normals = N x 2 x 3 x H x W
            normals = model(cat_images).view(B, 2, 2, H, W)
            x = normals[:, :, 0]
            y = normals[:, :, 1]
            # z = normals[:, :, 2]

            loss = batch_surface_normal_loss(normals, corrs, rotmat)

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            xm = x.mean()
            ym = y.mean()
            writer.add_scalar('data/loss', loss, total_step)
            sample_it.set_description('loss: {:.04f}, xm: {:.03f}, ym: {:.03f}'.format(loss, xm, ym))
            if ((total_step % args.image_step ) == 0):
                writer.add_images('images/images', images.view(-1, 3, H, W), total_step)
                ni = torch.ones(B * 2, 3, H, W)
                ni[:, :2] = normals.view(-1, 2, H, W)
                writer.add_images('images/normals', (ni + 1 / 2), total_step)
            total_step += 1
        # save every epoch
        checkpoint_name = '{}-{}.ckpt'.format(args.run_name, epoch)
        checkpoint_path = path.join(args.checkpoint_dir, checkpoint_name)
        CheckPoint.save(checkpoint_path, model, optimizer, scheduler, total_step, epoch)
        CheckPoint.save(latest_checkpoint_path, model, optimizer, scheduler, total_step, epoch)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, required=True, help='size of batch')
    parser.add_argument('--num-workers', type=int, required=True, help='workers for dataloder')
    parser.add_argument('--run-name', type=str, required=True, help='theme of this run')
    parser.add_argument('--config-file', type=str, required=True, help='path to augmentation config file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of Epochs to run')
    parser.add_argument('--dataset-dir', type=str, default='data', help='Path of Dataset. Defaults to ./data')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID used for this run. Default=CPU')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='Path of checkpoint. Defaults to ./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from previous checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not Resume from previous checkpoint')
    parser.add_argument('--image-step', type=int, default=50, help='Recurring number of steps for showing image in tensorboard')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    main(args)
