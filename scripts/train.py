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

# local imports
from dataset import COCODataset
from model.resnet50 import ResNet50
from utils.checkpoint import CheckPoint
from utils.loss import loss_fun


logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def main(args):
    # setup
    writer = SummaryWriter(log_dir='./logs/{}'.format(args.run_name))
    dev = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
    device = torch.device(dev)
    target_size = (args.target_height, args.target_width)

    # setup models
    model = ResNet50()
    model = model.to(device)

    # setup training optimizers
    learning_rate = args.learning_rate

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # setup colmap runner
    dataset_dir = args.dataset_dir
    if(args.dataset_type == 0):
        coco_dataset_dir = path.join(dataset_dir, 'coco')
        train_dataset = COCODataset(coco_dataset_dir, mode='train')
        val_dataset = COCODataset(coco_dataset_dir, mode='validation')
    else:
        raise NotImplementedError

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
        sample_indices = list(range(len(train_dataset)))
        random.shuffle(sample_indices)
        sample_it = tqdm(sample_indices, leave=False)
        for data_index in sample_it:
            data = dataset[data_index]
            images, corrs = data
            images = images.to(device)
            corrs = corrs.to(device)

            # compute loss and update
            normals = model(images)
            loss = loss_fun(normals, corrs)

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('data/loss', loss, total_step)
            sample_it.set_description('loss: {:.02f}'.format(loss))
            if((total_step) % args.image_step == 0):
                writer.add_images('images/normals', (normals + 1 / 2), total_step)
            total_step += 1

        ####################
        # run validation
        # we validate somehow using correspondence
        sample_indices = list(range(len(val_dataset)))
        sample_it = tqdm(sample_indices)
        sample_it.set_description('Iterating samples')
        model.eval()

        validation_score = None
        for data_index in sample_it:
            data = val_dataset[data_index]
            images, corrs = data
            images = images.to(device)
            corrs = corrs.to(device)

            normals = model(images)

            # implement local validation scoring
            precision = None
            raise NotImplementedError


            description = 'precision: {}'.format(precision)
            sample_it.set_description(description)

            # implement global validation scoring
            validation_score = None
            raise NotImplementedError

        # update scheduler based on validation score
        scheduler.step(validation_f_score)

        # save every epoch
        checkpoint_name = '{}-{}.ckpt'.format(args.run_name, epoch)
        checkpoint_path = path.join(args.checkpoint_dir, checkpoint_name)
        CheckPoint.save(checkpoint_path, model, optimizer, scheduler, total_step, epoch)
        CheckPoint.save(latest_checkpoint_path, model, optimizer, scheduler, total_step, epoch)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True, help='theme of this run')
    parser.add_argument('--epochs', type=int, default=30, help='Number of Epochs to run')
    parser.add_argument('--dataset-type', type=int, required=True, help='type of dataset: 0=coco, ...')
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
