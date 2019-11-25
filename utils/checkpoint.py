# -*- coding: utf-8 -*-
"""Boilerplate Model saver.

Handles loading / saving model / optimizer / scheduler / epoch and other 
variables used for running network.
"""

import os
from os import path

import torch
from torch import optim

class CheckPoint:
    @staticmethod
    def save(checkpoint_path, model, 
             optimizer=None, scheduler=None, total_step=None, epoch=None):
        """ Saves model in checkpoint path. """
        if(not path.exists(path.dirname(checkpoint_path))):
            os.makedirs(path.dirname(checkpoint_path))
        # save model (of course)
        checkpoint = {
            'model':None,
            'optimizer':None,
            'scheduler':None,
            'epoch':None,
            'total_step':None,
        }
        checkpoint['model'] = model.state_dict()
        if(optimizer is not None):
            checkpoint['optimizer'] = optimizer.state_dict()
        if(scheduler is not None):
            checkpoint['scheduler'] = scheduler.state_dict()
        if(epoch is not None):
            checkpoint['epoch'] = epoch
        if(total_step is not None):
            checkpoint['total_step'] = total_step
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load(checkpoint_path, device=None):
        """ Loads model from checkpoint path. """
        if(not path.exists(checkpoint_path)):
            raise Exception("Checkpoint Path Not Found!")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint

