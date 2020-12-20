import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join('./')))

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np

from net.models import SE_ResNeXt50_32x4d, SE_ResNeXt101_32x4d, EfficientNetB3

from transformers import get_cosine_schedule_with_warmup
from dataset.custom_dataset import CustomDataset
from transforms.transforms import RandomResizedCrop, RandomResizedCrop, RandomDicomNoise
from utils.logger import log


def get_loss(cfg):
    loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    #loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([1, 1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss

def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader

def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)

def get_model(cfg):

    log(f'model: {cfg.model.name}')

    if cfg.model.name == 'efficientnet-b3':
        model = EfficientNetB3()
        return model
    elif cfg.model.name == 'se_resnext50_32x4d':
        model = SE_ResNeXt50_32x4d()
        return model
    elif cfg.model.name == 'se_resnext101_32x4d':
        model = SE_ResNeXt101_32x4d()
        return model

def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim

def get_scheduler(cfg, optim, last_epoch, n_epochs, folds):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch

    elif cfg.scheduler.name == 'CosineScheduleWarmup':
        train_dataset = CustomDataset(cfg.data.train, folds)
        n_train_steps = int(len(train_dataset) / int(cfg.batch_size) * n_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=len(train_dataset)/int(cfg.batch_size)*5,
            num_training_steps=n_train_steps,
        )

    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f'last_epoch: {last_epoch}')
    return scheduler
