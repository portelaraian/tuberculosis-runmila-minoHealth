import os
import pickle
import random
import sys
import pandas as pd
import numpy as np
np.seterr(over='ignore')
import torch
import cv2
import pydicom
from skimage import exposure
wdir = os.getcwd()
sys.path.insert(0, os.path.join(wdir, ".."))
import factory
from utils.logger import log
from utils import mappings


def apply_dataset_policy(df, policy):
    if policy == 1: # use all records
        pass
    else:
        raise RuntimeError('Unexpected dataset policy %s' % policy)
    log('applied dataset_policy %s (%d records now)' % (policy, len(df)))
    return df

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg

        log(f'dataset_policy: {self.cfg.dataset_policy}')
        log(f'window_policy: {self.cfg.window_policy}')

        self.transforms = factory.get_transforms(self.cfg)
        with open(cfg.annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            log('read dataset (%d records)' % len(self.df))

        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)
        # self.df = self.df.sample(100)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = '%s/%s.npy' % (self.cfg.imgdir, row.ID)
        
        image = np.load(path, allow_pickle=True)
        
        image = self.transforms(image=image)['image']

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0

        return image, torch.FloatTensor(target), row.ID
