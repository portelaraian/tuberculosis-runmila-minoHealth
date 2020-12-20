import os
import random
import glob

import pandas as pd
import numpy as np
import torch

from .logger import log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lr(optim):
    if optim:
        return optim.param_groups[0]['lr']
    else:
        return 0


def save_model(model, optim, detail, fold, dirname):
    path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'detail': detail,
    }, path)
    log('saved model to %s' % path)


