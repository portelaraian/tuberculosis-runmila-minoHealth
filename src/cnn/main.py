import sys
import os
sys.path.append(os.path.abspath(os.path.join('./')))

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import time
import argparse
import random
import collections
from collections import OrderedDict
import pickle
import ssl

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F

import factory
from utils.lrs_schedulers import WarmRestart, warm_restart 
from utils import util
from utils.config import Config
from utils.logger import logger, log



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output') 
    parser.add_argument('--n-tta', default=1, type=int)
    return parser.parse_args()

def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.snapshot = args.snapshot
    cfg.output = args.output
    cfg.n_tta = args.n_tta
    cfg.gpu = args.gpu
    
    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(0)
    util.set_seed(cfg.seed)

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')
    log(f'acc: {cfg.data.train.n_grad_acc}')

    model = factory.get_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'valid':
        valid(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)

def test(cfg, model):
    assert cfg.output
    load_model(cfg.snapshot, model)
    loader_test = factory.get_dataloader(cfg.data.test)
    with torch.no_grad():
        results = [run_nn(cfg.data.test, 'test', model, loader_test) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)

def valid(cfg, model):
    assert cfg.output
    criterion = factory.get_loss(cfg)
    load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold] if cfg.fold is not None else None)
    with torch.no_grad():
        results = [run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)

def load_model(path, model, optim=None):
    # remap everthing onto CPU 
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    
    try:
        model.load_state_dict(state['model'])
    except RuntimeError:
        log('loaded model from new state_dict')
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        
    if optim:
        log('loading optim too')
        optim.load_state_dict(state['optim'])
    else:
        log('not loading optim')

    model = torch.nn.DataParallel(model).cuda()

    detail = state['detail']
    log('loaded model from %s' % path)

    return detail
    
def train(cfg, model):
    criterion = factory.get_loss(cfg)
    optim = factory.get_optim(cfg, model.parameters())

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }
    if cfg.resume_from:
        log("resuming from %s" % str(cfg.resume_from))

        detail = load_model(cfg.resume_from, model, optim=optim)
    
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })
    
        model = torch.nn.DataParallel(model).cuda()

    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    loader_train = factory.get_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])

    log('train data: loaded %d records' % len(loader_train.dataset))
    log('valid data: loaded %d records' % len(loader_valid.dataset))

    scheduler = WarmRestart(optim, T_max=5, T_mult=1, eta_min=1e-5)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in range(best['epoch']+1, cfg.epoch):

        log(f'\n----- epoch {epoch} ----- fold {cfg.fold} -----')

        if epoch < 30:
            if epoch != 0:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
        elif epoch > 29 and epoch < 33:
            optim.param_groups[0]['lr'] = 1e-5
        elif epoch < 45:
            scheduler.step()
            scheduler = warm_restart(scheduler, T_mult=2)
        elif epoch > 44 and epoch < 49:
            optim.param_groups[0]['lr'] = 1e-5
        else:
            optim.param_groups[0]['lr'] = 5e-6

        run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optim=optim, autocast=autocast, grad_scaler=scaler)
        
        with torch.no_grad():
            val = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['score'] > best['score']:
            best.update(detail)
            util.save_model(model, optim, detail, cfg.fold, cfg.workdir)
            
        log('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
            
def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, autocast=None, grad_scaler=None):

    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise RuntimeError('Unexpected mode %s' % mode)

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []
    
    
    for i, (inputs, targets, ids) in enumerate(loader):

        batch_size = len(inputs)       
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        
        if mode in ['train', 'valid']:
            # Runs the forward pass with autocasting.
            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                loss    = loss / cfg.n_grad_acc
                
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            grad_scaler.scale(loss).backward() # accumulate loss
            if (i+1) % cfg.n_grad_acc == 0:
                grad_scaler.step(optim) # update
                grad_scaler.update()
                optim.zero_grad() # flush
        
        if mode in ['test']:
            outputs = model(inputs)

        with torch.no_grad():
            ids_all.extend(ids)
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())
            
        
        

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} lr:{util.get_lr(optim):.2e}'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['train', 'valid']:
        try:
            result.update(calc_auc(result['targets'], result['outputs']))
            result['score'] = result['auc']
            log(progress + ' auc:%.4f' % (result['auc']))
        except Exception:
            log(' no gt')
    else:
        log('')

    return result

def calc_auc(targets, outputs):
    macro = roc_auc_score(np.round(targets), outputs, average='macro')
    micro = roc_auc_score(np.round(targets), outputs, average='micro')

    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }

if __name__ == '__main__':
    ssl._create_default_https_context  = ssl._create_unverified_context
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()
    print('benchmark', torch.backends.cudnn.benchmark)

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
