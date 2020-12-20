import sys
import os
import argparse
import pickle
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
wdir = os.getcwd()
sys.path.insert(0, os.path.join(wdir, ".."))
import mappings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--inputs', help='for ensembling. can be recursively nested for averaging.')
    parser.add_argument('--output')
    parser.add_argument('--sample_submission')
    parser.add_argument('--clip', type=float, default=1e-6)
    parser.add_argument('--is_cv', type=bool, default=False, help='calculate the auc for the local cv')

    args = parser.parse_args()
    assert args.input or args.inputs

    return args


def avg_predictions(results):
    outputs_all = np.array([result['outputs'] for result in results])
    outputs = outputs_all.mean(axis=0)
    return {
        'ids': results[0]['ids'],
        'outputs': outputs,
    }

def calc_auc(targets, outputs):

    macro = roc_auc_score(np.round(targets), outputs, average='macro')
    micro = roc_auc_score(np.round(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }


def read_prediction(path, dirname=''):
    if dirname:
        path = os.path.join(dirname, path)
    print('loading %s...' % path)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return avg_predictions(results)
    

def parse_inputs(inputs, dirname=''):
    results = []
    for elem in inputs:
        if type(elem) is list:
            result = parse_inputs(elem, dirname)
        else:
            result = read_prediction(elem, dirname)
        results.append(result)
    return avg_predictions(results)


def main():
    args = get_args()

    if args.input:
        result = read_prediction(args.input)
    else:
        result = parse_inputs(eval(args.inputs))

    sub = pd.read_csv(args.sample_submission)
    sub.columns = ['ID', 'Label']
    
    
    if args.is_cv == True:
        gt = sub['Label']

    IDs = {}
    for id, outputs in zip(result['ids'], result['outputs']):
        for i, output in enumerate(outputs):
            label = mappings.num_to_label[i]
            ID = '%s_%s' % (id, label)
            IDs[ID] = output

    
    
    sub['Label'] = sub.ID.map(IDs)
    sub.loc[sub.Label.isnull(),'Label'] = sub.Label.min()
    if args.clip:
        print('clip values by %e' % args.clip)
        sub['Label'] = np.clip(sub.Label, args.clip, 1-args.clip)

    print(sub.tail())

    if args.is_cv == True:
        score = calc_auc(gt, sub['Label']) 
        print(score)
        return
    
    sub_tuberculosis = {
    'ID': [],
    'LABEL': []
    }
    
    for id, label in zip(sub.ID, sub.Label):
        id_image, labels = id.split('_')

        if labels == 'tuberculosis':
            sub_tuberculosis['ID'].append(id_image)
            sub_tuberculosis['LABEL'].append(label)
        
    sub_tuberculosis = pd.DataFrame(sub_tuberculosis)


    sub_tuberculosis.to_csv(args.output, index=False)
    print(sub_tuberculosis.tail())
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
