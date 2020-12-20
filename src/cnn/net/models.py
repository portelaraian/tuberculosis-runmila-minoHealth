import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
from efficientnet_pytorch import EfficientNet
from torch.cuda.amp import autocast

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class SE_ResNeXt50_32x4d(nn.Module):
    def __init__(self):
        super(SE_ResNeXt50_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 2, bias=True))

    def forward(self, x):
        with autocast():
            x = self.model_ft(x)
            return x
        
class SE_ResNeXt101_32x4d(nn.Module):
    def __init__(self):
        super(SE_ResNeXt101_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 2, bias=True))

    def forward(self, x):
        with autocast():
            x = self.model_ft(x)
            return x


class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.model_ft = EfficientNet.from_pretrained('efficientnet-b5')
        num_ftrs = self.model_ft._fc.in_features
        self.model_ft._avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 2, bias=True))

    def forward(self, x):
        with autocast():
            x = self.model_ft(x)
            return x


