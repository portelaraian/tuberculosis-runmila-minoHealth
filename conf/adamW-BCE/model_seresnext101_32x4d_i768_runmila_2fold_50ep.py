workdir = './model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep'
seed = 300

n_fold = 2
epoch = 50
resume_from = None

batch_size = 20
num_workers = 32
imgsize = (768, 768) #(height, width)

loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optim = dict(
    name='AdamW',
    params=dict(
        lr=0.0003,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    ),
)

model = dict(
    name='se_resnext101_32x4d'
)


normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],}
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
crop_test = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
rotate_test = dict(name='Rotate', params=dict(limit=25, border_mode=0, p=0.7))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5))

'''
Additional augmentarions
------------------------

vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
random_brightness_contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
#gaussian_blur = dict(name='GaussianBlur', params=dict(blur_limit=7, always_apply=False, p=0.5))
#iaa_emboss = dict(name='IAAEmboss', params=dict(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5))
#iaa_sharpen = dict(name='IAASharpen', params=dict(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5))
hue_saturation_value = dict(name='HueSaturationValue', params=dict(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.4))
cut_out = dict(name='Cutout', params=dict(num_holes=8, max_h_size=546//8, max_w_size=546//8, fill_value=0, p=0.3))
blur = dict(name='Blur', params=dict(blur_limit=4, p=.25))
shift_scale_rotate = dict(name='ShiftScaleRotate', params=dict(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=1))
'''
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0, p=0.7))
dicomnoise = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.06, p=0.9))
dicomnoise_test = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.05, p=0.7))
elastic_transform = dict(name='ElasticTransform', params=dict(alpha=1, sigma=50,  p=0.5))
grid_distortion = dict(name='GridDistortion', params=dict(), p=0.5)


window_policy = 1

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations='./cache/train-runmila_2folds_seed123.pkl',
        imgdir='./input/runmila_i768',
        imgsize=imgsize,
        n_grad_acc=2,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, dicomnoise, totensor],
        dataset_policy=1,
        window_policy=window_policy,
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations='./cache/train-runmila_2folds_seed123.pkl',
        imgdir='./input/runmila_i768',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop_test, hflip, rotate_test, dicomnoise_test, totensor],
        dataset_policy=1,
        window_policy=window_policy,
    ),
    test = dict(
        dataset_type='CustomDataset',
        annotations='./cache/test.pkl',
        imgdir='./input/test_runmila_i768',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop_test, hflip, rotate_test, dicomnoise_test, totensor],
        dataset_policy=1,
        window_policy=window_policy,
    ),
)
