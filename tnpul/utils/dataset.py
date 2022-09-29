import torch
from torchvision import transforms, datasets
import numpy as np
import elasticdeform.torch as etorch
import random
from sklearn.model_selection import KFold
import logging

from tnpul.utils.augmentation import image_transformations


def get_datasets(dataset, datadir, transform_train, transform_test):
    dataset_list = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
    assert dataset in dataset_list, f"Dataset {dataset} not in {dataset_list}."

    if dataset == "MNIST":
        train_set = datasets.MNIST(
            datadir, download=True, transform=transform_train)
        val_set = datasets.MNIST(
            datadir, download=True, transform=transform_test)
        test_set = datasets.MNIST(datadir, download=True, transform=transform_test,
                                  train=False)

    if dataset == "FashionMNIST":
        train_set = datasets.FashionMNIST(
            datadir, download=True, transform=transform_train)
        val_set = datasets.FashionMNIST(
            datadir, download=True, transform=transform_test)
        test_set = datasets.FashionMNIST(datadir, download=True, transform=transform_test,
                                         train=False)
    if dataset == "CIFAR10":
        train_set = datasets.CIFAR10(
            datadir, download=True, transform=transform_train)
        val_set = datasets.CIFAR10(
            datadir, download=True, transform=transform_test)
        test_set = datasets.CIFAR10(datadir, download=True, transform=transform_test,
                                    train=False)

    if dataset == "CIFAR100":
        train_set = datasets.CIFAR100(
            datadir, download=True, transform=transform_train)
        val_set = datasets.CIFAR100(
            datadir, download=True, transform=transform_test)
        test_set = datasets.CIFAR100(datadir, download=True, transform=transform_test,
                                     train=False)

    return train_set, val_set, test_set


def init_loaders(*args, all_folds=False, nfolds=5, **kwargs):
    bs = kwargs['bs']

    use_grayscale = kwargs['use_grayscale']

    crop = kwargs['crop']
    random_crop = kwargs['aug_random_crop']

    color_jitter_prob = kwargs['aug_color_jitter_prob']
    brightness = kwargs['aug_brightness']
    contrast = kwargs['aug_contrast']
    saturation = kwargs['aug_saturation']
    hue = kwargs['aug_hue']

    sharp_prob = kwargs['aug_sharpness_prob']
    sharp_min = kwargs['aug_sharp_min']
    sharp_max = kwargs['aug_sharp_max']

    blur_prob = kwargs['aug_gblur_prob']
    blur_kernel_size = np.min([kwargs['aug_gblur_kernel'], crop])

    hflip = kwargs['aug_horizontal_flip']

    affine_prob = kwargs['aug_affine_prob']
    rotate = kwargs['aug_rotate']
    txy = kwargs['aug_translate']
    scale_min = kwargs['aug_scale_min']
    scale_max = kwargs['aug_scale_max']

    perspective_prob = kwargs['aug_perspective_prob']
    perspective_scale = kwargs['aug_perspective_scale']

    elastic_prob = kwargs['aug_elastic_prob']
    elastic_scale = kwargs['aug_elastic_strength']

    erasing_prob = kwargs['aug_erasing_prob']
    erasing_scale_min = kwargs['aug_erasing_scale_min']
    erasing_scale_max = kwargs['aug_erasing_scale_max']

    transform_train, transform_test = image_transformations(use_grayscale=use_grayscale,
                                                            crop=crop,
                                                            random_crop=random_crop,
                                                            color_jitter_prob=color_jitter_prob,
                                                            brightness=brightness,
                                                            contrast=contrast,
                                                            saturation=saturation,
                                                            hue=hue,
                                                            sharp_prob=sharp_prob, sharp_min=sharp_min, sharp_max=sharp_max,
                                                            blur_prob=blur_prob, blur_kernel_size=blur_kernel_size,
                                                            hflip=hflip,
                                                            affine_prob=affine_prob,
                                                            txy=txy,
                                                            rotate=rotate,
                                                            scale_min=scale_min, scale_max=scale_max,
                                                            perspective_prob=perspective_prob,
                                                            perspective_scale=perspective_scale,
                                                            elastic_prob=elastic_prob, elastic_scale=elastic_scale,
                                                            erasing_prob=erasing_prob, erasing_scale_min=erasing_scale_min, erasing_scale_max=erasing_scale_max
                                                            )
    train_set, val_set, test_set = get_datasets(
        kwargs['dataset'], kwargs['datadir'], transform_train, transform_test)

    if kwargs['seed'] > 0:
        kf = KFold(n_splits=nfolds, random_state=kwargs['seed'], shuffle=True)
    else:
        kf = KFold(n_splits=nfolds, random_state=None, shuffle=True)

    # We assume that the train set is sufficiently mixed
    # so we can take first ntrain items for training.
    train_ratio = kwargs["train_ratio"]
    ntrain = int(len(train_set)*train_ratio)
    ntest = len(test_set)
    itest = range(ntest)

    batch_size = {
        "train": bs,
        "test": 100,
        "val": 100
    }

    if all_folds:
        logging.info(f"Generating {nfolds} folds")
        cv_loaders = []
        cv_num_batches = []
        for itrain, ival in kf.split(range(ntrain)):
            samplers = {'train': torch.utils.data.SubsetRandomSampler(itrain),
                        'val': torch.utils.data.SubsetRandomSampler(ival),
                        'test': torch.utils.data.SubsetRandomSampler(itest)}
            loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size[name],
                                                         sampler=samplers[name], drop_last=True) for (name, dataset) in
                       [('train', train_set), ('val', val_set), ('test', test_set)]}
            num_batches = {name: total_num // batch_size[name] for (name, total_num) in
                           [('train', len(itrain)), ('val', len(ival)), ('test', ntest)]}
            cv_loaders.append(loaders)
            cv_num_batches.append(num_batches)

        return cv_loaders, cv_num_batches

    for i, (itrain, ival) in enumerate(kf.split(range(ntrain))):
        if i == kwargs['fold']:
            break
    samplers = {'train': torch.utils.data.SubsetRandomSampler(itrain),
                'val': torch.utils.data.SubsetRandomSampler(ival),
                'test': torch.utils.data.SubsetRandomSampler(itest)}
    loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size[name],
                                                 sampler=samplers[name], drop_last=True) for (name, dataset) in
               [('train', train_set), ('val', val_set), ('test', test_set)]}
    num_batches = {name: total_num // batch_size[name] for (name, total_num) in
                   [('train', len(itrain)), ('val', len(ival)), ('test', ntest)]}

    logging.info(f"Taking fold {kwargs['fold']} out of {nfolds} folds")

    return loaders, num_batches
