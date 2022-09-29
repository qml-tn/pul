import torch
import random
import numpy as np
from torchvision import transforms, datasets
import elasticdeform.torch as etorch


def image_transformations(use_grayscale=False,
                          resize=-1,
                          random_crop=False,
                          crop=24,
                          color_jitter_prob=0,
                          brightness=0,  # [0, 1]
                          contrast=0,  # [0, 1]
                          saturation=0,  # [0, 1]
                          hue=0,  # [0,0.5]
                          # [0.01, 1], [1, 3]
                          sharp_prob=0, sharp_min=0.01, sharp_max=3,
                          # [0, 1]  [3, image_width] should be odd
                          blur_prob=0, blur_kernel_size=11,
                          hflip=False,
                          affine_prob=0,
                          txy=0,  # [0., 2.0]
                          rotate=0,  # [0.0, 20.0]
                          scale_min=1, scale_max=1,  # [0.6,1], [1,1.5]
                          perspective_prob=0,  # [0, 1]
                          perspective_scale=0,  # [0, 1]
                          # [0,1], [0.05,5.0]
                          elastic_prob=0.0, elastic_scale=1.0,
                          # [0, 1], [0,0.02], [0.05,0.4]
                          erasing_prob=0.0, erasing_scale_min=0.02, erasing_scale_max=0.2
                          ):
    trans = []

    # Random ColorJitter
    if color_jitter_prob > 0:
        trans.append(transforms.RandomApply([transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                                    saturation=saturation, hue=hue)], p=color_jitter_prob))

    # Random sharpness
    if sharp_prob > 0:
        def f_sharp(X):
            sharp_fac = (sharp_max-sharp_min)*np.random.rand()+sharp_min
            return transforms.functional.adjust_sharpness(X, sharpness_factor=sharp_fac)

        trans.append(transforms.RandomApply(
            [transforms.Lambda(f_sharp)], p=sharp_prob))

    # Random GaussianBlur
    if blur_prob > 0:
        blur_kernel_size = blur_kernel_size - np.mod(blur_kernel_size+1, 2)
        trans.append(transforms.RandomApply([transforms.GaussianBlur(
            kernel_size=blur_kernel_size, sigma=(0.1, 2.0))], p=blur_prob))

    # Random flip
    if hflip:
        trans.append(transforms.RandomHorizontalFlip())

    # Random affine transformation
    if affine_prob > 0:
        trans.append(transforms.RandomApply([transforms.RandomAffine(
            rotate, translate=(txy, txy), scale=(scale_min, scale_max))], p=affine_prob))

    # Random perspective transformation
    if perspective_prob > 0:
        trans.append(transforms.RandomPerspective(
            distortion_scale=perspective_scale, p=perspective_prob))

    if not random_crop:
        if resize > 0:
            trans.append(transforms.Resize(resize))
            if crop > resize:
                crop = resize

        # Random crop
        trans.append(transforms.RandomCrop(crop))

    # To tensor
    trans.append(transforms.ToTensor())

    # Random elastic transformation
    if elastic_prob > 0:
        def elastic(X):
            displacement_val = np.random.randn(2, 3, 3) * elastic_scale
            displacement = torch.tensor(displacement_val)
            return etorch.deform_grid(X, displacement, order=3, axis=[(1, 2)])
        trans.append(transforms.RandomApply(
            [transforms.Lambda(elastic)], p=elastic_prob))

    # Random erasing
    if erasing_prob > 0:
        trans.append(transforms.RandomErasing(p=erasing_prob, scale=(erasing_scale_min, erasing_scale_max),
                                              ratio=(0.3, 3.3)))

    test_trans = [
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
    ]

    if use_grayscale:
        trans.append(transforms.Grayscale())
        test_trans.append(transforms.Grayscale())

    transform_train = transforms.Compose(trans)

    transform_test = transforms.Compose(test_trans)

    return transform_train, transform_test
