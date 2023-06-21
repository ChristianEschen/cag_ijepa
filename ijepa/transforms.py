# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstD,
    Compose,
    ScaleIntensityd,
    RepeatChanneld,
    ToPILd,
    ConcatItemsd,
    SpatialPadd,
    DeleteItemsd,
    EnsureTyped,
    TorchVisiond,
    RandSpatialCropd,
    RandTorchVisiond,
    SqueezeDimd,
    NormalizeIntensityd)
_GLOBAL_SEED = 0
logger = getLogger()

def make_cag_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
    keys=['DcmPathFlatten']
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    # cag aspecific
    transform_list = [LoadImaged(keys=keys),
                EnsureChannelFirstD(keys=keys),
                SpatialPadd(
                    keys=keys,
                    spatial_size=[crop_size,
                                crop_size,
                                -1]),
                RandSpatialCropd(keys=keys,
                    roi_size=[
                        -1,
                        -1,
                        1],
                    random_size=False),
                RepeatChanneld(keys=keys, repeats=3),
                SqueezeDimd(keys=keys, dim=3),
                ]
    transform_list += [
        RandTorchVisiond(
        keys=keys[0],
        name="RandomResizedCrop",
        size=crop_size,
        scale=crop_scale)
        ]
    # if horizontal_flip:
    #     transform_list += [transforms.RandomHorizontalFlip()]
    # if color_distortion:
    #     raise NotImplementedError
    #     transform_list += [get_color_distortion(s=color_jitter)]
    # if gaussian_blur:
    #     raise NotImplementedError
    #     transform_list += [GaussianBlur(p=0.5)]
    transform_list += [ScaleIntensityd(keys=keys),]
    transform_list += [NormalizeIntensityd(
                keys=keys,
                subtrahend=(0.45, 0.45, 0.45),#(0.43216, 0.394666, 0.37645),
                divisor=(0.225, 0.225, 0.225),#(0.22803, 0.22145, 0.216989),
                channel_wise=True)]

    transform = transforms.Compose(transform_list)
    return transform



def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
