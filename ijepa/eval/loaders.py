# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from PIL import Image

import numpy as np
import pandas as pd
import psycopg2
import os
from sklearn.model_selection import GroupShuffleSplit
import torch
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
    RandSpatialCropd)
logger = logging.getLogger("dinov2")


class CAGDataset(torch.utils.data.Dataset):
    def __init__(self, config, dino_transforms=None, target_transforms=None):
        self.config = config
        self.dino_transforms = dino_transforms
        self.target_transforms = target_transforms
        self._construct_loader()

        # contrusct specific cag transforms
        self.__transforms__()
                
    def getDataFromDatabase(self, config):
        connection = psycopg2.connect(
            host=config['host'],
            database=config['database'],
            user=config['username'],
            password=config['password'])
        sql = config['query'].replace(
            "?table_name", "\"" + config['table_name'] + "\"")
        sql = sql.replace(
            "?schema_name", "\"" + config['schema_name'] + "\"")
        sql = sql.replace(
            "??", "\"")
        df = pd.read_sql_query(sql, connection)
        if len(df) == 0:
            print('The requested query does not have any data!')
        connection.close()
        return df

    def set_data_path(self, features):
        for feature in features:
            self.df[feature] = self.df[feature].apply(
                        lambda x: os.path.join(self.config['DataSetPath'], x))

    def get_input_features(self, csv, features='DcmPathFlatten'):
        if features == 'DcmPathFlatten':
            features = [col for col in
                        csv.columns.tolist() if col.startswith(features)]
        else:
            features = features
        return features
    

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        self.df = self.getDataFromDatabase(self.config)
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.data = self.df[self.features + ['rowid']]
        self.data = self.data.to_dict('records')

        print('Constructing cag dataset')
        

    def __transforms__(self):
        self.transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                # SpatialPadd(
                #     keys=self.features,
                #     spatial_size=[512,
                #                 512,
                #                 -1]),
                RandSpatialCropd(
                    keys=self.features,
                    roi_size=[
                        -1,
                        -1,
                        1],
                    random_size=False),
                RepeatChanneld(keys=self.features, repeats=3),
                ScaleIntensityd(keys=self.features),
                EnsureTyped(keys=self.features, data_type='tensor'),
                ConcatItemsd(keys=self.features, name='inputs')                
                ]

        self.transforms = Compose(self.transforms, log_stats=True)
        self.transforms.set_random_state(seed=0)
        return self.transforms
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
            # Get the path for the video and its corresponding label
            # video_path = self._path_to_videos[index]
            sample = self.data[index]
            sample = self.transforms(sample)
            frames = sample[self.features[0]]
            frames = torch.squeeze(frames)
            frames = frames.permute(1, 2, 0)
            frames = Image.fromarray(np.uint8(frames.numpy()*255))
            frames = self.dino_transforms(frames)
            # Return preprocessed video frames and corresponding label
            return frames, 1 #, torch.tensor(label, dtype=torch.float32)
        
class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    # if name == "ImageNet":
    #     class_ = ImageNet
    #     if "split" in kwargs:
    #         kwargs["split"] = ImageNet.Split[kwargs["split"]]
    # elif name == "ImageNet22k":
    #     class_ = ImageNet22k
    if name == "CAG":
        class_ = CAGDataset
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cfg: dict = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    if dataset_str == "CAG":
        dataset = class_(cfg, dino_transforms=transform, target_transforms=target_transform)
    else:
        dataset = class_(transform=transform, target_transform=target_transform, **kwargs)
    dataset[0]
    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
