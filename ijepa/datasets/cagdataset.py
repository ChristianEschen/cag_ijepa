# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time
import monai
import numpy as np
from PIL import Image

from logging import getLogger
import psycopg2
import torch
import torchvision
import pandas as pd
import os
import torch.distributed as dist

_GLOBAL_SEED = 0
logger = getLogger()



def make_cagdataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    args=None,
    phase='train'
):
    dataset= CAGDataset(args, transform, phase)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler

class CAGDataset(torch.utils.data.Dataset):
    def __init__(self, config, transforms, phase='train'):
        self.config = config
        self.transforms = transforms
        self.phase = phase
        self._construct_loader()
        # contrusct specific cag transforms
                
    def getDataFromDatabase(self, config):
        connection = psycopg2.connect(
            host=config['host'],
            database=config['database'],
            user=config['username'],
            password=config['password'])
        if self.phase == 'train':
            sql = config['query_pretraining'].replace(
                "?table_name", "\"" + config['table_name'] + "\"")
        elif self.phase == 'train_eval':
            sql = config['query_pretraining_eval_train'].replace(
                "?table_name", "\"" + config['table_name'] + "\"")
        elif self.phase == 'val_eval':
            sql = config['query_pretraining_eval_val'].replace(
                "?table_name", "\"" + config['table_name'] + "\"")
        else:
            raise ValueError('phase not supported')
        sql = sql.replace(
            "?schema_name", "\"" + config['schema_name'] + "\"")
        sql = sql.replace(
            "??", "\"")
        df = pd.read_sql_query(sql, connection)
        if len(df) == 0:
            print('The requested query does not have any data!')
        connection.close()
        # replicate dataframe df 100 times
      #  df = pd.concat([df]*4, ignore_index=True)
        return df

    def set_data_path(self, features):
        for feature in features:
            self.df[feature] = self.df[feature].apply(
                        lambda x: os.path.join(self.config['data']['root_path'], x))

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
        self.data = self.df[self.features + ['rowid', "labels_transformed"]]
        self.data = self.data.to_dict('records')
        self.data_part = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,)[dist.get_rank()]
        print('Constructing cag dataset')
        
    def __len__(self):
        return len(self.data_part)
    
    def __getitem__(self, index):
   #     try:
        sample = self.data_part[index]
        sample = self.transforms(sample)
        frames = sample[self.features[0]]
        return frames, (index, sample['labels_transformed'])
        # except:
        #     print('Error in loading data')
        #     return None, 0
        