# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os
import pprint
import sys
import yaml

#import submitit

from ijepa.train import main as app_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    "--local_rank", type=int,
    help="Local rank: torch.distributed.launch.")
parser.add_argument(
            "--local-rank", type=int,
            help="Local rank: torch.distributed.launch.")
parser.add_argument(
    "--num_workers", type=int, help="Number of workers.")
# add argument for use of cpu (boolean)
parser.add_argument("--cpu", action="store_true", help="Use CPU")
parser.add_argument("--init_ddp", action="store_true", help="init ddp")



class Trainer:

    def __init__(self, fname='configs.yaml', num_workers=0, cpu=False, load_model=None, init_ddp=False):
        self.fname = fname
        self.load_model = load_model
        self.num_workers = num_workers
        self.init_ddp = init_ddp
        self.cpu = cpu

    def __call__(self):
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {fname}')

        # -- load script params
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            # add cpu and num_workers to params
            params['cpu'] = self.cpu
            params['num_workers'] = self.num_workers
            params['init_ddp'] = self.init_ddp
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

        resume_preempt = False if load_model is None else load_model
        app_main(args=params, resume_preempt=resume_preempt)
        
    def __call_pipeline__(self, config):
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {fname}')

        config['cpu'] = self.cpu
        config['num_workers'] = self.num_workers
        config['init_ddp'] = self.init_ddp

        resume_preempt = False if load_model is None else load_model
        app_main(args=config, resume_preempt=resume_preempt)

    # def checkpoint(self):
    #     fb_trainer = Trainer(self.fname, True)
    #     return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch():
    config_fnames = [args.fname]
    for cf in config_fnames:
        fb_trainer = Trainer(cf, num_workers=args.num_workers, cpu=args.cpu, init_ddp=args.init_ddp)
        fb_trainer()
 
def launch_in_pipeline(config, num_workers, cpu, init_ddp):
    fb_trainer = Trainer(config, num_workers=num_workers, cpu=cpu, init_ddp=init_ddp)
    fb_trainer.__call_pipeline__(config)
    
if __name__ == '__main__':
    args = parser.parse_args()
    launch()

