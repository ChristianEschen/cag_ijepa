# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    print('maybe look here')
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from ijepa.masks.multiblock import MaskCollator as MBMaskCollator
from ijepa.masks.utils import apply_masks
from ijepa.utils.distributed import (
    init_distributed,
    AllReduce
)
from ijepa.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from ijepa.utils.tensors import repeat_interleave_batch
from ijepa.datasets.imagenet1k import make_imagenet1k
from ijepa.datasets.cagdataset import make_cagdataset
from ijepa.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from ijepa.transforms import make_transforms, make_cag_transforms
from torch.utils.tensorboard import SummaryWriter
from ijepa.eval.knn import eval_knn_with_model
from ijepa.metrics import AccuracyAveraging
from ijepa.metrics import calc_rankme


# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def get_keys_with_suffix(d, suffix):
    return [value for key, value in d.items() if key.endswith(suffix)][0]

def get_autocast_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float
def getdataset(transform, batch_size, mask_collator, pin_mem, num_workers, world_size, rank, root_path, image_folder, copy_data, args):
    if args['data']['dataset'] == 'cag':
        _, unsupervised_loader_train, unsupervised_sampler_train = make_cagdataset(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True,
            args=args,
            phase='train')

    else:
        ValueError('dataset not supported')
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    return _, unsupervised_loader_train, unsupervised_sampler_train

def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    load_pretrained_checkpoint =  args['meta']['load_pretrained_checkpoint'] # boolean
    load_pretrained_model_path = args['meta']['load_pretrained_checkpoint_path']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    # if not torch.cuda.is_available():
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device('cuda:0')
    #     torch.cuda.set_device(device)
    init_ddp = args['init_ddp']

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    cpu = args['cpu']
    use_bfloat16 = args['meta']['use_bfloat16']
    if cpu:
        use_bfloat16 = False


    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']


    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    if init_ddp:
        torch.distributed.init_process_group(
                    backend="nccl" if cpu is False else "Gloo",
                    init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # generate timestamp for folder
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    tensorboard_folder = os.path.join(folder, 'tb_' + timestamp)

    if rank == 0:
        # create tensorboard folder if not exists
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
    local_rank = os.environ['LOCAL_RANK']
    if cpu:
        device = torch.device("cpu")  
        

    else:
        device = torch.device("cuda:"+str(local_rank))

        torch.cuda.set_device("cuda:"+str(local_rank))
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}_model.pt')
    latest_path = os.path.join(folder, 'model.pt')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    if args['data']['dataset'] == 'cag':
        transform = make_cag_transforms(
            crop_size=crop_size,
            crop_scale=crop_scale,
            gaussian_blur=use_gaussian_blur,
            horizontal_flip=use_horizontal_flip,
            color_distortion=use_color_distortion,
            color_jitter=color_jitter,
            keys=["DcmPathFlatten"] if args['data']['dataset']=='cag' else None,)
    else:
        transform = make_transforms(
            crop_size=crop_size,
            crop_scale=crop_scale,
            gaussian_blur=use_gaussian_blur,
            horizontal_flip=use_horizontal_flip,
            color_distortion=use_color_distortion,
            color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader_train, unsupervised_sampler_train = getdataset(
        transform, batch_size, mask_collator,
        pin_mem, num_workers, world_size,
        rank, root_path, image_folder, copy_data, args)
    ipe = len(unsupervised_loader_train)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, 
                                      device_ids=[device] if cpu is False else None,
                                      static_graph=True)
    predictor = DistributedDataParallel(predictor,
                                        device_ids=[device] if cpu is False else None,
                                        static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # always load from pretrained imagenet:
    if rank == 0:
        if load_pretrained_checkpoint:
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                    device=device,
                    r_path=load_pretrained_model_path,
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    opt=optimizer,
                    scaler=scaler)
            for _ in range(start_epoch*ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    # -- load training checkpoint
    # if rank == 0:
    #     if load_model:
    #         encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
    #             device=device,
    #             r_path=load_path,
    #             encoder=encoder,
    #             predictor=predictor,
    #             target_encoder=target_encoder,
    #             opt=optimizer,
    #             scaler=scaler)
    #         for _ in range(start_epoch*ipe):
    #             scheduler.step()
    #             wd_scheduler.step()
    #             next(momentum_scheduler)
    #             mask_collator.step()
    if load_model:
        for param in encoder.parameters():
            dist.broadcast(param.data, src=0)  # src=0 means we're broadcasting from rank 0

        for param in predictor.parameters():
            dist.broadcast(param.data, src=0)

    # if target_encoder is not None:
        for param in target_encoder.parameters():
            dist.broadcast(param.data, src=0)
        # epoch_tensor = torch.tensor(start_epoch, dtype=torch.int).to(device)
        # dist.broadcast(epoch_tensor, src=0)
        # start_epoch = epoch_tensor.item()
        
        # if scaler is not None:
        #     # The state_dict of the scaler has '_scale', '_growth_tracker', and possibly other keys.
        #     # You'll want to broadcast each tensor in the state.

        #     state_dict = scaler.state_dict()

        #     # Broadcast the '_scale' tensor
        #     dist.broadcast(state_dict["_scale"], src=0)

        #     # Broadcast the '_growth_tracker' tensor
        #     dist.broadcast(state_dict["_growth_tracker"], src=0)
        
        
        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         if p.requires_grad:
        #             state = optimizer.state[p]
        #             # For AdamW, you have 'step', 'exp_avg', and 'exp_avg_sq' 
        #             # as the primary state components.
        #             dist.broadcast(state['step'], src=0)
        #             dist.broadcast(state['exp_avg'], src=0)
        #             dist.broadcast(state['exp_avg_sq'], src=0)
        #             # For the AdamW variant, you might also have 'max_exp_avg_sq'.
        #             if 'max_exp_avg_sq' in state:
        #                 dist.broadcast(state['max_exp_avg_sq'], src=0)
                    
                
    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # tensorboard logger:
    if rank == 0:
        writer = SummaryWriter(log_dir=tensorboard_folder)

    # -- TRAINING LOOP
            # start timer
    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler_train.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        # validation
        #if epoch % 20 == 0:
       # with torch.inference_mode():
        with torch.no_grad():
            results_dict =eval_knn_with_model(
                args=args,
                model=target_encoder,
                output_dir=folder,
                nb_knn=(10,), #(10, 20, 100, 200)
                temperature=0.07,
                autocast_dtype=get_autocast_dtype("bf16") if cpu is False else torch.float,
                accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
                transform=transform,
                gather_on_cpu=True,
                batch_size=batch_size,
                num_workers=num_workers,
                n_per_class_list=[-1],
                n_tries=1,
                n_classes=2
            )
        if rank == 0:
            f1_score = get_keys_with_suffix(results_dict, "_F1-score")
            accuracy = get_keys_with_suffix(results_dict, "_Top 1")
            rankme = get_keys_with_suffix(results_dict, "rankme")
            writer.add_scalar('Validation Accuracy', accuracy, global_step=epoch)
            writer.add_scalar('Train rankme', rankme, global_step=epoch)
            writer.add_scalar('Validation F1_score_macro', f1_score, global_step=epoch)
                

        # training
        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader_train):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                labels = udata[1][1].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        # Compute rankme
                        # rankme = calc_rankme(h)
                        # torch.distributed.reduce(rankme, dst=0, op=torch.distributed.ReduceOp.SUM)
                        # if rank == 0:
                        #     rankme = rankme / dist.get_world_size()
                        #     writer.add_scalar('Rankme', rankme,
                        #                     global_step=epoch * len(unsupervised_loader_train) + itr)
                                
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                if cpu is False:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)
                else:
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)
                # copy loss value to new variable
                loss_value = loss
                torch.distributed.reduce(loss_value, dst=0, op=torch.distributed.ReduceOp.SUM)
                # only write if rank == 0
                if rank == 0:
                    loss_value = loss_value / dist.get_world_size()
                    writer.add_scalar('Training loss', loss_value, global_step=epoch * len(unsupervised_loader_train) + itr)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                # only logg every 10 epoch
              #  if epoch % 20 == 0:
                    
                    


                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            
            

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
    if rank == 0:
        writer.close()
    print('stopping time is', time.time() - start)


if __name__ == "__main__":
    main()
