import os
import random
from collections import OrderedDict
import json

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import ReduceOp
from dataset.dataset import Seq_action

import utils
from embedding_support import annotation_hashes, write_json
from model.helpers import get_lr_schedule_with_warmup
from pdpp_runtime import build_pdpp_diffusion, checkpoint_payload, load_pdpp_checkpoint, save_pdpp_checkpoint, unwrap_model
from utils import *
from utils.args import get_args
import numpy as np
from model.helpers import Logger


def reduce_tensor(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_text_tensor(action_pool, action_text_dict):
    text_list = []
    for action in action_pool:
        text_embedding = torch.tensor(action_text_dict[action])
        text_list.append(text_embedding)
    text_tensor = torch.cat([tensor for tensor in text_list], dim=0)
    return text_tensor


def main():
    print("Cuda support:", torch.cuda.is_available(), ":", torch.cuda.device_count(), "devices")
    args = get_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    split = args.split
    T = args.horizon
    feat = args.feat
    is_pad = args.is_pad
    root_dir = os.path.dirname(__file__) + '../../data'

    with open('data/task_info.json') as f:
        task_info = json.load(f)
    with open('data/base_action_pool_' + str(split) + '.json') as f:
        train_action_pool = json.load(f)
    with open('data/novel_action_pool_' + str(split) + '.json') as f:
        test_action_pool = json.load(f)
    with open('data/total_action_pool.json') as f:
        total_action_pool = json.load(f)

    if feat == 's3d':
        with open('data/s3d_action_feat_dict.json') as f:
            actions_text_dict = json.load(f)
        args.action_dim = 512
    elif feat == 'videoclip':
        with open('data/vc_action_feat_dict.json') as f:
            actions_text_dict = json.load(f)
        args.action_dim = 768
    else:
        assert 0

    args.observation_dim = args.action_dim * 3

    train_text_tensor = get_text_tensor(train_action_pool, actions_text_dict).cuda()
    test_text_tensor = get_text_tensor(test_action_pool, actions_text_dict).cuda()
    total_text_tensor = get_text_tensor(total_action_pool, actions_text_dict).cuda()

    train_train_dataset = Seq_action(root='data/', split=split, feat=feat, T=T, is_pad=is_pad,
                                     is_total=0, is_val=0)
    test_novel_dataset = Seq_action(root='data/', split=split, feat=feat, T=T, is_pad=is_pad,
                                    is_total=0, is_val=1)
    test_base_dataset = Seq_action(root='data/', split=split, feat=feat, T=T, is_pad=is_pad,
                                   is_total=0, is_val=2)
    train_val_dataset = Seq_action(root='data/', split=split, feat=feat, T=T, is_pad=is_pad,
                                   is_total=0, is_val=3)

    # Data loading code

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(train_val_dataset)
        test_base_sampler = torch.utils.data.distributed.DistributedSampler(test_base_dataset)
        test_novel_sampler = torch.utils.data.distributed.DistributedSampler(test_novel_dataset)
    else:
        train_sampler = None
        val_sampler = None
        test_base_sampler = None
        test_novel_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )

    test_base_loader = torch.utils.data.DataLoader(
        test_base_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_base_sampler,
    )

    test_novel_loader = torch.utils.data.DataLoader(
        test_novel_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_novel_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=val_sampler,
    )

    # create model
    diffusion_model = build_pdpp_diffusion(args)

    model = utils.Trainer(diffusion_model, train_loader, None, None, None, args.ema_decay, args.lr,
                          args.gradient_accumulate_every,
                          args.step_start_ema, args.update_ema_every, args.log_freq)

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            model.model.cuda(args.gpu)
            model.ema_model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, device_ids=[args.gpu], find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(
                model.ema_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.model.cuda()
            model.ema_model.cuda()
            model.model = torch.nn.parallel.DistributedDataParallel(model.model, find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(model.ema_model,
                                                                        find_unused_parameters=True)

    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.ema_model = torch.nn.DataParallel(model.ema_model).cuda()

    scheduler = get_lr_schedule_with_warmup(model.optimizer, int(args.n_train_steps * args.epochs))

    if not args.checkpoint_dir:
        raise ValueError('--checkpoint_dir is required for a fresh, reproducible PDPP run')
    checkpoint_dir = os.path.join(args.checkpoint_root, args.checkpoint_dir)
    if args.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    tb_logdir = os.path.join(args.log_root, args.checkpoint_dir)
    if args.rank == 0:
        os.makedirs(tb_logdir, exist_ok=True)
        tb_logger = Logger(tb_logdir)
        tb_logger.log_info(args)
        write_json(
            os.path.join(checkpoint_dir, 'run_metadata.json'),
            {
                'fresh_initialization': not args.resume,
                'args': vars(args).copy(),
                'annotation_hashes': annotation_hashes('data', args.split, args.feat),
            },
        )
    if args.resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'last.pt')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'--resume requested but {checkpoint_path} does not exist')
        checkpoint = load_pdpp_checkpoint(checkpoint_path, torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        unwrap_model(model.model).load_state_dict(checkpoint['model'])
        unwrap_model(model.ema_model).load_state_dict(checkpoint['ema'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        model.step = checkpoint['step']
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.rank == 0:
            log(f"=> resumed checkpoint '{checkpoint_path}' at epoch {checkpoint['epoch']}", args)

    if args.cudnn_benchmark:
        raise ValueError('cudnn_benchmark is incompatible with reproducible Experiment 4 PDPP runs')
    total_batch_size = args.world_size * args.batch_size
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ), args
    )

    max_eva = -1
    max_acc = -1
    # old_max_epoch = 0
    # save_max = os.path.join(os.path.dirname(__file__), 'save_max')

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if (epoch + 1) % 1 == 0:  # calculate on training set
            losses1, acc_top11, acc_top51, trajectory_success_rate_meter1, MIoU1_meter1, MIoU2_meter1, \
            acc_a01, acc_aT1, = model.train(args.n_train_steps, True, args, scheduler, train_text_tensor)

            losses_reduced1 = reduce_tensor(losses1.cuda()).item()
            acc_top1_reduced1 = reduce_tensor(acc_top11.cuda()).item()
            acc_top5_reduced1 = reduce_tensor(acc_top51.cuda()).item()
            trajectory_success_rate_meter_reduced1 = reduce_tensor(trajectory_success_rate_meter1.cuda()).item()
            MIoU1_meter_reduced1 = reduce_tensor(MIoU1_meter1.cuda()).item()
            MIoU2_meter_reduced1 = reduce_tensor(MIoU2_meter1.cuda()).item()
            acc_a0_reduced1 = reduce_tensor(acc_a01.cuda()).item()
            acc_aT_reduced1 = reduce_tensor(acc_aT1.cuda()).item()

            if args.rank == 0:
                print('lrs:')
                for p in model.optimizer.param_groups:
                    print(p['lr'])
                print('---------------------------------')
                logs = OrderedDict()
                logs['Train1/EpochLoss'] = losses_reduced1
                logs['Train1/EpochAcc@1'] = acc_top1_reduced1
                logs['Train1/EpochAcc@5'] = acc_top5_reduced1
                logs['Train1/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced1
                logs['Train1/MIoU1'] = MIoU1_meter_reduced1
                logs['Train1/MIoU2'] = MIoU2_meter_reduced1
                logs['Train1/acc_a0'] = acc_a0_reduced1
                logs['Train1/acc_aT'] = acc_aT_reduced1

                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
        else:
            losses1 = model.train(args.n_train_steps, False, args, scheduler, train_text_tensor)
            losses_reduced1 = reduce_tensor(losses1.cuda()).item()
            if args.rank == 0:
                print('lrs:')
                for p in model.optimizer.param_groups:
                    print(p['lr'])
                print('---------------------------------')

                logs = OrderedDict()
                logs['Train1/EpochLoss'] = losses_reduced1
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()

        validation_metrics = {}
        if ((epoch + 1) % 1 == 0) and args.evaluate:  # or epoch >= 10
            torch.manual_seed(args.sampling_seed)
            torch.cuda.manual_seed_all(args.sampling_seed)
            acc_top1_reduced1 = 0.
            acc_top5_reduced1 = 0.
            trajectory_success_rate_meter_reduced1 = 0.
            MIoU1_meter_reduced1 = 0.
            MIoU2_meter_reduced1 = 0.
            acc_a0_reduced1 = 0.
            acc_aT_reduced1 = 0.

            for times in range(1):
                acc_top11_t, acc_top51_t, \
                trajectory_success_rate_meter1_t, MIoU1_meter1_t, MIoU2_meter1_t, \
                acc_a01_t, acc_aT1_t = validate(val_loader, None, None, None, model.ema_model, args, train_text_tensor)

                acc_top1_reduced1_t = reduce_tensor(acc_top11_t.cuda()).item()
                acc_top5_reduced1_t = reduce_tensor(acc_top51_t.cuda()).item()
                trajectory_success_rate_meter_reduced1_t = reduce_tensor(trajectory_success_rate_meter1_t.cuda()).item()
                MIoU1_meter_reduced1_t = reduce_tensor(MIoU1_meter1_t.cuda()).item()
                MIoU2_meter_reduced1_t = reduce_tensor(MIoU2_meter1_t.cuda()).item()
                acc_a0_reduced1_t = reduce_tensor(acc_a01_t.cuda()).item()
                acc_aT_reduced1_t = reduce_tensor(acc_aT1_t.cuda()).item()

                acc_top1_reduced1 += acc_top1_reduced1_t
                acc_top5_reduced1 += acc_top5_reduced1_t
                trajectory_success_rate_meter_reduced1 += trajectory_success_rate_meter_reduced1_t
                MIoU1_meter_reduced1 += MIoU1_meter_reduced1_t
                MIoU2_meter_reduced1 += MIoU2_meter_reduced1_t
                acc_a0_reduced1 += acc_a0_reduced1_t
                acc_aT_reduced1 += acc_aT_reduced1_t

            if args.rank == 0:
                logs = OrderedDict()
                logs['Val1/EpochAcc@1'] = acc_top1_reduced1
                logs['Val1/EpochAcc@5'] = acc_top5_reduced1
                logs['Val1/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced1
                logs['Val1/MIoU1'] = MIoU1_meter_reduced1
                logs['Val1/MIoU2'] = MIoU2_meter_reduced1
                logs['Val1/acc_a0'] = acc_a0_reduced1
                logs['Val1/acc_aT'] = acc_aT_reduced1

                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
            trajectory_success_rate_meter_reduced = trajectory_success_rate_meter_reduced1
            acc_top1_reduced = acc_top1_reduced1
            validation_metrics = {
                'sr': trajectory_success_rate_meter_reduced,
                'acc': acc_top1_reduced,
                'acc_top5': acc_top5_reduced1,
                'miou1': MIoU1_meter_reduced1,
                'miou2': MIoU2_meter_reduced1,
            }
            print(trajectory_success_rate_meter_reduced, acc_top1_reduced)
            is_best = (trajectory_success_rate_meter_reduced, acc_top1_reduced) > (max_eva, max_acc)
            if is_best:
                max_eva = trajectory_success_rate_meter_reduced
                max_acc = acc_top1_reduced
        if args.rank == 0:
            payload = checkpoint_payload(
                epoch=epoch + 1,
                model=model.model,
                ema_model=model.ema_model,
                optimizer=model.optimizer,
                scheduler=scheduler,
                step=model.step,
                args=args,
                validation_metrics=validation_metrics,
                tb_logdir=tb_logdir,
            )
            save_pdpp_checkpoint(os.path.join(checkpoint_dir, 'last.pt'), payload)
            if validation_metrics and is_best:
                save_pdpp_checkpoint(os.path.join(checkpoint_dir, 'best.pt'), payload)

        if ((epoch + 1) % 1 == 0) and args.evaluate and args.test_during_training:
            acc_top1_reduced1 = 0.
            acc_top5_reduced1 = 0.
            trajectory_success_rate_meter_reduced1 = 0.
            MIoU1_meter_reduced1 = 0.
            MIoU2_meter_reduced1 = 0.
            acc_a0_reduced1 = 0.
            acc_aT_reduced1 = 0.

            for times in range(1):
                acc_top11_t, acc_top51_t, \
                trajectory_success_rate_meter1_t, MIoU1_meter1_t, MIoU2_meter1_t, \
                acc_a01_t, acc_aT1_t = validate(test_base_loader, None, None, None, model.ema_model, args, train_text_tensor)

                acc_top1_reduced1_t = reduce_tensor(acc_top11_t.cuda()).item()
                acc_top5_reduced1_t = reduce_tensor(acc_top51_t.cuda()).item()
                trajectory_success_rate_meter_reduced1_t = reduce_tensor(trajectory_success_rate_meter1_t.cuda()).item()
                MIoU1_meter_reduced1_t = reduce_tensor(MIoU1_meter1_t.cuda()).item()
                MIoU2_meter_reduced1_t = reduce_tensor(MIoU2_meter1_t.cuda()).item()
                acc_a0_reduced1_t = reduce_tensor(acc_a01_t.cuda()).item()
                acc_aT_reduced1_t = reduce_tensor(acc_aT1_t.cuda()).item()

                acc_top1_reduced1 += acc_top1_reduced1_t
                acc_top5_reduced1 += acc_top5_reduced1_t
                trajectory_success_rate_meter_reduced1 += trajectory_success_rate_meter_reduced1_t
                MIoU1_meter_reduced1 += MIoU1_meter_reduced1_t
                MIoU2_meter_reduced1 += MIoU2_meter_reduced1_t
                acc_a0_reduced1 += acc_a0_reduced1_t
                acc_aT_reduced1 += acc_aT_reduced1_t

            if args.rank == 0:
                logs = OrderedDict()
                logs['Test_base/EpochAcc@1'] = acc_top1_reduced1
                logs['Test1_base/EpochAcc@5'] = acc_top5_reduced1
                logs['Test1_base/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced1
                logs['Test1_base/MIoU1'] = MIoU1_meter_reduced1
                logs['Test1_base/MIoU2'] = MIoU2_meter_reduced1
                logs['Test1_base/acc_a0'] = acc_a0_reduced1
                logs['Test1_base/acc_aT'] = acc_aT_reduced1

                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
            trajectory_success_rate_meter_reduced = trajectory_success_rate_meter_reduced1
            acc_top1_reduced = acc_top1_reduced1
            print(trajectory_success_rate_meter_reduced, acc_top1_reduced)

        if ((epoch + 1) % 1 == 0) and args.evaluate and args.test_during_training:
            acc_top1_reduced1 = 0.
            acc_top5_reduced1 = 0.
            trajectory_success_rate_meter_reduced1 = 0.
            MIoU1_meter_reduced1 = 0.
            MIoU2_meter_reduced1 = 0.
            acc_a0_reduced1 = 0.
            acc_aT_reduced1 = 0.

            for times in range(1):
                acc_top11_t, acc_top51_t, \
                trajectory_success_rate_meter1_t, MIoU1_meter1_t, MIoU2_meter1_t, \
                acc_a01_t, acc_aT1_t = validate(test_novel_loader, None, None, None, model.ema_model, args, test_text_tensor)

                acc_top1_reduced1_t = reduce_tensor(acc_top11_t.cuda()).item()
                acc_top5_reduced1_t = reduce_tensor(acc_top51_t.cuda()).item()
                trajectory_success_rate_meter_reduced1_t = reduce_tensor(trajectory_success_rate_meter1_t.cuda()).item()
                MIoU1_meter_reduced1_t = reduce_tensor(MIoU1_meter1_t.cuda()).item()
                MIoU2_meter_reduced1_t = reduce_tensor(MIoU2_meter1_t.cuda()).item()
                acc_a0_reduced1_t = reduce_tensor(acc_a01_t.cuda()).item()
                acc_aT_reduced1_t = reduce_tensor(acc_aT1_t.cuda()).item()

                acc_top1_reduced1 += acc_top1_reduced1_t
                acc_top5_reduced1 += acc_top5_reduced1_t
                trajectory_success_rate_meter_reduced1 += trajectory_success_rate_meter_reduced1_t
                MIoU1_meter_reduced1 += MIoU1_meter_reduced1_t
                MIoU2_meter_reduced1 += MIoU2_meter_reduced1_t
                acc_a0_reduced1 += acc_a0_reduced1_t
                acc_aT_reduced1 += acc_aT_reduced1_t

            if args.rank == 0:
                logs = OrderedDict()
                logs['Test1_novel/EpochAcc@1'] = acc_top1_reduced1
                logs['Test1_novel/EpochAcc@5'] = acc_top5_reduced1
                logs['Test1_novel/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced1
                logs['Test1_novel/MIoU1'] = MIoU1_meter_reduced1
                logs['Test1_novel/MIoU2'] = MIoU2_meter_reduced1
                logs['Test1_novel/acc_a0'] = acc_a0_reduced1
                logs['Test1_novel/acc_aT'] = acc_aT_reduced1

                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
            trajectory_success_rate_meter_reduced = trajectory_success_rate_meter_reduced1
            acc_top1_reduced = acc_top1_reduced1
            print(trajectory_success_rate_meter_reduced, acc_top1_reduced)


def log(output, args):
    os.makedirs(args.log_root, exist_ok=True)
    with open(os.path.join(args.log_root, args.checkpoint_dir + '.txt'), "a") as f:
        f.write(output + '\n')


if __name__ == "__main__":
    main()

