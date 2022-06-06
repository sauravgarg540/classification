import argparse
import logging
import time
import pprint

import torch
import torch.nn.parallel
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

import __init_paths
from lib.config import config
from lib.config.default import update_config
from lib.core.criterion.build import build_criterion
from lib.core.data.dataloader.build import build_dataloader
from lib.core.models.build import build_model
from lib.core.optimizers.build import build_optimizer
from lib.engine.test import test
from lib.engine.train import train_one_epoch
from utils.utils import create_logger, fix_seed, init_distributed
from utils.common import comm


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str
    )
    
    parser.add_argument(
        '--seed', help='set seed for reproducibility of results', type=int, default=20
    )

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args


def train():

    args = parse_args()
    fix_seed(args.seed)
    init_distributed(args)
    comm.synchronize()
    
    update_config(config, args)
    create_logger(config)
    
    if comm.is_main_process():    
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))
    
    model = build_model(config, num_classes=config.MODEL.NUM_CLASSES)
    model.to(torch.device('cuda'))

    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)
    criterion = build_criterion(config)
    criterion_eval = build_criterion(config)

    train_loader = build_dataloader(cfg=config, is_train=True)
    val_loader = build_dataloader(cfg=config, is_train=False)

    best_perf = 0.0
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):

        # train for one epoch
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer, epoch)

        if epoch % config.TRAIN.EVAL_BEGIN_EPOCH == 0:
            perf, _, _, _ = test(config, model, val_loader, criterion_eval, epoch)
            best_perf = perf if perf > best_perf else best_perf

    # TODO: save checkpoint
    # TODO: add distributed training


if __name__ == '__main__':
    train()
