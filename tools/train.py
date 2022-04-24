import argparse
import logging
import time

import torch
import torch.nn.parallel
import torch.optim

import __init_paths
from lib.config import config
from lib.config.default import update_config
from lib.core.criterion.build import build_criterion
from lib.core.data.dataloader.build import build_dataloader
from lib.core.models.build import build_model
from lib.core.optimizers.build import build_optimizer
from lib.engine.test import test
from lib.engine.train import train_one_epoch
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str
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
    update_config(config, args)
    create_logger(config)
    
    logging.info('=> config: {}'.format(config))
    
    model = build_model(config, num_classes=config.MODEL.NUM_CLASSES)
    model.to(torch.device('cuda'))

    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)
    criterion = build_criterion(config)
    criterion_eval = build_criterion(config)

    # TODO: add dataloader
    train_loader = build_dataloader(cfg=config, is_train=True)
    val_loader = build_dataloader(cfg=config, is_train=False)

    # lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)
    best_perf = 0.0

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):

        # train for one epoch
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer, epoch)

        if epoch % config.TRAIN.EVAL_BEGIN_EPOCH == 0:
            perf, _, _, _ = test(config, model, val_loader, criterion_eval, epoch)
            best_perf = perf if perf > best_perf else best_perf

    # TODO: add lr scheduler
    # TODO: save checkpoint
    # TODO: add distributed training


if __name__ == '__main__':
    train()
