import argparse
import logging
import pprint
import time
import torch
import torch.nn.parallel
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

import _init_paths
_init_paths.add_path()
from lib.config import config
from lib.config.default import update_config
from lib.core.criterion.build import build_criterion
from lib.core.data.dataloader.build import build_dataloader
from lib.core.models.build import build_model
from lib.core.optimizers.build import build_optimizer
from lib.engine.test import test
from lib.engine.train import train_one_epoch
from lib.utils.utils import create_logger, fix_seed, init_distributed, setup_cudnn
from lib.utils.distributed import distributed
from tensorboardX import SummaryWriter


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


def train(args, config, writer):
    
    model = build_model(config, num_classes=config.MODEL.NUM_CLASSES)
    if args.distributed:
        device = torch.device("cuda:{}".format(args.local_rank))
    else:
        device = torch.device("cuda")
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )


    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)
    criterion = build_criterion(config)

    train_loader = build_dataloader(cfg=config, is_train=True)
    val_loader = build_dataloader(cfg=config, is_train=False)

    best_perf = 0.0
    
    logging.info('=> start training')
    start = time.time()
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):

        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer, epoch)

        if epoch % config.TRAIN.EVAL_BEGIN_EPOCH == 0:
            perf, _, _, _ = test(config, model, val_loader, epoch)
            best_perf = perf if perf > best_perf else best_perf
            
     #TODO: save checkpoints
    logging.info(
            '=> Training completed in: {:.2f}s'.format(time.time() - start)
        )       
            

if __name__ == '__main__':
    
    args = parse_args()
    init_distributed(args)
    
    setup_cudnn(config)
    update_config(config, args)
    fix_seed(config.seed)
    distributed.synchronize()
    
    log_dir = create_logger(config)
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    if distributed.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

    train(
        args=args,
        config=config,
        writer=writer,
    )
