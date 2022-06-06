import logging
import os
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
from .distributed import distributed

def init_distributed(args):

    args.num_gpus = torch.cuda.device_count()
    args.distributed = args.num_gpus > 1
    if args.distributed:
        print("=> init process group start")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(minutes=180)
        )
        distributed.local_rank = args.local_rank
        print("=> init process group end")


def setup_cudnn(config):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def create_logger(cfg, phase='train'):
    output_dir = Path(cfg.OUTPUT_DIR)
    print('=> creating {} ...'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=> setup logger ...')
    setup_logger(output_dir, phase)
    return str(output_dir)


def setup_logger(final_output_dir, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.txt'.format(phase, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + distributed.head + ' %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(head))
    logging.getLogger('').addHandler(console)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
