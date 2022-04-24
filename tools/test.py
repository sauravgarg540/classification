import argparse
import logging
import time

import torch
import torch.nn.parallel
import torch.optim

from lib.config import config
from lib.config.default import update_config
from lib.core.criterion.build import build_criterion
from lib.core.models.build import build_model
from lib.core.optimizers.build import build_optimizer
from lib.engine.test import test
from lib.engine.train import train_one_epoch


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
    model = build_model(config)
    model.to(torch.device('cuda'))
    criterion_eval = build_criterion(config)
    val_loader = None
    start = time.time()

    with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        accuracy, precision, recall, f1_score = test(
            config,
            model,
            val_loader,
            criterion_eval,
        )

        logging.info(
            '=> {} validate end, duration: {:.2f}s'.format(
                head, time.time() - val_start
            )
        )


def test(args):
    print("Inside_test")


if __name__ == '__main__':
    test()
