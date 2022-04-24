import logging
import os

import torch
import torch.utils.data
import torchvision.datasets as datasets
from ..transforms.build import build_transforms


def build_dataset(cfg, is_train):
    dataset = None
    if 'imagenet' in cfg.DATASET.DATASET:
        dataset = _build_imagenet_dataset(cfg, is_train)
    else:
        raise ValueError('Unkown dataset: {}'.format(cfg.DATASET.DATASET))
    return dataset


def _build_imagenet_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)
    dataset_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DATASET)
    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(os.path.join(dataset_path, dataset_name), transforms)

    return dataset
