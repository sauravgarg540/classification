import logging

import torchvision.transforms as T
from timm.data import create_transform
from .transforms import *


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96 * 96 else (512, 480)


def build_transforms(cfg, is_train=True):

    # assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)),
    # 'DATASET.OUTPUT_SIZE should be list or tuple'

    transforms = None
    normalize = T.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)

    if is_train:
        if cfg.FINETUNE.FINETUNE and not cfg.FINETUNE.USE_TRAIN_AUG:
            # precrop, crop = get_resolution(cfg.TRAIN.IMAGE_SIZE)
            crop = cfg.TRAIN.IMAGE_SIZE[0]
            precrop = crop + 32
            transforms = T.Compose(
                [
                    T.Resize((precrop, precrop), interpolation=cfg.AUG.INTERPOLATION),
                    T.RandomCrop((crop, crop)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            aug = cfg.AUG
            scale = aug.SCALE
            ratio = aug.RATIO
            ts = [
                T.RandomResizedCrop(
                    cfg.TRAIN.IMAGE_SIZE[0],
                    scale=scale,
                    ratio=ratio,
                    interpolation=cfg.AUG.INTERPOLATION,
                ),
                T.RandomHorizontalFlip(),
            ]

            cj = aug.COLOR_JITTER
            if cj[-1] > 0.0:
                ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

            gs = aug.GRAY_SCALE
            if gs > 0.0:
                ts.append(T.RandomGrayscale(gs))

            gb = aug.GAUSSIAN_BLUR
            if gb > 0.0:
                ts.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=gb))

            ts.append(T.ToTensor())
            ts.append(normalize)

            transforms = T.Compose(ts)
    else:
        if cfg.TEST.CENTER_CROP:
            transforms = T.Compose(
                [
                    T.Resize(
                        int(cfg.TEST.IMAGE_SIZE[0] / 0.875),
                        interpolation=cfg.TEST.INTERPOLATION,
                    ),
                    T.CenterCrop(cfg.TEST.IMAGE_SIZE[0]),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            transforms = T.Compose(
                [
                    T.Resize(
                        (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0]),
                        interpolation=cfg.TEST.INTERPOLATION,
                    ),
                    T.ToTensor(),
                    normalize,
                ]
            )

    return transforms
