import torch.nn as nn


def build_criterion(config):
    """
    Build criterion from config
    """
    criterion = None
    if config.LOSS.LOSS == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    return criterion
