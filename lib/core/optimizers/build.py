import torch.optim as optim

from lib.utils.registry import register_optimizers


def build_optimizer(config, model, **kwargs):
    """
    Build optimizer from config
    """
    optimizer_name = config.TRAIN.OPTIMIZER
    if not register_optimizers.is_available(optimizer_name):
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    return register_optimizers.entrypoints(optimizer_name)(config, model, **kwargs)


@register_optimizers.register('sgd')
def sgd(config, model, **kwargs):
    return optim.SGD(
        model.parameters(),
        lr=config.TRAIN.LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV,
    )


@register_optimizers.register('adam')
def adam(config, model, **kwargs):
    return optim.Adam(
        model.parameters(),
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WD,
    )
