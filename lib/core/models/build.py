from lib.utils.registry import register_models


def build_model(cfg, **kwargs):
    model_name = cfg.MODEL.NAME
    if not register_models.is_available(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return register_models.entrypoints(model_name)(pretrained = cfg.MODEL.PRETRAINED, **kwargs)
