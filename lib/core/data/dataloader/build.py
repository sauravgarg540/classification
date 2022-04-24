from ..dataset.build import build_dataset
from torch.utils.data import DataLoader


def build_dataloader(cfg, is_train=True, distributed=False):
    print("Building dataloader...")
    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = True

    dataset = build_dataset(cfg, is_train)

    # update with distributed training
    sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=sampler,
        drop_last=True if is_train else False,
    )

    return data_loader
