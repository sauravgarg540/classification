import os.path as op

import yaml
from yacs.config import CfgNode as CN

from utils.common import comm

_C = CN()

_C.OUTPUT_DIR = './output'
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.VERBOSE = True
_C.WORKERS = 0

_C.AUG = CN(new_allowed=True)
_C.AUG.NORMALIZE = CN()
_C.AUG.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.AUG.NORMALIZE.STD = [0.229, 0.224, 0.225]
# data augmentation
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0 / 4.0, 4.0 / 3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.INTERPOLATION = 2  # data augmentation
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0 / 4.0, 4.0 / 3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.INTERPOLATION = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = False
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.SPEC = CN(new_allowed=True)

_C.LOSS = CN(new_allowed=True)
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'softmax'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'imagenet'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.LABELMAP = ''
_C.DATASET.SAMPLER = 'default'

_C.DATASET.TARGET_SIZE = -1

_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.USE_TRAIN_AUG = False
_C.FINETUNE.BASE_LR = 0.003
_C.FINETUNE.BATCH_SIZE = 512
_C.FINETUNE.EVAL_EVERY = 3000
_C.FINETUNE.TRAIN_MODE = True
_C.FINETUNE.FROZEN_LAYERS = []
_C.FINETUNE.LR_SCHEDULER = CN(new_allowed=True)
_C.FINETUNE.LR_SCHEDULER.DECAY_TYPE = 'step'

# train
_C.TRAIN = CN()
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SCALE_LR = True
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.WITHOUT_WD_LIST = []
_C.TRAIN.NESTEROV = True
# for adam
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100

_C.TRAIN.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EVAL_EPOCH = 1

_C.TRAIN.DETECT_ANOMALY = False

_C.TRAIN.CLIP_GRAD_NORM = 0.0
_C.TRAIN.SAVE_ALL_MODELS = False

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.CENTER_CROP = False
_C.TEST.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.TEST.INTERPOLATION = 2
_C.TEST.MODEL_FILE = ''
_C.TEST.REAL_LABELS = False
_C.TEST.VALID_LABELS = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, op.join(op.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    config.merge_from_list(args.opts)
    if config.TRAIN.SCALE_LR:
        config.TRAIN.LR *= comm.world_size
    config.freeze()


def save_config(cfg, path):
    if comm.is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())
