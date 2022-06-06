import os.path as osp
import sys


def add_path():
    this_dir = osp.dirname(__file__)
    path = osp.join(this_dir, '..')
    if path not in sys.path:
        sys.path.insert(0, path)
