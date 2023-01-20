import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from cvbase.optflow.visualize import flow2rgb
from detectron2.data import detection_utils as d2_utils

__LOGGER = logging.Logger(__name__)
__TAR_SP = [Path('/usr/bin/tar'), Path('/bin/tar')]

TAG_FLOAT = 202021.25


def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def read_flow(sample_dir, resolution=None, to_rgb=False):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    if resolution:
        flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
        flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
        flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')


def read_rgb(sample_dir, resolution=None):
    rgb = d2_utils.read_image(sample_dir)
    rgb = ((rgb / 255.0) - 0.5) * 2.0
    if resolution:
        rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, -1., 1.)
    return einops.rearrange(rgb, 'h w c -> c h w')


### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
