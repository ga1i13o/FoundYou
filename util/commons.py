import copy
import io
import logging
import os
import random
import sys
import traceback
from os.path import join

import numpy as np
import torch
from torch.nn import functional as F


def setup_logging(
    save_dir: str,
    console: str = "info",
    info_filename: str = "info.log",
    redirect_std: bool = True,
    rank: int = 0,
):
    """
    Logging setup (DDP-aware).
    - Only rank 0 writes to files and console.
    - Single file: info.log (INFO and above).
    - If redirect_std=True, print()/stderr are routed into logging on rank 0,
      and discarded on other ranks.
    """
    root = logging.getLogger()
    # avoid duplicate handlers if called twice
    if getattr(root, "_commons_logging_init_done", False):
        return
    root._commons_logging_init_done = True

    root.setLevel(logging.DEBUG)  # keep DEBUG level so console="debug" works

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")

        # single info file
        if info_filename:
            fh_info = logging.FileHandler(join(save_dir, info_filename), encoding='utf-8')
            fh_info.setLevel(logging.INFO)
            fh_info.setFormatter(fmt)
            root.addHandler(fh_info)

        # console handler
        if console is not None:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG if console == "debug" else logging.INFO)
            ch.setFormatter(fmt)
            root.addHandler(ch)

        # pipe uncaught exceptions to logs
        def _excepthook(tp, val, tb):
            root.error("\n" + "".join(traceback.format_exception(tp, val, tb)))
        sys.excepthook = _excepthook

        # optional: redirect stdout/stderr so print()/tqdm go to logs
        if redirect_std:
            class _StreamToLogger(io.TextIOBase):
                def __init__(self, logger, level):
                    self.logger, self.level = logger, level
                def write(self, buf):
                    for line in buf.rstrip().splitlines():
                        if line:
                            self.logger.log(self.level, line)
                    return len(buf)
                def flush(self): pass
            sys.stdout = _StreamToLogger(root, logging.INFO)

    else:
        # non-zero ranks: no files, no console; optionally silence std streams
        if redirect_std:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")


def make_deterministic(seed: int = 0):
    """Make results deterministic.
    Running the script in a deterministic way might slow it down.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model):
    if not checkpoint_path:
        print('No checkpoint provided.')
        return model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if list(checkpoint['model'].keys())[0].startswith('module'):
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        

    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded checkpoint from {checkpoint_path}.')

    return model


def resize_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Resize a mask to the model image size.

    Args:
        mask: Tensor [1, N, H, W].

    Returns:
        Boolean tensor of shape [1, N, IMG, IMG].
    """
    mask = F.interpolate(
        mask,
        (image_size, image_size),
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )
    return (mask.float() > 0)


def rescale_points(points: torch.Tensor, from_hw, to_hw):
    """
    points: (..., 2) tensor of (x, y) in pixels for an image of size from_hw=(H0,W0)
    returns: points rescaled to to_hw=(H1,W1)
    """
    H0, W0 = from_hw
    H1, W1 = to_hw
    scale = points.new_tensor([W1 / W0, H1 / H0])
    return points * scale
