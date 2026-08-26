"""Datasets for SwinSC training: CIFAR10 (32x32) or Imagenette-160 (random 128x128 crops)."""
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_main import wpath   # the ONE configuration (standard 7.9)

#: Where the datasets live. Follows SWINSC_ROOT when it is set, and otherwise
#: the repository's own data/ directory, so a fresh clone needs no editing.
ROOT = wpath("data")


def _loader(ds, bs, train, workers):
    ctx = "fork" if (workers > 0 and os.name == "posix") else None
    return DataLoader(ds, bs, shuffle=train, num_workers=workers, pin_memory=True, drop_last=train,
                      multiprocessing_context=ctx, persistent_workers=workers > 0)


def get_loader(name, train, bs, img_size, workers=4):
    if name == "cifar":
        tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
        ds = datasets.CIFAR10(ROOT, train=train, download=True, transform=transforms.Compose(tf + [transforms.ToTensor()]))
    elif name == "imagenette":
        d = os.path.join(ROOT, "imagenette160", "train" if train else "val")
        tf = ([transforms.RandomCrop(img_size, pad_if_needed=True), transforms.RandomHorizontalFlip()] if train
              else [transforms.CenterCrop(img_size)])
        ds = datasets.ImageFolder(d, transforms.Compose(tf + [transforms.ToTensor()]))
    else:
        raise ValueError(name)
    return _loader(ds, bs, train, workers)
