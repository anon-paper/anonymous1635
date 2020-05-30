import os

import cv2
import torch
import logging
# import msgpack
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.datasets as dset
import math

# import lmdb
import six
from tqdm import tqdm, trange
# import pyarrow as pa
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import Sampler
import random


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


class Mem(dset.ImageFolder):
    def __init__(self, root):
        super(Mem, self).__init__(root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        # image = self.loader(path)
        image = cv2.imread(path)

        return image, target


class ImageFolderInstance(dset.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False, train=False, patch_dataset=False, _print=print):
        # root = '/home1/lcl_work/imagenet/val'
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.root = root
        self.train = train
        self.patch_dataset = patch_dataset
        self.length = self.__len__()
        if patch_dataset: self.load_images(_print)

    def load_images(self, _print=None):
        self.images = []
        self.targets = []
        # dataset = dset.ImageFolder(self.root)
        dataset = dset.ImageFolder(self.root, loader=raw_reader)
        data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x)

        # for i, data in enumerate(data_loader):
        for i, data in enumerate(tqdm(data_loader)):
            image, label = data[0]
            self.images.append(image)
            # self.images.append(np.array(image, dtype=np.uint8))
            self.targets.append(label)

            if i % 1e5 == 0:
                if self.train:
                    logging.info("Loading training images: Step {:03d}/{:03d}".format(i, len(self.imgs)))
                else:
                    logging.info("Loading testing images: Step {:03d}/{:03d}".format(i, len(self.imgs)))
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        if self.patch_dataset:
            target = self.targets[index]
            image = self.images[index]

            buf = six.BytesIO()
            buf.write(image)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
        else:
            # path, target = Image.fromarray(self.imgs[index])
            path, target = self.imgs[index]
            image = self.loader(path)

        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop and self.train:
            img2 = self.transform(image)
            img = [img, img2]
            # img = torch.cat([img, img2], dim=0)

        return img, target



class DistSubRandSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, indices, num_replicas=None, rank=None):
        self.indices = indices
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]

        # print('!!!!') # 1024934
        # print(int(math.ceil(len(self.indices) * 1.0 / self.num_replicas) * self.num_replicas))
        # print(len(self.indices))
        # print(len(indices))
        # print(self.total_size)
        # print()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_images(root, train=True):
    images = []
    targets = []
    # dataset = dset.ImageFolder(root)
    dataset = dset.ImageFolder(root, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x)

    # for i, data in enumerate(data_loader):
    for i, data in enumerate(tqdm(data_loader)):
        image, label = data[0]
        images.append(image)
        # images.append(np.array(image, dtype=np.uint8))
        targets.append(label)

        if i % 1e5 == 0:
            if train:
                print("Loading training images: Step {:03d}/{:03d}".format(i, len(dataset.imgs)))
            else:
                print("Loading testing images: Step {:03d}/{:03d}".format(i, len(dataset.imgs)))
    
    return images, targets