import os

import torch
import torchvision
import torchvision.transforms as transforms

from . import lmdb_dataset
from . import torchvision_extension as transforms_extension
from .prefetch_data import fast_collate

class ImageNet12(object):

    def __init__(self, trainFolder, testFolder, num_workers=8, pin_memory=True, 
                size_images=224, scaled_size=256, 
                data_config=None, dist=False):

        self.data_config = data_config
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_dataset = self.data_config.patch_dataset
        self.dist = dist

        #images will be rescaled to match this size
        if not isinstance(size_images, int):
            raise ValueError('size_images must be an int. It will be scaled to a square image')
        self.size_images = size_images
        self.scaled_size = scaled_size

    def _getTransformList(self):

        list_of_transforms = []
        list_of_transforms.append(transforms.Resize(self.scaled_size))
        list_of_transforms.append(transforms.CenterCrop(self.size_images))
        
        return transforms.Compose(list_of_transforms)

    def _getTestSet(self):

        test_transform = self._getTransformList()
        # print(test_transform)
        if self.data_config.val_data_type == 'img':
            test_set = lmdb_dataset.ImageFolderInstance(self.testFolder, test_transform,
                                patch_dataset=self.patch_dataset)
        elif self.data_config.val_data_type == 'lmdb':
            test_set = lmdb_dataset.ImageFolderLMDBInstance(self.testFolder, test_transform,
                            patch_dataset=self.patch_dataset)
            self.test_num_examples = test_set.__len__()
        return test_set

    def getTestLoader(self, batch_size):
        
        test_set = self._getTestSet()

        if self.dist:
            sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, sampler=sampler,
            collate_fn=fast_collate)
        return test_loader
