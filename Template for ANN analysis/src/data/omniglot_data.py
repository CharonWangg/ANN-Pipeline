import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision

from torchvision import transforms
from sklearn.model_selection import train_test_split

# TODO: Add support for Omniglot Dataset
class OmniglotData(data.Dataset):
    def __init__(self, data_dir=r'data/omniglot',
                 class_num=9,
                 ds_type="train",
                 aug=False,
                 train_size=0.8,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225),
                 seed=42):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = (self.ds_type == "train") and self.aug
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.
        # TODO: Check if the path_list is existed, if not, download
        data_dir = self.data_dir + '/cifar10'
        if not op.exists(data_dir):
            print('Downloading CIFAR10 dataset...')
            torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
            print('Done.')
        else:
            print('CIFAR10 dataset already exists.')

        # Build dataset for training/validation/testing
        # TODO: seems not very efficient to repetitively load the dataset for train and valid
        trans = self.augmentation()
        if self.ds_type == "train":
            dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=trans)
            n_train = int(len(dataset) * self.train_size)
            n_val = len(dataset) - n_train
            train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))
            self.dataset = train
        elif self.ds_type == "valid":
            dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=trans)
            n_train = int(len(dataset) * self.train_size)
            n_val = len(dataset) - n_train
            train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))
            self.dataset = val
        elif self.ds_type == "test":
            dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=trans)
            self.dataset = dataset

    def __len__(self):
        # get total length of 3 datasets
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def augmentation(self):
        trans = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.RandomCrop(128),
            transforms.Normalize(self.img_mean, self.img_std),
        ) if self.aug else transforms.ToTensor()
        return trans
