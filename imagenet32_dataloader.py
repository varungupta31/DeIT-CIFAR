import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torchvision

_base_folder = 'Imagenet32_train'
_train_list = ['train_data_batch_1',
               'train_data_batch_2',
               'train_data_batch_3',
               'train_data_batch_4',
               'train_data_batch_5',
               'train_data_batch_6',
               'train_data_batch_7',
               'train_data_batch_8',
               'train_data_batch_9',
               'train_data_batch_10']
_val_list = ['val_data']
_label_file = 'map_clsloc.txt'

def get_imagenet32_labels(root):
    file = os.path.join(root, _base_folder, _label_file)
    return np.loadtxt(file, dtype=str)[:, 2].tolist()


def remap(old_array, mapping):
    new_array = np.copy(old_array)
    for k, v in mapping.items():
        new_array[old_array == k] = v
    return new_array

class ImageNet32(Dataset):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, exclude=None, remap_labels=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        # Now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for f in _train_list:
                file = os.path.join(self.root, _base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    self.train_labels += entry['labels']
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.train_labels = np.array(self.train_labels) - 1
        else:
            f = _val_list[0]
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.val_data = entry['data']
                self.val_labels = entry['labels']
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.val_labels = np.array(self.val_labels) - 1

        if exclude is not None:
            if self.train:
                include_idx = np.isin(self.train_labels, exclude, invert=True)
                self.train_data = self.train_data[include_idx]
                self.train_labels = self.train_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.train_labels))}
                    self.train_labels = remap(self.train_labels, mapping)

            else:
                include_idx = np.isin(self.val_labels, exclude, invert=True)
                self.val_data = self.val_data[include_idx]
                self.val_labels = self.val_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.val_labels))}
                    self.val_labels = remap(self.val_labels, mapping)

        if self.train:
            self.train_labels = self.train_labels.tolist()
        else:
            self.val_labels = self.val_labels.tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.val_data[index], self.val_labels[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.val_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
