import os
import torch
from PIL import Image
from typing import List
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split


class CustomDataset(Dataset):
    def __init__(self, file_path: List, transform=None, target_transform=None):
        self.file_path = file_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.file_path)

    def __getitem__(self, idx):
        path = self.file_path[idx]  # 根据索引index获取该图片
        image = Image.open(path)
        label = int(os.path.split(path)[0][-1])
        if self.transform:
            image = self.transform(image)
        else:
            pass
        if self.target_transform:
            label = self.target_transform(label)
        else:
            pass
        return image, label


def split_dataset(root: str, transform=None, train_ratio: float = 0.8, val_ratio: float = 0.1):
    data_dict = {}
    for dir_name in os.listdir(root):
        tmp_path = os.path.join(root, dir_name)
        for file_name in os.listdir(tmp_path):
            if dir_name not in data_dict:
                data_dict[dir_name] = []
            else:
                data_dict[dir_name] += [os.path.join(tmp_path, file_name)]
    path_dict = {'train': [], 'val': [], 'test': []}
    for label in data_dict:
        train_num = int(train_ratio*len(data_dict[label]))
        val_num = int(val_ratio*len(data_dict[label]))
        test_num = len(data_dict[label])-train_num-val_num
        lengths = [train_num, val_num, test_num]
        train_data, val_data, test_data = random_split(
            data_dict[label],
            lengths,
            torch.Generator().manual_seed(0)
        )
        path_dict['train'] += train_data
        path_dict['val'] += val_data
        path_dict['test'] += test_data
    return {
        split: CustomDataset(path_dict[split], transform) for split in ['train', 'val', 'test']
    }


if __name__ == '__main__':
    print(split_dataset('./data/mnist', transforms.ToTensor()))
