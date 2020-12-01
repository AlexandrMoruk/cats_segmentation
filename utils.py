from list_classes import *
import pandas as pd
import torch
from torch.utils.data import Dataset

from catalyst.utils import imread
import albumentations as albu
from albumentations.pytorch import ToTensor
import numpy as np
import scipy.io as sio



class SegmentationDataset(Dataset):
    def __init__(self, path_to_csv="train.csv", masks=True, transforms=None) -> None:
        self.df = pd.read_csv(path_to_csv)
        self.transforms = transforms
        self.masks = masks

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.df['image'][idx]
        mask_path = self.df['mask'][idx]

        image = imread(image_path)

        if mask_path is np.nan:
            mask = self._create_empty_mask(image.shape[:2])
        else:
            mat_file = sio.loadmat(self.df['mask'][idx])
            mask = self._create_mask(mat_file)

        result = {"image": image}

        if self.masks is not None:
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)
            result['mask'] = torch.squeeze(result['mask']).permute(2, 0, 1)

        return result

    def _create_mask(self, mat_file):
        list_classes = mat_file['anno'][0][0][1][0]

        w, h = list_classes[0][2].shape
        out_array = np.zeros([18, w, h], dtype='int')
        out_array[0] = np.ones([w, h], dtype="int")

        insert_classes = [0]
        for k, i in enumerate(list_classes):
            if i[0][0] == 'cat':
                for i in list_classes[k][3][0]:
                    index = classes_map[i[0][0]]
                    out_array[index] += i[1].copy()
                    if index not in insert_classes:
                        insert_classes.append(index)

        out_array = out_array.clip(0, 1)
        insert_classes.sort()

        for i in range(len(insert_classes) - 1):
            for j in range(i + 1, len(insert_classes)):
                out_array[insert_classes[i]] -= out_array[insert_classes[j]]

        return np.transpose(out_array.clip(0, 1).astype(np.uint8), [1, 2, 0]) * 255

    def _create_empty_mask(self, size, n_class=18):
        h, w = size
        mask = np.zeros([h, w, n_class], dtype=np.uint8)
        mask[:, :, 0] = np.ones([h, w], dtype=np.uint8) * 255
        return mask


def pre_transforms(image_size=512):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=512):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])


    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():

    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


train_transforms = compose([
    pre_transforms(),
    resize_transforms(),
    hard_transforms(),
    post_transforms()
])
valid_transforms = compose([pre_transforms(), post_transforms()])

show_transforms = compose([resize_transforms(), hard_transforms()])