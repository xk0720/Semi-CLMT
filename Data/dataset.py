import random
from skimage.io import imread
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms.functional as TF


class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transformation_train_weak=None,
        transformation_train_strong=None,
        transformation_valid_input=None,
        transformation_valid_target=None,
        transformation_toTensor_Normalization=None,
        transformation_toTensor_Normalization_target=None,
        train=True,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transformation_train_weak = transformation_train_weak
        self.transformation_train_strong = transformation_train_strong
        self.transformation_valid_input = transformation_valid_input
        self.transformation_valid_target = transformation_valid_target
        self.transformation_toTensor_Normalization = transformation_toTensor_Normalization
        self.transformation_toTensor_Normalization_target = transformation_toTensor_Normalization_target
        self.train = train

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]
        
        # read the image and mask
        image, mask = imread(input_ID), imread(target_ID)

        if self.train:
            # 1. obtain the weak augmented view (both image and mask)
            transformed_weak = self.transformation_train_weak(image=image, mask=mask)
            img_weak = self.transformation_toTensor_Normalization(transformed_weak['image'])
            mask_weak = self.transformation_toTensor_Normalization_target(transformed_weak['mask'])
            
            # 2. obtain the strong augmented view (both image and mask)
            transformed_strong = self.transformation_train_strong(image=transformed_weak['image'],
                                                                  mask=transformed_weak['mask'])
            img_strong = self.transformation_toTensor_Normalization(transformed_strong['image'])
            mask_strong = self.transformation_toTensor_Normalization_target(transformed_strong['mask'])

            return img_weak, mask_weak, img_strong, mask_strong

        else:
            img = self.transformation_valid_input(image)
            mask = self.transformation_valid_target(mask)

            return img, mask
