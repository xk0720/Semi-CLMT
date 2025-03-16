import random
from skimage.io import imread
import cv2
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import glob
import numpy as np

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class SegDataset(data.Dataset):
    def __init__(
        self,
        file_path="/home/kevin/PhD-datasets/GlaS/*",
        transformation_toTensor_Normalization=None,
        transformation_train_weak=None,
        transformation_train_strong=None,
        transformation_valid_input=None,
        transformation_valid_target=None,
        transformation_toTensor_Normalization_target=None,
        train=True,
    ):
        self.transformation_toTensor_Normalization = transformation_toTensor_Normalization
        self.transformation_train_weak = transformation_train_weak
        self.transformation_train_strong = transformation_train_strong
        self.transformation_valid_input = transformation_valid_input
        self.transformation_valid_target = transformation_valid_target
        self.transformation_toTensor_Normalization_target = transformation_toTensor_Normalization_target
        self.train = train

        # construct train and test images list
        all_image_list = sorted(glob.glob(file_path))
        all_image_list.remove('/home/kevin/PhD-datasets/GlaS/Grade.csv')
        self.train_image_list = []
        self.test_image_list = []
        for file_name in all_image_list:
            if 'train' in file_name:
                if 'anno' not in file_name:
                    self.train_image_list.append(file_name)
            else:
                if 'anno' not in file_name:
                    self.test_image_list.append(file_name)

    def __len__(self):
        if self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)

    def __getitem__(self, index: int):
        if self.train:
            input_ID = self.train_image_list[index]
        else:
            input_ID = self.test_image_list[index]

        target_ID = input_ID.replace('.bmp', '_anno.bmp')

        image = cv2.imread(input_ID) # (H, W, 3)
        mask = cv2.imread(target_ID, 0) # flag==0 means grayscale mode (H, W)

        image, mask = correct_dims(image, mask) # correct the dimension of image (H, W, 3) and mask (H, W, 1)

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
