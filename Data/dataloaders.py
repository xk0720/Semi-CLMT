import numpy as np
import random
import multiprocessing
import albumentations
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
from Data.dataset import SegDataset
from utils import sampler


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(input_paths, target_paths, batch_size, semi_ratio=0.25):
    
    # 1. construct a weak augmentation (based on albumentations)
    # process both image and mask (training data)
    transformation_train_weak = albumentations.Compose([
        albumentations.Resize(height=352, width=352),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    ])
    
    # 2. construct a strong augmentation (based on albumentations), 
    # process both image and mask (training data)
    transformation_train_strong = albumentations.Compose([
        albumentations.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.4, p=1.0),
        # albumentations.GaussianBlur(blur_limit=5, p=0.8),
        albumentations.Cutout(num_holes=10, max_h_size=10, max_w_size=10, fill_value=0, p=0.8),
    ])
    
    # 3. construct a augmentation following weak/strong (based on transforms)
    # process only image (training data)
    transformation_toTensor_Normalization = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 4. construct a augmentation following weak/strong (based on transforms)
    # process only mask (training data)
    transformation_toTensor_Normalization_target = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(), # change 3(RGB channel) to 1(Grayscale)
    ])

    # 5. construct a augmentation for valid data (image)
    transformation_valid_input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((352, 352), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 6. construct a augmentation for valid data (mask)
    transformation_valid_target = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((352, 352)),
        transforms.Grayscale(),
    ])

    # split data (training, valid and test data)
    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = SegDataset(
        input_paths=list(input_paths[i] for i in train_indices),
        target_paths=list(target_paths[i] for i in train_indices),
        train=True,
        transformation_train_weak=transformation_train_weak,
        transformation_train_strong=transformation_train_strong,
        transformation_toTensor_Normalization=transformation_toTensor_Normalization,
        transformation_toTensor_Normalization_target=transformation_toTensor_Normalization_target,
    )

    test_dataset = SegDataset(
        input_paths=list(input_paths[i] for i in test_indices),
        target_paths=list(target_paths[i] for i in test_indices),
        train=False,
        transformation_valid_input=transformation_valid_input,
        transformation_valid_target=transformation_valid_target,
    )

    val_dataset = SegDataset(
        input_paths=list(input_paths[i] for i in val_indices),
        target_paths=list(target_paths[i] for i in val_indices),
        train=False,
        transformation_valid_input=transformation_valid_input,
        transformation_valid_target=transformation_valid_target,
    )

    # create batch_sampler (based on labeled_idxs and unlabeled_idxs)
    labeled_idxs = list(range(int(len(train_indices) * semi_ratio)))
    unlabeled_idxs = list(range(int(len(train_indices) * semi_ratio), len(train_indices)))

    if semi_ratio < 1.0:
        if semi_ratio >= 0.125:
            batch_sampler = sampler.TwoStreamBatchSampler(primary_indices=labeled_idxs,
                                                          secondary_indices=unlabeled_idxs,
                                                          batch_size=batch_size,
                                                          secondary_batch_size=int(batch_size * (1 - semi_ratio)))
        else:
            # Option_2: semi-ratio too small
            batch_sampler = sampler.TwoStreamBatchSampler(primary_indices=labeled_idxs,
                                                          secondary_indices=unlabeled_idxs,
                                                          batch_size=batch_size, secondary_batch_size=6)

        # semi-supervised, batch_size
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            # drop_last=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=0,
        )

    else:
        # fully-supervised, batch_size
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=0,
        )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=multiprocessing.Pool()._processes,
        num_workers=0,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=multiprocessing.Pool()._processes,
        num_workers=0,
    )

    return train_dataloader, test_dataloader, val_dataloader



