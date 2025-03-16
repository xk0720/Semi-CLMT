import numpy as np
import random
import multiprocessing
import albumentations
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
from Data.dataset_RITE import SegDataset
from utils import sampler


def get_dataloaders(batch_size, semi_ratio=0.2):

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
        # transforms.Grayscale(),  # change 3(RGB channel) to 1(Grayscale)
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
        # transforms.Grayscale(),
    ])

    train_dataset = SegDataset(
        transformation_train_weak=transformation_train_weak,
        transformation_train_strong=transformation_train_strong,
        transformation_toTensor_Normalization=transformation_toTensor_Normalization,
        transformation_toTensor_Normalization_target=transformation_toTensor_Normalization_target,
        train=True,
    )

    test_dataset = SegDataset(
        transformation_valid_input=transformation_valid_input,
        transformation_valid_target=transformation_valid_target,
        train=False,
    )

    val_dataset = SegDataset(
        transformation_valid_input=transformation_valid_input,
        transformation_valid_target=transformation_valid_target,
        train=False,
    )

    # # 1. semi-supervised loss
    if semi_ratio < 1.0:

        length_train = len(train_dataset)  # the length of RITE train/test dataset is 20

        # randomly split the labeled / unlabeled data
        unlabeled_idxs, labeled_idxs = train_test_split(
            np.linspace(0, length_train - 1, length_train).astype("int"),
            test_size=int(length_train * semi_ratio),
            random_state=42,
        )
        batch_sampler = sampler.TwoStreamBatchSampler(primary_indices=labeled_idxs, secondary_indices=unlabeled_idxs,
                                                      batch_size=batch_size, secondary_batch_size=4)

        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            # num_workers=multiprocessing.Pool()._processes,
            num_workers=0,
        )

    # # 2. fully-supervised loss
    else:
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
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



