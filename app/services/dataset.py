"""
    Create a dataset from images and labels and provide a dataloader
"""

import os
from glob import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import cv2


class carImageDataset(Dataset):
    def __init__(self, img_path_list, label_path_list):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list


def img_path_train_test_val_split(img_dir, label_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
        Splitting the dataset into train, val and test datasets.
        Useful for training and evaluation purposes.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing labels.
            train_ratio (float): Ratio of training data.
            val_ratio (float): Ratio of validation data.
            test_ratio (float): Ratio of testing data.
        Returns:
            train_img_list (list): List of training image paths.
            val_img_list (list): List of validation image paths.
            test_img_list (list): List of testing image paths.
            train_label_list (list): List of training label paths.
            val_label_list (list): List of validation label paths.
            test_label_list (list): List of testing label paths.
    """

    img_path_list = glob(os.path.join(img_dir, "*.png"))
    label_path_list = glob(os.path.join(label_dir, "*.txt"))
    
    # Ensure that the images and labels are sorted in the same order
    # The images and labels are supposed to have the same name. If not, good luck
    img_path_list.sort()
    label_path_list.sort()
    
    assert len(img_path_list) == len(label_path_list), "Number of images and labels do not match"

    # Split the dataset into train and test+val
    train_img_list, test_img_list, train_label_list, test_label_list = train_test_split(
        img_path_list, label_path_list, test_size=(val_ratio + test_ratio), random_state=42
    )
    # Split the test+val into test and val
    val_img_list, test_img_list, val_label_list, test_label_list = train_test_split(
        test_img_list, test_label_list, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
    )

    return train_img_list, val_img_list, test_img_list, train_label_list, val_label_list, test_label_list

if __name__ == "__main__":
    # Example usage
    img_dir = os.path.join("datasets", "training", "image_2")
    label_dir = os.path.join("datasets", "labels", "training", "label_2")
    
    train_img_list, val_img_list, test_img_list, train_label_list, val_label_list, test_label_list = img_path_train_test_val_split(
        img_dir, label_dir
    )
    
    print("Train images:", train_img_list[:5])
    print("Validation images:", val_img_list[:5])
    print("Test images:", test_img_list[:5])
    print("Train labels:", train_label_list[:5])
    print("Validation labels:", val_label_list[:5])
    print("Test labels:", test_label_list[:5])