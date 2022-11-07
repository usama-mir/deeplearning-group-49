import sys
import argparse
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--directory_path", required=True, type=str)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--test_size", choices=range(0, 1), default=0.2, type=float)
args = parser.parse_args()


class CarSegmentationDataset(Dataset):
    def __init__(self, image_directory_one, transform=None):
        self.car_images_one = os.listdir(image_directory_one)
        self.directory_one = image_directory_one
        self.transform = transform

    def __len__(self):
        return len(self.car_images_one)

    def grayScaling(self, img):
        img_float32 = np.float32(img)
        gray_image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
        return gray_image

    def __getitem__(self, index):
        img_path = os.path.join(self.directory_one, self.car_images_one[index])
        X_temp = cv2.rotate(np.load(img_path)[:3].T, cv2.ROTATE_90_CLOCKWISE)
        X_train = self.grayScaling(X_temp)
        y_train = cv2.rotate(np.load(img_path)[3:4].T, cv2.ROTATE_90_CLOCKWISE)
        return X_train, y_train


dataset = CarSegmentationDataset(args.directory_path)
train, test = train_test_split(dataset, test_size=args.test_size)
train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle)
test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=args.shuffle)
train_features, train_labels = next(iter(train_dataloader))


print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


