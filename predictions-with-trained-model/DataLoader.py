'''
Imports
'''
import sys
import argparse
<<<<<<< Updated upstream
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
=======
from torch import dtype
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
# from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import torchvision
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import time
from tqdm import tqdm
# import matplotlib.pyplot as plt
from torchmetrics.functional import dice as pt_dice_score
import torch.optim as optim
from torchvision.transforms import ToTensor
import logging
from torchvision import transforms
from pathlib import Path
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
# --directory_path=C:\Users\tala1\Downloads\carseg_data\carseg_data\clean_data_test
from torch.utils.data.sampler import SubsetRandomSampler

# Determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--init_lr", default=0.001)
parser.add_argument("--num_epochs", default=40, type=int)
parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--weight_decay", default=1e-8, type=float)
>>>>>>> Stashed changes
parser.add_argument("--directory_path", required=True, type=str)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--test_size", choices=range(0, 1), default=0.2, type=float)
args = parser.parse_args()

<<<<<<< Updated upstream
=======
'''
Custom Segmentation Dataset Class
'''

>>>>>>> Stashed changes

class CarSegmentationDataset(Dataset):
    def __init__(self, image_directory_one, transform=None):
        self.car_images_one = os.listdir(image_directory_one)
        self.directory_one = image_directory_one
        self.transform = transform

    def __len__(self):
<<<<<<< Updated upstream
        return len(self.car_images_one)
=======
        return len(self.data_list)

    def __getitem__(self, idx):
        aNumpyFilePath = self.data_list[idx]
        # load npy array
        numpy_array = np.load(aNumpyFilePath)
        # get RGB image
        rgb_img = (np.transpose(numpy_array[:3], (1, 2, 0)) * 255).astype(float)
        # get grayscale maske
        mask_img = numpy_array[3]
        # get distinct classes list in image from mask
        # distinct_classes_list = list(set(mask_img.flatten()))
        # get filename
        # filename = aNumpyFilePath.stem
        # to tensor
        rgb_img = self.transform(rgb_img).type(torch.float)
        mask_img = self.transform_mask(mask_img).type(torch.int)
        # return {'image':rgb_img, 'mask':mask_img, 'distinct_classes':distinct_classes_list, 'filename':filename, 'path':aNumpyFilePath}
        return rgb_img, mask_img


'''
Splitting data
'''


def split_dataset(aPath, train_ratio=0.85, valid_ratio=0.15, seed_random=42, transform=None):
    """_summary_
    :param aPath: path to folder that contains all npy data files
    :type aPath: str
    :param aTestTXTFilenamesPath: path to txt file with test set filenames (in references/test_set_ids.txt)
    :type aTestTXTFilenamesPath: str
    :param train_ratio: percentage of training data, defaults to 0.85
    :type train_ratio: float, optional
    :param valid_ratio: percentage of validation data, defaults to 0.15
    :type valid_ratio: float, optional
    :param seed_random: random seed to use, defaults to 42
    :type seed_random: int, optional
    :return: train_dataset, valid_dataset, test_dataset
    :rtype: 3 DeloitteDataset objects
    """
    # set random np seed
    np.random.seed(seed_random)

    # building training, validation and test sets
    assert train_ratio + valid_ratio == 1.0

    # get test filenames list
    # test_ids = np.loadtxt(aTestTXTFilenamesPath, dtype=str)
    # test_filenames = np.array([f.split('.')[0] for f in test_ids])

    # get all data
    all_paths = [Path(p).absolute() for p in glob.glob(aPath + '/*')]

    # get filepaths lists
    train_valid_data_list = []
    for aPath in all_paths:
        train_valid_data_list.append(aPath)

        # get filename
        filename = aPath.stem
    #    if filename in test_filenames:
    #        test_data_list.append(aPath)
    #    else:

    # get test dataset
    # if transform == None:
    #   test_dataset = CarSegmentationDataset(test_data_list)
    # else:
    # test_dataset = CarSegmentationDataset(test_data_list, transform)

    # get train and valid datasets
    train_valid_data_list = np.array(train_valid_data_list)
    print("train_valid_data_list", train_valid_data_list)
    print(len(train_valid_data_list))
    permutation = np.random.permutation(len(train_valid_data_list))
    train_indices = permutation[:int(train_ratio * len(train_valid_data_list))]
    valid_indices = permutation[int(train_ratio * len(train_valid_data_list)):]
    print("train_indices", train_indices)
    print("valid_indices", valid_indices)
    if transform == None:
        train_dataset = CarSegmentationDataset(list(train_valid_data_list[train_indices]))
        valid_dataset = CarSegmentationDataset(list(train_valid_data_list[valid_indices]))
    else:
        train_dataset = CarSegmentationDataset(list(train_valid_data_list[train_indices]), transform)
        valid_dataset = CarSegmentationDataset(list(train_valid_data_list[valid_indices]), transform)
    print("type train", type(train_dataset))
    print("type valid", type(valid_dataset))
    return train_dataset, valid_dataset


recurrent_generator = torch.manual_seed(42)
train, valid = split_dataset(args.directory_path)


'''
DataLoader
'''
#train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)
#valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)
#train_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=args.drop_last, sampler=train_sampler)
#valid_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=args.drop_last, sampler=valid_sampler)
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)
valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)

'''
UNet implementation
Reference> https://github.com/milesial/Pytorch-UNet/tree/master/unet 
'''


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


'''
Load model
'''

model = UNet(n_channels=3, n_classes=2)
model.outc = OutConv(64, 9)
model = model.to(DEVICE)

'''
Loss function + optmizier
'''
# Set optimizer and scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)


def dice_score(preds, target, average_method='macro', mdmc_average='samplewise'):
    result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1],
                           mdmc_average=mdmc_average)
    result = result.cpu().detach().numpy()
    return result


'''
Training the Segmentation Model
'''
# Training loop
num_epochs = args.num_epochs
# validation_every_steps = cfg.hyperparameters.validation_every_steps
validation_every_steps = (len(train) // (10 * args.batch_size))
step = 0
cur_loss = 0

model.train()

train_dice_scores, valid_dice_score = [], []

max_valid_dice_score, best_model = None, None

for epoch in tqdm(range(num_epochs)):

    train_dice_scores_batches = []
    cur_loss = 0
    model.train()
    for rgb_img, mask_img in train_loader:
        # rgb_img, mask_img = aDictionary['image'], aDictionary['mask']
        rgb_img, mask_img = rgb_img.to(DEVICE), mask_img.to(DEVICE)

        # Forward pass, compute gradients, perform one training step.
        optimizer.zero_grad()

        output = model(rgb_img)

        batch_loss = loss_fn(
            output.flatten(start_dim=2, end_dim=len(output.size()) - 1),
            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1).type(torch.long)
        )
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss

        # Increment step counter
        step += 1

        # Compute DICE score.
        predictions = output.flatten(start_dim=2, end_dim=len(output.size()) - 1).softmax(1)
        train_dice_scores_batches.append(
            dice_score(
                predictions,
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1)
            )
        )

        if step % validation_every_steps == 0:

            # Append average training DICE score to list.
            train_dice_scores.append(np.mean(train_dice_scores_batches))

            train_dice_scores_batches = []
>>>>>>> Stashed changes

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


<<<<<<< Updated upstream
=======
print("Finished training.")
# Save model
model.load_state_dict(best_model)
>>>>>>> Stashed changes
