import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torchvision.transforms import *
import torch

import segmentation_models_pytorch as smp
from args import get_args
import segmentation_models_pytorch.utils.losses as smpLoss

from car_dataset import CarDataset

args = get_args()

validation_path = args.validation_path  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/validation/"
train_path = args.train_path  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/train/"
test_path = args.test_path  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/test/"

transform = transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomApply(transforms=[
        RandomResizedCrop(size=(256, 256), scale=(0.40, 1.0)),
    ], p=0.4),
    #      transforms.RandomApply(transforms=[
    #          GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #      ], p=0.2),
    transforms.RandomErasing(p=0.2),
    transforms.RandomRotation(degrees=(-10, 10)),
])

# Trained with this data:

# def __init__(self, imgs_dir, seed=42, num_opel=-1, num_door=-1,
#              num_deloitte_aug=-1, num_gan=-1, num_primary_multiple=1, augmentation=None,
#              test=False, predictor=None, bg_manager=None, grayscale=False):

train_dataset = CarDataset(train_path, augmentation=transform)
validation_dataset = CarDataset(validation_path, num_gan=0, num_deloitte_aug=0, num_opel=0, num_door=0,
                                num_primary_multiple=1)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
valid_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=0)

# We define the model:
model = smp.Unet(
    encoder_name='timm-resnest200e',  # We use the ResNeSt 200 backbone
    encoder_weights='imagenet',  # The backbone is trained on imagenet
    classes=9,  # We have 9 classes
    activation='softmax2d',  # The last activation is a softmax
    in_channels=3
)


def save_logs(train_log, valid_log):
    np.save("../unet-32-epoch-80-dice/train_log.npy", train_log)
    np.save("../unet-32-epoch-80-dice/valid_log.npy", valid_log)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

criterion = smpLoss.DiceLoss()  # The SMP library also contains various loss functions

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

model.to(DEVICE)

min_score = 1

train_logs = []
valid_logs = []

EPOCHS = args.num_epochs
for i in range(0, EPOCHS):
    print('\nEpoch: {}'.format(i))
    train_log = []
    model.train()
    for image, mask in train_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, mask)
        loss.backward()

        optimizer.step()

        train_log.append(loss.item())

    train_mean = np.mean(train_log)
    print("Training loss: ", train_mean)
    train_logs.append(train_mean)

    valid_log = []
    model.eval()
    for image, mask in valid_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        pred = model(image)

        loss = criterion(pred, mask)
        valid_log.append(loss.item())

    valid_mean = np.mean(valid_log)
    print("Validation loss: ", valid_mean)
    valid_logs.append(valid_mean)

    if (min_score > valid_mean):
        min_score = valid_mean
        torch.save(model.state_dict(), 'unet-32-epoch-80-dice/best_model_dict.pth')
        print("Model saved!")
    if i == EPOCHS / 2:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('---- Decreased Learning Rate to 1e-5! ----')

save_logs(train_logs, valid_logs)

train_log = np.load(
    args.train_log)  # np.load(r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/train_log.npy')
valid_log = np.load(
    args.valid_log)  # r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/valid_log.npy')

plt.clf()
plt.plot(train_log, label="Training")
plt.plot(valid_log, label="Validation")
plt.xlabel('Epochs')
plt.ylabel('Micro Dice Loss')
plt.legend()
plt.show()
