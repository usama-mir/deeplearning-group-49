from numpy import load
from glob import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import math
from args import get_args
import numpy as np
import os

args = get_args()

txt_paths = args.txt_paths

save_path_filtered = args.save_path_filtered  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/filtered_data/"
data_path = args.data_path  # r"C:/Users/tala1/Downloads/carseg_data/carseg_data/clean_data_test/"
files = os.listdir(data_path)

for file in files:
    path_d = data_path + file
    img_array = np.load(path_d)
    rgb_dims = img_array[0:3]
    rgb_dims[0, :, :] = (rgb_dims[0, :, :] + 0.485 / 0.229 - 0.485) / 0.229
    rgb_dims[1, :, :] = (rgb_dims[1, :, :] + 0.456 / 0.224 - 0.456) / 0.224
    rgb_dims[2, :, :] = (rgb_dims[2, :, :] + 0.406 / 0.225 - 0.406) / 0.225

    mask = img_array[3]
    combined_data = np.append(rgb_dims, [mask], axis=0)
    path_s = save_path_filtered + file
    np.save(path_s, combined_data)


def get_data_paths(files, data_target):
    data_paths = []
    for f in files:
        for t in data_target:
            if (f[:-4] == t[:-4]):
                data_paths.append(f)
    return data_paths


def get_data_without_folder(files, folder):
    data = []
    for f in files:
        if f not in folder:
            data.append(f)
    return data


def get_augmented(data_no_test, test_data):
    test_augmented = []
    for f in data_no_test:
        for t in test_data:
            if t[:-4] + "-aug" == f[:-6]:
                test_augmented.append(f)
    return test_augmented


def get_specific_data(dataset, data_type):
    dataset_na = []
    for f in dataset:
        if data_type in f:
            dataset_na.append(f)
    return dataset_na


def get_gan_images(dataset):
    gan_file = open(txt_paths + "cycle_gan_files.txt", "r")
    gan_data_jpg = gan_file.read().split("\n")
    gan = get_data_paths(dataset, gan_data_jpg)
    return gan


def get_data_paths(files, data_target):
    data_paths = []
    for f in files:
        for t in data_target:
            if (f[:-4] == t[:-4]):
                data_paths.append(f)
    return data_paths


def get_data_without_folder(files, folder):
    data = []
    for f in files:
        if f not in folder:
            data.append(f)
    return data


def get_augmented(data_no_test, test_data):
    test_augmented = []
    for f in data_no_test:
        for t in test_data:
            if t[:-4] + "-aug" == f[:-6]:
                test_augmented.append(f)
    return test_augmented


def get_specific_data(dataset, data_type):
    dataset_na = []
    for f in dataset:
        if data_type in f:
            dataset_na.append(f)
    return dataset_na


def save_files(data_path, save_path, arr):
    for i in arr:
        path = data_path + i
        data = np.load(path)
        new_path = save_path + i
        np.save(new_path, data)


# The folder where you run this script


txt_paths = args.txt_paths  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/txts/"

save_path_splitted = args.save_path_splitted  # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/"
files = os.listdir(data_path)

# First we get the 30 test images given by martin, and remove the augmented versions of them
test_file = open(txt_paths + "test_data.txt", "r")
test_data_jpg = test_file.read().split("\n")

test = get_data_paths(files, test_data_jpg)
data_no_test = get_data_without_folder(files, test)
test_augmented = get_augmented(data_no_test, test)
training_data = get_data_without_folder(data_no_test, test_augmented)

# We then get all the gan images and save them
gan_file = open(txt_paths + "cycle_gan_files.txt", "r")
gan_data_jpg = gan_file.read().split("\n")
gan = get_data_paths(files, gan_data_jpg)
training_no_gan = get_data_without_folder(training_data, gan)

# We get all the opel images and save them
opel = get_specific_data(training_no_gan, 'OPEL')
training_no_opel = get_data_without_folder(training_no_gan, opel)

# We get all the door images and save them
door = get_specific_data(training_no_opel, 'DOOR')
training_no_door = get_data_without_folder(training_no_opel, door)

# We get all the augmented images and save them
aug = get_specific_data(training_no_door, "-aug")
training_primary = get_data_without_folder(training_no_door, aug)

##################################################################################
# Now we split each of the different data kinds up in 80/20 train and validation #
seed_int = 32  # set seed to get same splits everytime

random.seed(seed_int)
train_primary = random.sample(
    training_primary, math.ceil(len(training_primary) * 0.80))
val_primary = [i for i in training_primary if i not in train_primary]
print("train prim len:", len(train_primary),
      "- val prim len:", len(val_primary))

# We get all the augmented of our primary data and add them in train and aug
train_aug = get_augmented(aug, train_primary)
val_aug = get_augmented(aug, val_primary)
print("train aug len:", len(train_aug), "- val aug len:", len(val_aug))

random.seed(seed_int)
train_opel = random.sample(opel, math.ceil(len(opel) * 0.80))
val_opel = [i for i in opel if i not in train_opel]

random.seed(seed_int)
train_door = random.sample(door, math.ceil(len(door) * 0.80))
val_door = [i for i in door if i not in train_door]

random.seed(seed_int)
train_gan = random.sample(gan, math.ceil(len(gan) * 0.80))
val_gan = [i for i in gan if i not in train_gan]

# And finally we now have a completely fair split of 80/20 of each kind
validation = val_gan + val_door + val_opel + val_aug + val_primary
train = train_gan + train_door + train_opel + train_aug + train_primary

# Unit tests to ensure that the 3 are completely split up
for i in train:
    assert (i not in validation)
    assert (i not in test)

for i in validation:
    assert (i not in test)

# Save images
save_files(data_path, save_path_splitted + "test/", test)
save_files(data_path, save_path_splitted + "validation/", validation)
save_files(data_path, save_path_splitted + "train/", train)


class CarDataset(Dataset):
    def __init__(self, imgs_dir, seed=42, num_opel=-1, num_door=-1,
                 num_deloitte_aug=-1, num_gan=-1, num_primary_multiple=1, augmentation=None,
                 test=False, predictor=None, bg_manager=None, grayscale=False):
        self.imgs_dir = imgs_dir
        self.augmentation = augmentation
        self.predictor = predictor
        self.bg_manager = bg_manager
        self.grayscale = grayscale

        raw_ids = os.listdir(imgs_dir)

        random.seed(seed)
        random.shuffle(raw_ids)

        self.ids = []

        if (test == False):
            opel = get_specific_data(raw_ids, 'OPEL')
            if (num_opel == -1):
                self.ids = self.ids + opel
            else:
                assert (len(opel) >= num_opel)
                random.seed(seed)
                sample_opel = random.sample(opel, num_opel)
                self.ids = self.ids + sample_opel

            door = get_specific_data(raw_ids, 'DOOR')
            if (num_door == -1):
                self.ids = self.ids + door
            else:
                assert (len(door) >= num_door)
                random.seed(seed)
                sample_door = random.sample(door, num_door)
                self.ids = self.ids + sample_door

            aug = get_specific_data(raw_ids, '-aug')
            if (num_deloitte_aug == -1):
                self.ids = self.ids + aug
            else:
                assert (len(aug) >= num_deloitte_aug)
                random.seed(seed)
                sample_aug = random.sample(aug, num_deloitte_aug)
                self.ids = self.ids + sample_aug

            gan = get_gan_images(raw_ids)
            if (num_gan == -1):
                self.ids = self.ids + gan
            else:
                assert (len(gan) >= num_gan)
                random.seed(seed)
                sample_gan = random.sample(gan, num_gan)
                self.ids = self.ids + sample_gan

            primary_images = []
            for f in raw_ids:
                if ((f not in aug) and (f not in gan) and (f not in door) and (f not in opel)):
                    primary_images.append(f)
            for i in range(num_primary_multiple):
                self.ids = self.ids + primary_images
        else:
            self.ids = raw_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        np_obj = glob(self.imgs_dir + idx)

        data = load(np_obj[0])

        img = data[0:3]
        # We try to remove the background
        if (self.predictor != None):
            image_bg = self.bg_manager.get_image(np_obj[0])
            if (image_bg == 'empty'):
                img_no_bg = self.bg_manager.get_img_no_bg(self.predictor, np.dstack(img))
                if (img_no_bg != 'empty'):
                    self.bg_manager.add_image(np_obj[0], img_no_bg)
                else:
                    self.bg_manager.add_image(np_obj[0], 'empty')
            else:
                img_no_bg = image_bg
            if (img_no_bg != 'empty'):
                img = torch.from_numpy(img_no_bg).type(
                    torch.FloatTensor)  # numpy -> torch
                # The predictor takes H,W,C so we make it C,H,W again
                img = img.permute(2, 0, 1)
            else:
                # print("------- COULD NOT REMOVE BACKGROUND OF IMAGE: ---------")
                # print(np_obj)
                # print("-------------------------------------------------------")
                img = torch.from_numpy(img).type(
                    torch.FloatTensor)  # numpy -> torch
        else:
            img = torch.from_numpy(img).type(
                torch.FloatTensor)  # numpy -> torch

        mask = data[-1]
        mask = torch.from_numpy(mask).type(torch.FloatTensor)  # numpy -> torch
        mask = torch.nn.functional.one_hot(
            mask.to(torch.int64), 9)  # We one-hot-encode the mask
        mask = mask.permute(2, 0, 1)  # (256,256,9) -> (9,256,256)

        if (self.augmentation != None):
            # Choose a numpy seed and ensure that transforms use it
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            torch.manual_seed(seed)
            img = self.augmentation(img)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask = self.augmentation(mask)

        if (self.grayscale == True):
            img = T.Grayscale()(img)

        return img, mask
