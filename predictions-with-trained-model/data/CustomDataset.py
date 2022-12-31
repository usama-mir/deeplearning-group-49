from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import torch
from pathlib import Path
import glob

'''
Custom Segmentation Dataset Class
'''


class CarSegmentationDataset(Dataset):
    def __init__(self, data_list, transform=ToTensor(), transform_mask=ToTensor()):
        # list of paths
        self.car_images = data_list
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.car_images)

    def __getitem__(self, idx):
        img_path = self.car_images[idx]
        # load npy array
        numpy_array = np.load(img_path)
        # RGB image
        rgb_img = (np.transpose(numpy_array[:3], (1, 2, 0)) * 255).astype(float)
        # mask image
        mask_img = numpy_array[3]
        # Convert to tensor
        rgb_img = self.transform(rgb_img).type(torch.float)
        mask_img = self.transform_mask(mask_img).type(torch.int)
        return rgb_img, mask_img


'''
Splitting data
'''


def split_dataset(data_path, train_ratio, valid_ratio , seed_random , transform=None):
    """_summary_
    :param data_path: path to folder that contains all npy data files
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
    all_paths = [Path(p).absolute() for p in glob.glob(data_path + '/*')]

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
    permutation = np.random.permutation(len(train_valid_data_list))
    train_indices = permutation[:int(train_ratio * len(train_valid_data_list))]
    valid_indices = permutation[int(train_ratio * len(train_valid_data_list)):]

    if transform == None:
        train_dataset = CarSegmentationDataset(list(train_valid_data_list[train_indices]))
        valid_dataset = CarSegmentationDataset(list(train_valid_data_list[valid_indices]))
    else:
        train_dataset = CarSegmentationDataset(list(train_valid_data_list[train_indices]), transform)
        valid_dataset = CarSegmentationDataset(list(train_valid_data_list[valid_indices]), transform)

    return train_dataset, valid_dataset
