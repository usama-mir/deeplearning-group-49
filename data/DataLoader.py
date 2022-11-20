from torch.utils.data import DataLoader
from .CustomDataset import split_dataset

'''
DataLoader
'''


def loaders(directory_path, train_ratio, valid_ratio, seed_random, batch_size, shuffle, drop_last):
    train, valid = split_dataset(directory_path, train_ratio, valid_ratio, seed_random)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return train,valid, train_loader, valid_loader
