import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch

import segmentation_models_pytorch as smp

# We load utils from the root of our project:
from car_dataset import CarDataset
from args import get_args
args = get_args()

# We define the model:
model = smp.UnetPlusPlus(
    encoder_name='timm-resnest200e',  # We use the ResNeSt 200 backbone
    encoder_weights='imagenet',  # The backbone is trained on imagenet
    classes=9,  # We have 9 classes
    activation='softmax2d',  # The last activation is a softmax
    in_channels=3
)

train_log = np.load(args.train_log) #np.load(r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/train_log.npy')
valid_log = np.load(args.valid_log) #np.load(r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/valid_log.npy')


from torchmetrics.functional import dice_score, accuracy


def calc_test_metrics(model, test_dataloader):
    dice_scores_macro = []
    accuracy_macro = []

    for i in test_dataloader:
        img, mask = i
        pr_mask = model.predict(img)  # Predict the mask according to the image
        pred = pr_mask[0]
        truth = mask[0]

        # We go from [9,256,256] -> [256,256] - e.i. onehot encode to integer encode
        pred_label = torch.argmax(pred, dim=0)
        truth_label = torch.argmax(truth, dim=0)

        truth_flat = truth_label.view(-1)  # go from [256,256] -> [256*256]
        pred_flat = torch.flatten(pred, start_dim=1)  # go from [9,256,256] -> [9,256*256]
        pred_flat = pred_flat.permute(1, 0)  # go from [9,256*256] -> [256*256,9]]

        # calculate dice score macro with only present channels
        data_dicescore = dice_score(pred_flat, truth_flat, reduction='none', no_fg_score=-1)
        masked_dices = torch.masked_select(data_dicescore, data_dicescore.not_equal(-1))
        dice_scores_macro.append(masked_dices.mean())

        # calculate accuracy
        acc = accuracy(pred_label, truth_label, average='macro', num_classes=9)
        accuracy_macro.append(acc)

    return np.mean(dice_scores_macro), np.mean(accuracy_macro)

models_base_path = args.models_base_path # r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models'
test_path = args.test_path # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/test/"
test_dataset = CarDataset(test_path, test=True)
test_dataloader = DataLoader(test_dataset, shuffle=False)

# We use the model to measure performance on the test data:
model.eval()
dice, accuracy = calc_test_metrics(model, test_dataloader)
print("Dice Score: ", dice)
print("Accuracy: ", accuracy)


def visualize(car_img=None, mask=None, predicted=None):
    n = 3
    plt.figure(figsize=(16, 5))
    plt.subplot(1, n, 1)
    plt.imshow(np.dstack(car_img))
    plt.title("Actual image")
    plt.subplot(1, n, 2)
    plt.imshow(mask)
    plt.title("True mask")
    plt.subplot(1, n, 3)
    plt.imshow(predicted)
    plt.title("Model prediction")
    plt.show()


def prep_and_viz(data, model):
    img, mask = data

    mask = mask.permute(1, 2, 0)
    mask = torch.argmax(mask, dim=2)

    pred = model.predict(img.unsqueeze(0))

    pred = pred.squeeze().cpu().permute(1, 2, 0)
    pred = torch.argmax(pred, dim=2)
    print(img.shape)
    print(mask.shape)
    visualize(img, mask, pred)

prep_and_viz(test_dataset[1], model)
