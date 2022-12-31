#imports
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

from model import UNet
from model import OutConv
import numpy as np
from data import loaders
from utils.dice_score import dice_score
from utils.args import get_args
import time
import matplotlib.pyplot as plt
args = get_args()
train, valid, train_loader, valid_loader = loaders(args.directory_path, args.train_ratio, args.valid_ratio,
                                                   args.seed_random, args.batch_size, args.shuffle, args.drop_last)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
Load model
'''
model = UNet(n_channels=3, n_classes=2)
model.outc = OutConv(64, 9)
model = model.to(DEVICE)

'''
Loss function + optmizier
'''
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



'''
Training the Segmentation Model
'''
# Training loop
num_epochs = args.num_epochs
validation_every_steps = (len(train) // (10 * args.batch_size))

global_step = 0
model.train()
train_dice_scores, valid_dice_scores = [], []
max_valid_dice_score, new_model = None, None

for epoch in tqdm(range(num_epochs)):
    start_time = time.time()
    train_dice_scores_batches = []


    ''' Train '''
    epoch_loss_train = 0

    model.train()
    for rgb_img, mask_img in train_loader:
        rgb_img, mask_img = rgb_img.to(DEVICE), mask_img.to(DEVICE)

        optimizer.zero_grad()

        y_pred_train = model(rgb_img)

        loss_train = criterion(
            y_pred_train.flatten(start_dim=2, end_dim=len(y_pred_train.size()) - 1),
            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1).type(torch.long)
        )
        loss_train.backward()
        optimizer.step()

        epoch_loss_train += loss_train


        # Train DICE score
        predictions = y_pred_train.flatten(start_dim=2, end_dim=len(y_pred_train.size()) - 1).softmax(1)
        train_dice_scores_batches.append(
            dice_score(
                predictions,
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1)
            )
        )


        # Increment step counter
        global_step += 1


        if global_step % validation_every_steps == 0:

            # Average training DICE score
            train_dice_scores.append(np.mean(train_dice_scores_batches))
            train_dice_scores_batches = []

            '''Evaluate '''
            valid_dice_scores_batches = []
            epoch_loss_valid = 0
            with torch.no_grad():
                model.eval()
                for rgb_img, mask_img in valid_loader:
                    rgb_img, mask_img = rgb_img.to(DEVICE), mask_img.to(DEVICE)
                    y_pred_valid = model(rgb_img)
                    loss = criterion(
                        y_pred_valid.flatten(start_dim=2, end_dim=len(y_pred_valid.size()) - 1),
                        mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1).type(torch.long)
                    )
                    epoch_loss_valid += loss



                    predictions = y_pred_valid.flatten(start_dim=2, end_dim=len(y_pred_valid.size()) - 1).softmax(1)
                    valid_dice_scores_batches.append(
                        dice_score(
                            predictions,
                            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1)
                        ) * len(rgb_img)
                    )

                valid_loss = epoch_loss_valid / len(valid_loader)

            # Train Loss
            train_loss = epoch_loss_train / len(train_loader)



            # Keep the best model
            if (max_valid_dice_score == None) or (valid_dice_scores_batches[-1] > max_valid_dice_score):
                max_valid_dice_score = valid_dice_scores_batches[-1]
                new_model = model.state_dict()


            # Append average validation DICE score to list.
            valid_dice_scores.append(np.sum(valid_dice_scores_batches) / len(valid))



            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s| Step {global_step:<5} \n '
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            data_str += f"training DICE score: {train_dice_scores[-1]}\n"
            data_str += f"test DICE score: {valid_dice_scores[-1]}\n"
            print(data_str)

print("Finished training.")
# Save model
model.load_state_dict(new_model)
# Plot and label the training and validation loss values
plt.plot(list(range(global_step)), train_dice_scores, label='Training Dice Loss')
plt.plot(list(range(global_step)), valid_dice_scores, label='Validation Dice Loss')
plt.plot(list(range(global_step)), train_loss, label='Training Loss')
plt.plot(list(range(global_step)), valid_loss, label='Training Loss')


# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# Display the plot
plt.legend(loc='best')
plt.show()