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

'''
Training the Segmentation Model
'''
# Training loop
num_epochs = args.num_epochs
validation_every_steps = (len(train) // (10 * args.batch_size))

global_step = 0
model.train()
train_dice_scores, valid_dice_score = [], []
max_valid_dice_score, new_model = None, None

for epoch in tqdm(range(num_epochs)):

    train_dice_scores_batches = []
    epoch_loss = 0
    model.train()
    for rgb_img, mask_img in train_loader:
        rgb_img, mask_img = rgb_img.to(DEVICE), mask_img.to(DEVICE)

        # Forward pass, compute gradients, perform one training step.
        optimizer.zero_grad()

        output = model(rgb_img)

        batch_loss = criterion(
            output.flatten(start_dim=2, end_dim=len(output.size()) - 1),
            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1).type(torch.long)
        )
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss

        # Increment step counter
        global_step += 1

        # Compute DICE score.
        predictions = output.flatten(start_dim=2, end_dim=len(output.size()) - 1).softmax(1)
        train_dice_scores_batches.append(
            dice_score(
                predictions,
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1)
            )
        )

        if global_step % validation_every_steps == 0:

            # Append average training DICE score to list.
            train_dice_scores.append(np.mean(train_dice_scores_batches))

            train_dice_scores_batches = []

            # Compute DICE scores on validation set.
            valid_dice_scores_batches = []
            valid_loss = 0
            with torch.no_grad():
                model.eval()
                for rgb_img, mask_img in valid_loader:
                    rgb_img, mask_img = rgb_img.to(DEVICE), mask_img.to(DEVICE)
                    output = model(rgb_img)
                    loss = criterion(
                        output.flatten(start_dim=2, end_dim=len(output.size()) - 1),
                        mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1).type(torch.long)
                    )
                    valid_loss += loss

                    predictions = output.flatten(start_dim=2, end_dim=len(output.size()) - 1).softmax(1)

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_dice_scores_batches.append(
                        dice_score(
                            predictions,
                            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size()) - 1)
                        ) * len(rgb_img)
                    )

                    # Keep the best model
                    if (max_valid_dice_score == None) or (valid_dice_scores_batches[-1] > max_valid_dice_score):
                        max_valid_dice_score = valid_dice_scores_batches[-1]
                        new_model = model.state_dict()

                model.train()

            # Append average validation DICE score to list.
            valid_dice_score.append(np.sum(valid_dice_scores_batches) / len(valid))

            print(f"Step {global_step:<5}   training DICE score: {train_dice_scores[-1]}")
            print(f"             test DICE score: {valid_dice_score[-1]}")

print("Finished training.")
# Save model
model.load_state_dict(new_model)
