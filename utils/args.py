import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--init_lr", default=0.001)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--directory_path", required=True, type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--drop_last", default=True, type=bool)
    parser.add_argument("--test_size", choices=range(0, 1), default=0.2, type=float)
    parser.add_argument("--train_ratio", default=0.85, type=float)
    parser.add_argument("--valid_ratio", default=0.15, type=float)
    parser.add_argument("--seed_random", default= 42, type=int)


    args = parser.parse_args()

    return args

