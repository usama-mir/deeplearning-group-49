import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--init_lr", default=0.001)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    #parser.add_argument("--directory_path", required=True, type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--drop_last", default=True, type=bool)
    parser.add_argument("--test_size", choices=range(0, 1), default=0.2, type=float)
    parser.add_argument("--train_ratio", default=0.85, type=float)
    parser.add_argument("--valid_ratio", default=0.15, type=float)
    parser.add_argument("--seed_random", default= 42, type=int)

    path = "/zhome/4b/9/89148/workspace/python/deeplearning/"
    #datapath = r"C:/Users/tala1/Downloads/carseg_data/carseg_data/clean_data_test/"
    #path = r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/"
    parser.add_argument("--train_log", default=path+'models/train_log.npy', type=str)
    parser.add_argument("--valid_log", default=path+'models/train_log.npy', type=str)
    parser.add_argument("--validation_path", default=path+'splitted_data/validation/', type=str)
    parser.add_argument("--train_path", default=path+'splitted_data/train/', type=str)
    parser.add_argument("--test_path", default=path+'splitted_data/test/', type=str)
    parser.add_argument("--txt_paths", default=path+'txts/', type=str)
    parser.add_argument("--data_path", required=True,type=str)
    parser.add_argument("--save_path_filtered", default=path+"filtered_data/", type=str)
    parser.add_argument("--save_path_splitted", default=path+"splitted_data/", type=str)
    parser.add_argument("--models_base_path", default=path +"models/", type=str)



    #'/zhome/4b/9/89148/workspace/python/deeplearning'

    #r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/"
    # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/filtered_data/"
   # r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/txts/"
   # validation_path = r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/validation/"
   # train_path = r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/train/"
   # test_path = r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/splitted_data/test/"

    #train_log = np.load(r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/train_log.npy')
    # valid_log = np.load(r'C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/models/valid_log.npy')

    args = parser.parse_args()

    return args

