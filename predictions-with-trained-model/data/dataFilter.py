import numpy as np
import os

save_path = r"C:/Users/tala1/Skrivebord/deeplearning/deeplearning-final-project/filtered_data/"
data_path = r"C:/Users/tala1/Downloads/carseg_data/carseg_data/clean_data_test/"
files = os.listdir(data_path)

for file in files:
    path_d = data_path + file
    img_array = np.load(path_d)
    rgb_dims = img_array[0:3]
    rgb_dims[0,:,:]=(rgb_dims[0,:,:]+0.485/0.229-0.485)/0.229
    rgb_dims[1,:,:]=(rgb_dims[1,:,:]+0.456/0.224-0.456)/0.224
    rgb_dims[2,:,:]=(rgb_dims[2,:,:]+0.406/0.225-0.406)/0.225

    mask = img_array[12]
    combined_data = np.append(rgb_dims, [mask], axis=0)
    path_s = save_path + file
    np.save(path_s, combined_data)