import glob
import numpy as np
import cv2
import pickle
import os
import warnings

warnings.filterwarnings("ignore")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_list = glob.glob("./data/test_batch*")

save_path = "./dataset/test"

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

for l in train_list:
    l_dict = unpickle(l)

    for im_idx, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        # print(im_label,im_name,im_data)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, (3, 32, 32))

        im_data = np.transpose(im_data, (1,2, 0))

        # cv2.imshow("im_data", cv2.resize(im_data,(200,200)))
        #
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path, im_label_name)):
            os.mkdir("{}/{}".format(save_path, im_label_name))

        cv2.imwrite("{}/{}/{}".format(
            save_path, im_label_name,im_name.decode("utf8")), im_data)







