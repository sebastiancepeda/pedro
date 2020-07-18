import pathlib

import numpy as np
from loguru import logger

from cv.pytorch.pytorch_utils import train_model
from cv.pytorch.unet import UNet
from io_utils.data_source import get_image_label_gen, get_metadata


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/data/plates'
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': (572, 572),
        'im_channels': 3,
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.model',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
    }
    return params


def train_plate_segmentation(params):
    dsize = params['dsize']
    folder = params['folder']
    metadata = get_metadata(params)
    train_metadata = metadata.query("set == 'train'")
    test_metadata = metadata.query("set == 'test'")
    train_metadata = train_metadata.assign(idx=range(len(train_metadata)))
    test_metadata = test_metadata.assign(idx=range(len(test_metadata)))
    x_train, y_train = get_image_label_gen(folder, train_metadata, dsize)
    x_val, y_val = get_image_label_gen(folder, test_metadata, dsize)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    model_params = {
        'in_channels': 3,
        'out_channels': 2,
    }
    train_model(x_train, y_train, x_val, y_val, UNet, model_params, params,
                logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
