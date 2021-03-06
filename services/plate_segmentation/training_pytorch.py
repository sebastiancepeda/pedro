import pathlib

import numpy as np
from loguru import logger

from cv.pytorch.pytorch_utils import train_model
from cv.pytorch.unet_small import UNetSmall
from io_utils.data_source import get_image_label, get_plates_bounding_metadata
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data/'
    folder = f'{path}/plates'
    dsize = (572, 572)
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': dsize,
        'im_channels': 3,
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.model',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
        'model_params': {
            'in_channels': 3,
            'out_channels': 2,
        },
    }
    return params


def train_plate_segmentation(params):
    dsize = params['dsize']
    folder = params['folder']
    metadata = get_plates_bounding_metadata(params)
    train_meta = metadata.query("set == 'train'")
    test_meta = metadata.query("set == 'test'")
    train_meta = set_index(train_meta)
    test_meta = set_index(test_meta)
    x_train, y_train = get_image_label(folder, train_meta, dsize)
    x_val, y_val = get_image_label(folder, test_meta, dsize)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    train_model(x_train, y_train, x_val, y_val, UNetSmall, params, logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
