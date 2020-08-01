import pathlib

from loguru import logger
import pandas as pd

from cv.tensorflow_models.unet_little import get_model_definition
from cv.tensorflow_models.tensorflow_utils import train_model
from io_utils.data_source import get_image_label, get_plates_text_metadata, get_plates_text_area_metadata
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/input'
    output_folder = f'{path}/plates/output_plate_segmentation'
    #dsize = (576, 576)
    dsize = (256, 256)
    # alphabet = '0p'
    alphabet = [' ', 'plate']
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 3
    out_channels = len(alphabet)
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'epochs': 1000,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'labels': f"{input_folder}/labels_plate_text.json",
        'metadata': f"{input_folder}/files.csv",
        'alphabet': alphabet,
        'model_params': {
            'img_height': dsize[0],
            'img_width': dsize[1],
            'in_channels': in_channels,
            'out_channels': out_channels,
        },
    }
    return params


def train_plate_segmentation(params):
    dsize = params['dsize']
    in_channels = params['model_params']['in_channels']
    out_channels = params['model_params']['out_channels']
    input_folder = params['input_folder']
    metadata = get_plates_text_area_metadata(params)
    train_meta = metadata.query("set == 'train'")
    test_meta = metadata.query("set == 'test'")
    train_meta = set_index(train_meta)
    test_meta = set_index(test_meta)
    x_train, y_train = get_image_label(input_folder, train_meta, dsize, in_channels, out_channels, params)
    x_val, y_val = get_image_label(input_folder, test_meta, dsize, in_channels, out_channels, params)
    train_model(x_train, y_train, x_val, y_val, get_model_definition, params, logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
