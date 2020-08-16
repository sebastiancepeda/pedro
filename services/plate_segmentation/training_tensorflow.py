from loguru import logger

from cv.tensorflow_models.tensorflow_utils import train_model
from cv.tensorflow_models.unet_little import get_model_definition
from io_utils.data_source import (
    get_image_label, get_filenames,
    get_segmentation_labels)
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/input'
    output_folder = f'{path}/plates/plate_segmentation'
    # dsize = (576, 576)
    dsize = (256, 256)
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
        'labels': f"{input_folder}/labels/segmentation",
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

    train_meta = get_filenames(f"{input_folder}/train")
    test_meta = get_filenames(f"{input_folder}/test")
    logger.info(f"Train meta shape: {train_meta.shape}")
    logger.info(f"Test meta shape: {test_meta.shape}")
    labels = get_segmentation_labels(params['labels'])
    train_meta = train_meta.merge(labels, on=['file_name'], how='left')
    test_meta = test_meta.merge(labels, on=['file_name'], how='left')
    train_meta = train_meta.loc[train_meta.label.notnull()]
    test_meta = test_meta.loc[test_meta.label.notnull()]
    train_meta = set_index(train_meta)
    test_meta = set_index(test_meta)
    x_train, y_train = get_image_label(input_folder, train_meta, dsize, in_channels, out_channels, params)
    x_val, y_val = get_image_label(input_folder, test_meta, dsize, in_channels, out_channels, params)
    train_model(x_train, y_train, x_val, y_val, get_model_definition, params, logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
