import pathlib

from loguru import logger

from cv.tensorflow_models.tensorflow_utils import train_model
from io_utils.data_source import get_image_label, get_plates_bounding_metadata
from cv.seg_models.model_definition import get_model_definition
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data/'
    folder = f'{path}/plates'
    dsize = (768, 768)
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': dsize,
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.h5',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
        'model_params': {}
    }
    return params


def train_plate_segmentation(params):
    dsize = params['dsize']
    folder = params['folder']
    metadata = get_plates_bounding_metadata(params)
    train_metadata = metadata.query("set == 'train'")
    test_metadata = metadata.query("set == 'test'")
    train_metadata = set_index(train_metadata)
    test_metadata = set_index(test_metadata)
    x_train, y_train = get_image_label(folder, train_metadata, dsize)
    x_val, y_val = get_image_label(folder, test_metadata, dsize)
    train_model(x_train, y_train, x_val, y_val, get_model_definition, params, logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
