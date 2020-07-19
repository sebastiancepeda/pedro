import pathlib

from loguru import logger

from cv.tensorflow_models.model_definition import get_model_definition
from cv.tensorflow_models.tensorflow_utils import train_model
from io_utils.data_source import get_image_label_gen, get_metadata


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/data/plates'
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': (576, 576),
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.h5',
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
    model_params = {
        'img_height': dsize[0],
        'img_width': dsize[1],
        'in_channels': 3,
        'out_channels': 2,
    }
    train_model(x_train, y_train, x_val, y_val, get_model_definition,
                model_params, params, logger)


if __name__ == "__main__":
    train_plate_segmentation(get_params())
