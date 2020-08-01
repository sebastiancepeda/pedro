from loguru import logger

from cv.tensorflow_models.tensorflow_utils import train_model
from cv.tensorflow_models.unet_little import get_model_definition, normalize_image_shape
from io_utils.data_source import get_image_label, get_plates_text_metadata


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    folder = f'{path}/plates/output_plate_segmentation'
    width = 200
    height = 50
    height, width = normalize_image_shape(height, width)
    dsize = (height, width)
    in_channels = 1
    out_channels = 2
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': dsize,
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.h5',
        'labels': f"{folder}/labels_plates_text-name_20200724050222.json",
        'metadata': f"{folder}/files.csv",
        'model_params': {
            'img_height': dsize[0],
            'img_width': dsize[1],
            'in_channels': in_channels,
            'out_channels': out_channels,
        },
    }
    return params


def train_ocr_model(params):
    dsize = params['dsize']
    in_channels = params['model_params']['in_channels']
    out_channels = params['model_params']['out_channels']
    folder = params['folder']
    metadata = get_plates_text_metadata(params)
    train_metadata = metadata.query("set == 'train'")
    test_metadata = metadata.query("set == 'test'")
    train_metadata = train_metadata.assign(idx=range(len(train_metadata)))
    test_metadata = test_metadata.assign(idx=range(len(test_metadata)))
    x_train, y_train = get_image_label(folder, train_metadata, dsize,
                                       in_channels, out_channels)
    x_val, y_val = get_image_label(folder, test_metadata, dsize,
                                   in_channels, out_channels)
    train_model(x_train, y_train, x_val, y_val, get_model_definition, params,
                logger)


if __name__ == "__main__":
    train_ocr_model(get_params())
