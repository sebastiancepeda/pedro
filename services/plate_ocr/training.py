from loguru import logger
import pandas as pd

from cv.tensorflow_models.tensorflow_utils import train_model
from cv.tensorflow_models.unet2text import get_model_definition, normalize_image_shape
from io_utils.data_source import get_image_text_label_gen, get_plates_text_metadata


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/output_plate_text_segmentation'
    output_folder = f'{path}/plates/output_plate_ocr'
    width = 200
    height = 50
    height, width = normalize_image_shape(height, width)
    dsize = (height, width)
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 1
    out_channels = len(alphabet)
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'epochs': 10*1000,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'labels': f"{input_folder}/labels_plates_ocr_1.json",
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


def train_ocr_model(params):
    dsize = params['dsize']
    in_channels = params['model_params']['in_channels']
    out_channels = params['model_params']['out_channels']
    input_folder = params['input_folder']
    metadata = get_plates_text_metadata(params)
    train_meta = metadata.query("set == 'train'")
    test_meta = metadata.query("set == 'test'")
    train_im_idx = train_meta.image_name.unique()
    test_im_idx = test_meta.image_name.unique()
    train_im_idx = pd.DataFrame(data={
        'image_name': train_im_idx,
        'idx': range(len(train_im_idx)),
    })
    test_im_idx = pd.DataFrame(data={
        'image_name': test_im_idx,
        'idx': range(len(test_im_idx)),
    })
    train_meta = train_meta.merge(train_im_idx, on=['image_name'], how='left')
    test_meta = test_meta.merge(test_im_idx, on=['image_name'], how='left')
    x_train, y_train = get_image_text_label_gen(input_folder, train_meta, dsize,
                                           in_channels, out_channels, params)
    x_val, y_val = get_image_text_label_gen(input_folder, test_meta, dsize,
                                       in_channels, out_channels, params)
    train_model(x_train, y_train, x_val, y_val, get_model_definition, params,
                logger)


if __name__ == "__main__":
    train_ocr_model(get_params())
