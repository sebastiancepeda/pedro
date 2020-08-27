from loguru import logger

from cv.tensorflow_models.tensorflow_utils import train_model_gen
from cv.tensorflow_models.unet2text2 import (
    get_model_definition,
    normalize_image_shape
)
from io_utils.data_source import (
    get_plates_text_metadata,
    get_image_text_label,
    get_image_text_label_sim
)
from io_utils.image_text_label_generator import ImageTextLabelGenerator
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/plate_segmentation'
    output_folder = f'{path}/plates/plate_ocr'
    width = 200
    height = 50
    height, width = normalize_image_shape(height, width)
    #height = height + 1
    #width = width + 1
    dsize = (height, width)
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 1
    out_channels = len(alphabet)
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'epochs': 10 * 1000,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'metadata': f"{path}/plates/input/labels/ocr/files.csv",
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
    model_params = params['model_params']
    in_folder = params['input_folder']
    alphabet = params['alphabet']
    #
    in_channels = model_params['in_channels']
    out_channels = model_params['out_channels']
    metadata = get_plates_text_metadata(params)
    metadata.file_name = 'plate_' + metadata.file_name
    metadata.file_name = metadata.file_name.str.split('.').str[0] + '.png'
    train_meta = metadata.query("set == 'train'")
    test_meta = metadata.query("set == 'test'")
    train_meta = set_index(train_meta)
    test_meta = set_index(test_meta)
    model, preprocess_input = get_model_definition(**model_params)
    f_train_params = {
        'folder': in_folder, 'metadata': train_meta, 'dsize': dsize,
        'in_channels': in_channels, 'out_channels': out_channels,
        'alphabet': alphabet
    }
    f_test_params = {
        'folder': in_folder, 'metadata': test_meta, 'dsize': dsize,
        'in_channels': in_channels, 'out_channels': out_channels,
        'alphabet': alphabet
    }
    data_train = ImageTextLabelGenerator(get_image_text_label,
                                         preprocess_input, f_train_params)
    data_val = ImageTextLabelGenerator(get_image_text_label, preprocess_input,
                                       f_test_params)
    train_model_gen(data_train, data_val, model, params, logger)


def train_ocr_model_sim(params):
    dsize = params['dsize']
    model_params = params['model_params']
    in_folder = params['input_folder']
    alphabet = params['alphabet']
    #
    in_channels = model_params['in_channels']
    out_channels = model_params['out_channels']
    metadata = get_plates_text_metadata(params)
    metadata.image = 'plates_' + metadata.image
    metadata.image = metadata.image.str.split('.').str[0] + '.png'
    test_meta = metadata.query("set == 'test'")
    test_meta = set_index(test_meta)
    model, preprocess_input = get_model_definition(**model_params)
    f_train_params = {
        'dsize': dsize, 'in_channels': in_channels,
        'out_channels': out_channels, 'alphabet': alphabet
    }
    f_test_params = {
        'folder': in_folder, 'metadata': test_meta, 'dsize': dsize,
        'in_channels': in_channels, 'out_channels': out_channels,
        'alphabet': alphabet
    }
    data_train = ImageTextLabelGenerator(get_image_text_label_sim,
                                         preprocess_input, f_train_params)
    data_val = ImageTextLabelGenerator(get_image_text_label, preprocess_input,
                                       f_test_params)
    train_model_gen(data_train, data_val, model, params, logger)


if __name__ == "__main__":
    train_ocr_model(get_params())
