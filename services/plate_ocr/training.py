from loguru import logger

from cv.tensorflow_models.tensorflow_utils import train_model_gen
from cv.tensorflow_models.unet2text import get_model_definition, \
    normalize_image_shape
from io_utils.data_source import get_plates_text_metadata
from io_utils.image_text_label_generator import ImageTextLabelGenerator
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/output_plate_segmentation'
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
        'epochs': 1 * 1000,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        # 'labels': f"{input_folder}/labels_plates_ocr_1.json",
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
    model_params = params['model_params']
    in_folder = params['input_folder']
    #
    in_channels = model_params['in_channels']
    out_channels = model_params['out_channels']
    metadata = get_plates_text_metadata(params)
    metadata.image = 'plates_' + metadata.image
    metadata.image = metadata.image.str.split('.').str[0] + '.png'
    train_meta = metadata.query("set == 'train'")
    test_meta = metadata.query("set == 'test'")
    train_meta = set_index(train_meta)
    test_meta = set_index(test_meta)
    model, preprocess_input = get_model_definition(**model_params)
    data_train = ImageTextLabelGenerator(
        in_folder, train_meta, dsize, in_channels, out_channels,
        preprocess_input, params)
    data_val = ImageTextLabelGenerator(
        in_folder, test_meta, dsize, in_channels, out_channels,
        preprocess_input, params)
    x_train, y_train = data_train.__getitem__(0)
    x_val, y_val = data_val.__getitem__(0)
    train_model_gen(data_train, data_val, model, params, logger)


if __name__ == "__main__":
    train_ocr_model(get_params())
