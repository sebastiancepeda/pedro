import cv2
import pandas as pd
import numpy as np
from loguru import logger

from cv.image_processing import (
    pred2im, save_image
)
from cv.tensorflow_models.unet2text3 import (normalize_image_shape)
from cv.tensorflow_models.unet2text3 import get_model_definition as plate_ocr_model_def
from io_utils.data_source import (
    get_image_text_label, get_plates_text_metadata)
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/plate_segmentation'
    output_folder = f'{path}/plates/plate_ocr'
    width = 200
    height = 50
    ocr_height, ocr_width = normalize_image_shape(50, 200)
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'plate_dsize': (ocr_height, ocr_width),
        'plate_ocr_model_file': f'{output_folder}/model/best_model.h5',
        'metadata': f"{path}/plates/input/labels/ocr/files.csv",
        'alphabet': alphabet,
        'debug_level': 1,
        'plate_ocr_model_params': {
            'img_height': ocr_height,
            'img_width': ocr_width,
            'in_channels': 1,
            'out_channels': len(alphabet),
        },
    }
    return params


def draw_rectangle(im, r):
    x, y, w, h = r
    im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return im


def image_ocr(event, context):
    logger = context['logger']
    out_folder = context['output_folder']
    debug_level = context['debug_level']
    plate_ocr_model = context['plate_ocr_model']
    plate_ocr_preprocessing = context['plate_ocr_preprocessing']
    in_channels = context['plate_ocr_model_params']['in_channels']
    dsize = context['plate_dsize']
    image = event['image']
    filename = event['file']
    rectangle = event['rectangle']
    image_debug = event['image_debug']
    if image is None:
        result = {
            'filename': filename,
            'text': 'none_image',
        }
        return result
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangle_point = rectangle.mean(axis=0).astype(int)
    image = cv2.resize(image, dsize=(dsize[1], dsize[0]), interpolation=cv2.INTER_CUBIC)
    image = np.reshape(image, dsize)
    image_pred = plate_ocr_preprocessing(image)
    image_pred = image_pred.reshape(1, dsize[0], dsize[1], in_channels)
    image_pred = plate_ocr_model.predict(image_pred)
    image_pred = np.argmax(image_pred, axis=3)
    alphabet = context['alphabet']
    inv_alphabet = {alphabet[char]: char for char in alphabet.keys()}
    image_pred = image_pred.flatten().tolist()
    text_pred = [inv_alphabet[idx] for idx in image_pred]
    text_pred = ''.join(text_pred)
    text_pred = text_pred.upper().strip()
    file_shortname = filename.split('/')[-1].split('.')[0]
    logger.info(f"[{file_shortname}] detected text: {text_pred.upper()}")
    if debug_level > 0:
        font = cv2.FONT_HERSHEY_TRIPLEX
        pos1 = (rectangle_point[0], rectangle_point[1]+100)
        pos2 = (rectangle_point[0]+200, rectangle_point[1]+200)
        image_debug = cv2.rectangle(image_debug, pos1, pos2, (0, 0, 0), -1)
        line = cv2.LINE_AA
        pos = (rectangle_point[0], rectangle_point[1] + 150)
        image_debug = cv2.putText(image_debug, text_pred, pos, font, 1, (0, 255, 0), 2, line)
        save_image(image_debug, f"{out_folder}/image_debug_text_{file_shortname}.png")
    result = {
        'filename': filename,
        'text': text_pred,
    }
    return result


def ocr_plates(params, logger):
    model_file = params['plate_ocr_model_file']
    input_folder = params['input_folder']
    dsize = params['plate_dsize']
    in_channels = params['plate_ocr_model_params']['in_channels']
    out_channels = params['plate_ocr_model_params']['out_channels']
    model_params = params['plate_ocr_model_params']
    alphabet = params['alphabet']
    logger.info("Loading model")
    plate_ocr_model, plate_ocr_preprocessing = plate_ocr_model_def(**model_params)
    plate_ocr_model.load_weights(model_file)
    logger.info("Loading data")
    meta = get_plates_text_metadata(params)
    meta.file_name = 'plate_' + meta.file_name
    meta.file_name = meta.file_name.str.split('.').str[0]+'.png'
    meta = set_index(meta)
    x, _ = get_image_text_label(input_folder, meta, dsize, in_channels, out_channels, alphabet)
    images = map(lambda idx: pred2im(x, dsize, idx, in_channels), range(len(x)))
    context = {
        'plate_ocr_model': plate_ocr_model,
        'plate_ocr_preprocessing': plate_ocr_preprocessing,
        'logger': logger,
    }
    context.update(params)
    events = [{'image': im, 'file': filename, 'ejec_id': ejec_id
               } for ejec_id, filename, im in zip(range(len(meta)), meta.file_name, images)]
    results = map(lambda e: image_ocr(event=e, context=context), events)
    results = map(lambda e: {k: e[k] for k in ('filename', 'text')}, results)
    results = pd.DataFrame(results)
    results.to_csv(f"{params['output_folder']}/ocr_events_results.csv")


if __name__ == "__main__":
    ocr_plates(get_params(), logger)
