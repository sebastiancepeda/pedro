import cv2
import numpy as np
from loguru import logger
import nltk
import skimage.transform
from skimage.util import img_as_ubyte

from cv.image_processing import (
    pred2im, print_images
)
from cv.tensorflow_models.unet2text2 import (
    get_model_definition, normalize_image_shape)
from io_utils.data_source import (
    get_image_text_label, get_plates_text_metadata)
from io_utils.utils import set_index


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/plate_segmentation'
    output_folder = f'{path}/plates/plate_ocr'
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


def draw_rectangle(im, r):
    x, y, w, h = r
    im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return im


def ocr_plates(params, logger):
    model_file = params['model_file']
    input_folder = params['input_folder']
    dsize = params['dsize']
    out_folder = params['output_folder']
    in_channels = params['model_params']['in_channels']
    out_channels = params['model_params']['out_channels']
    model_params = params['model_params']
    alphabet = params['alphabet']
    logger.info("Loading model")
    model, preprocess_input = get_model_definition(**model_params)
    model.load_weights(model_file)
    logger.info("Loading data")
    meta = get_plates_text_metadata(params)
    meta.file_name = 'plate_' + meta.file_name
    meta.file_name = meta.file_name.str.split('.').str[0]+'.png'
    meta = set_index(meta)
    images, _ = get_image_text_label(input_folder, meta, dsize, in_channels, out_channels, alphabet)
    images = [pred2im(images, dsize, idx, in_channels) for idx in range(len(images))]
    logger.info("Pre process input")
    images_pred = [preprocess_input(im) for im in images]
    logger.info("Inference")
    images_pred = [im.reshape(1, dsize[0], dsize[1], in_channels) for im in images_pred]
    images_pred = [model.predict(im) for im in images_pred]
    images_pred = [np.argmax(im, axis=3) for im in images_pred]
    print_images(images, meta, out_folder, "images_text", logger)
    alphabet = params['alphabet']
    inv_alphabet = {alphabet[char]: char for char in alphabet.keys()}
    texts = []
    for y, im_name, text in zip(images_pred, meta.image_name, meta.plate):
        y = y.flatten().tolist()
        text_pred = [inv_alphabet[idx] for idx in y]
        text_pred = ''.join(text_pred)
        text_pred = text_pred.upper().strip()
        # logger.info(f"Text {im_name: <7}: {text_pred.upper()} - {text}")
        texts.append(text_pred)
    meta['text_pred'] = texts
    meta["edit_distance"] = meta.loc[:, ["plate", "text_pred"]].apply(lambda row: nltk.edit_distance(*row), axis=1)
    meta.to_csv(f"{out_folder}/results.csv")
    font = cv2.FONT_HERSHEY_TRIPLEX
    clr = (0, 255, 0)
    pos = (30, 30)
    line = cv2.LINE_AA
    images = [cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) for im in images]
    images = [cv2.putText(im, txt, pos, font, 1, clr, 2, line) for im, txt in zip(images, texts)]
    print_images(images, meta, out_folder, "images_text", logger)
    # afine_tf = skimage.transform.AffineTransform(shear=0.1)
    # images = [skimage.transform.warp(im, inverse_map=afine_tf) for im in images]
    # images = [img_as_ubyte(im) for im in images]
    # print_images(images, meta, out_folder, "images_text_warp", logger)


if __name__ == "__main__":
    ocr_plates(get_params(), logger)
