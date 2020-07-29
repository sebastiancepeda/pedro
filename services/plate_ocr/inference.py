import cv2
import numpy as np
import pandas as pd
from loguru import logger

from cv.image_processing import (
    get_contours_rgb, print_named_images, get_warping,
    warp_image, pred2im, get_min_area_rectangle,
)
from cv.tensorflow_models.unet_little import (
    get_model_definition, normalize_image_shape)
from io_utils.data_source import (
    get_image_label_gen, get_plates_text_metadata)


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
    # Constants
    dsize_cv2 = (dsize[1], dsize[0])
    color = (255, 0, 0)
    min_area = 10 * 40
    max_area = 100 * 200
    thick = 3
    debug_level = 0
    #
    logger.info("Loading model")
    model_params = params['model_params']
    model, preprocess_input = get_model_definition(**model_params)
    model.load_weights(model_file)
    logger.info("Loading data")
    metadata = get_plates_text_metadata(params)
    metadata_idx = metadata.image_name.unique()
    metadata_idx = pd.DataFrame(data={
        'image_name': metadata_idx,
        'idx': range(len(metadata_idx)),
    })
    metadata = metadata.merge(metadata_idx, on=['image_name'], how='left')
    images, _ = get_image_label_gen(input_folder, metadata, dsize,
                                    in_channels, out_channels, params)
    images = [pred2im(images, dsize, idx, in_channels) for idx in range(len(images))]
    logger.info("Pre process input")
    images_pred = [preprocess_input(im) for im in images]
    logger.info("Inference")
    images_pred = [im.reshape(1, dsize[0], dsize[1], in_channels) for im in
                   images_pred]
    images_pred = [model.predict(im) for im in images_pred]
    images_pred = [np.argmax(im, axis=3) for im in images_pred]
    images = [im.reshape(dsize[0], dsize[1], in_channels) for im in images]
    images_pred = [(im * 100).astype('uint8').reshape(dsize[0], dsize[1]) for
                   im in images_pred]
    im_pred_b = [((im > 0) * 255).astype('uint8') for im in images_pred]
    im_pred_b = [cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) for im in im_pred_b]
    print_named_images(im_pred_b, metadata, out_folder, "binary", logger)
    images_pred = [cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) for im in images_pred]
    print_named_images(images_pred, metadata, out_folder, "argmax", logger)
    logger.info("Getting contours")
    contours = [get_contours_rgb(im, min_area, max_area) for im in im_pred_b]
    logger.info("Draw contours")
    images_pred = [cv2.drawContours(
        im, c, -1, color, thick, 8) for im, c in zip(images_pred, contours)]
    logger.info("Min area bounding boxes")
    boxes = [get_min_area_rectangle(c) for c in contours]
    im_boxes = [cv2.drawContours(im[:,:,0], [r], 0, color, thick) for im, r in
                   zip(images, boxes)]
    print_named_images(im_boxes, metadata, out_folder, "boxes", logger)
    logger.info("Warp images")
    warpings = [get_warping(q, dsize_cv2) for q in boxes]
    images_pred = [warp_image(im, w, dsize_cv2) for (im, w) in
                   zip(images, warpings)]
    print_named_images(images_pred, metadata, out_folder, "plates", logger)
    debug_images = [np.concatenate((im[:, :, 0], im2[:,:,0]), axis=1) for im, im2 in
                    zip(images, im_pred_b)]
    print_named_images(debug_images, metadata, out_folder, "model_output",
                       logger)


if __name__ == "__main__":
    ocr_plates(get_params(), logger)
