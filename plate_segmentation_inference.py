import pathlib

import cv2
import pandas as pd
from loguru import logger

from io_utils.data_source import (
    get_image_label_gen, load_label_data)
from cv.image_processing import (
    get_contours_rgb,
    print_named_images,
    get_warping,
    warp_image,
    pred2im,
    get_min_area_rectangle,
)
# from cv.seg_models.model_definition import get_model_definition
from cv.tensorflow_models.unet import get_model_definition


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/data/plates'
    dsize = (576, 576)
    params = {
        'folder': folder,
        'dsize': dsize,
        'model_file': f'{folder}/model/best_model.h5',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
        'model_params': {
            'img_height': dsize[0],
            'img_width': dsize[1],
            'in_channels': 3,
            'out_channels': 2,
        }
    }
    return params


def draw_rectangle(im, r):
    x, y, w, h = r
    im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return im


def segment_plates(params, logger):
    model_file = params['model_file']
    labels = params['labels']
    metadata = params['metadata']
    folder = params['folder']
    dsize = params['dsize']
    #
    plate_shape = (200, 50)
    min_area = 20 * 80
    max_area = 100 * 200
    thickness = 3
    color = (255, 0, 0)
    contours_color = (0, 0, 255)
    debug_level = 0
    #
    out_folder = f"{folder}/output_plate_segmentation"
    logger.info("Loading model")
    model_params = params['model_params']
    model, preprocess_input = get_model_definition(**model_params)
    model.load_weights(model_file)
    logger.info("Loading data")
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = pd.read_csv(metadata)
    metadata = metadata.merge(labels, on=['image'], how='left')
    metadata = metadata.assign(idx=range(len(metadata)))
    images, _ = get_image_label_gen(folder, metadata, dsize)
    images = [pred2im(images, dsize, idx) for idx in range(len(images))]
    logger.info("preprocess_input")
    images_pred = [preprocess_input(im) for im in images]
    logger.info("Inference")
    images_pred = [im.reshape(1, dsize[0], dsize[0], 3) for im in images_pred]
    images_pred = [(model.predict(im) * 255).round() for im in images_pred]
    images = [im.reshape(dsize[0], dsize[0], 3) for im in images]
    images_pred = [pred2im(y, dsize, 0) for y in images_pred]
    logger.info("Getting contours")
    contours = [get_contours_rgb(im, min_area, max_area) for im in images_pred]
    logger.info("Draw contours")
    images_pred = [cv2.drawContours(
        im, c, -1, color, thickness, 8
    ) for im, c in zip(images_pred, contours)]
    print_named_images(images_pred, out_folder, "contours", logger)
    logger.info("Straight bounding boxes")
    bounding_boxes = [cv2.boundingRect(c[0]) for c in contours]
    images_pred = [draw_rectangle(im, b) for im, b in
                   zip(images_pred, bounding_boxes)]
    logger.info("Min area bounding boxes")
    bounding_boxes = [get_min_area_rectangle(c) for c in contours]
    if debug_level > 0:
        images_pred = [cv2.drawContours(
            im, [r], 0, contours_color, thickness
        ) if r is not None else im for im, r in
                       zip(images_pred, bounding_boxes)]
        print_named_images(images_pred, out_folder,
                           "min_area_bounding_boxes", logger)
    logger.info("Warp images")
    warpings = [get_warping(q, plate_shape) for q in bounding_boxes]
    images_pred = [warp_image(
        im, w, plate_shape) for (im, w) in zip(images, warpings)]
    print_named_images(images_pred, out_folder, "plates", logger)


if __name__ == "__main__":
    segment_plates(get_params(), logger)
