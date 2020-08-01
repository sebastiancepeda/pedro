import cv2
import pandas as pd
from loguru import logger

from cv.image_processing import (
    get_contours_rgb,
    print_named_images,
    get_warping,
    warp_image,
    pred2im,
    get_min_area_rectangle,
)
from cv.tensorflow_models.unet_little import get_model_definition
from io_utils.data_source import (
    get_image_label_gen, get_plates_text_area_metadata)


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/input'
    output_folder = f'{path}/plates/output_plate_segmentation'
    # dsize = (576, 576)
    dsize = (256, 256)
    # alphabet = '0p'
    alphabet = [' ', 'plate']
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 3
    out_channels = len(alphabet)
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'labels': f"{input_folder}/labels_plate_text.json",
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
    if (r is not None) and (len(r) > 0):
        x, y, w, h = r
        im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return im


def segment_plates(params, logger):
    model_file = params['model_file']
    input_folder = params['input_folder']
    output_folder = params['output_folder']
    dsize = params['dsize']
    in_channels = params['model_params']['in_channels']
    out_channels = params['model_params']['out_channels']
    # Constants
    plate_shape = (200, 50)
    color = (255, 0, 0)
    min_area = 10 * 40
    max_area = 100 * 200
    thickness = 3
    debug_level = 0
    #
    logger.info("Loading model")
    model_params = params['model_params']
    model, preprocess_input = get_model_definition(**model_params)
    model.load_weights(model_file)
    logger.info("Loading data")
    meta = get_plates_text_area_metadata(params)
    meta_im_idx = meta.image_name.unique()
    meta_im_idx = pd.DataFrame(data={
        'image_name': meta_im_idx,
        'idx': range(len(meta_im_idx)),
    })
    meta = meta.merge(meta_im_idx, on=['image_name'], how='left')
    images, im_labels = get_image_label_gen(input_folder, meta, dsize, in_channels, out_channels, params)
    im_labels = [pred2im(im_labels*255, dsize, idx, 1) for idx in range(len(im_labels))]
    # print_named_images(im_labels, meta, output_folder, "im_labels", logger)
    images = [pred2im(images, dsize, idx, in_channels) for idx in range(len(images))]
    logger.info("Pre process input")
    ims_pred = [preprocess_input(im) for im in images]
    logger.info("Inference")
    ims_pred = [im.reshape(1, dsize[0], dsize[0], 3) for im in ims_pred]
    ims_pred = [(model.predict(im) * 255).round() for im in ims_pred]
    images = [im.reshape(dsize[0], dsize[0], 3) for im in images]
    ims_pred = [pred2im(y, dsize, 0, 3) for y in ims_pred]
    # print_named_images(ims_pred, meta, output_folder, "im_pred", logger)
    logger.info("Getting contours")
    contours = [get_contours_rgb(im, min_area, max_area) for im in ims_pred]
    logger.info("Draw contours")
    ims_pred = [cv2.drawContours(
        im, c, -1, color, thickness, 8) for im, c in zip(ims_pred, contours)]
    logger.info("Straight bounding boxes")
    boxes = [cv2.boundingRect(c[0]) if len(c) > 0 else None for c in contours]
    ims_pred = [draw_rectangle(im, b) for im, b in zip(ims_pred, boxes)]
    logger.info("Min area bounding boxes")
    boxes = [get_min_area_rectangle(c) for c in contours]
    if debug_level > 0:
        ims_pred = [cv2.drawContours(im, [r], 0, color, thickness) for im, r in zip(ims_pred, boxes)]
        print_named_images(ims_pred, meta, output_folder, "min_area_boxes", logger)
    logger.info("Warp images")
    warpings = [get_warping(q, plate_shape) for q in boxes]
    ims_pred = [warp_image(im, w, plate_shape) for (im, w) in zip(images, warpings)]
    print_named_images(ims_pred, meta, output_folder, "plates", logger)


if __name__ == "__main__":
    segment_plates(get_params(), logger)