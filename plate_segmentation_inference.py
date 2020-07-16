import pathlib

import cv2
import pandas as pd
from loguru import logger

from data_source import (
    get_image_label_gen, load_label_data)
from image_processing import (
    get_quadrilateral,
    get_contours,
    print_named_images,
    get_warping,
    warp_image,
    pred2im,
)
from model_definition import get_model_definition


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/plates'
    params = {
        'folder': folder,
        'dsize': (768, 768),
        'model_file': f'{folder}/model/best_model.h5',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
    }
    return params


def segment_plates(params):
    model_file = params['model_file']
    labels = params['labels']
    metadata = params['metadata']
    folder = params['folder']
    dsize = params['dsize']
    #
    output_folder = f"{folder}/output_plate_segmentation"
    print("Loading model")
    model, preprocess_input = get_model_definition()
    model.load_weights(model_file)
    plate_shape = (200, 50)
    min_area = 20*80
    max_area = 100*200
    thickness = 3
    color = (255, 0, 0)
    contours_color = (0, 255, 0)
    print("Loading data")
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = pd.read_csv(metadata)
    metadata = metadata.merge(labels, on=['image'], how='left')
    metadata = metadata.assign(idx=range(len(metadata)))
    images, _ = get_image_label_gen(folder, metadata, dsize)
    images = [pred2im(images, dsize, idx) for idx in range(len(images))]
    print("preprocess_input")
    images_pred = [preprocess_input(im) for im in images]
    print("Inference")
    images_pred = [im.reshape(1, dsize[0], dsize[0], 3) for im in images_pred]
    images_pred = [(model.predict(im) * 255).round() for im in images_pred]
    images = [im.reshape(dsize[0], dsize[0], 3) for im in images]
    images_pred = [pred2im(y, dsize, 0) for y in images_pred]
    print("Getting contours")
    contours = [get_contours(im, min_area, max_area) for im in images_pred]
    print("Draw contours")
    images_pred = [cv2.drawContours(
        im, c, -1, color, thickness, 8
    ) for im, c in zip(images_pred, contours)]
    print("Getting quadrilaterals")
    contours = [get_quadrilateral(
        c[0]) if c is not None and len(c) > 0 else None for c in contours]
    print("Drawing quadrilaterals")
    images_pred = [cv2.drawContours(
        im, [r], 0, contours_color, thickness
    ) if r is not None else im for im, r in zip(images_pred, contours)]
    print_named_images(images_pred, output_folder, "quadrilaterals")
    print("Warp images")
    warpings = [get_warping(q, plate_shape) for q in contours]
    images_pred = [warp_image(
        im, w, plate_shape) for (im, w) in zip(images, warpings)]
    print_named_images(images_pred, output_folder, "plates")


if __name__ == "__main__":
    print = logger.info
    segment_plates(get_params())
