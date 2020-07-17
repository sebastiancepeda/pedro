"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import glob
import pathlib

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from data_source import (
    load_label_data)
from cv.image_processing import (
    print_named_images,
    has_dark_font,
    get_binary_im,
    print_limits,
    get_contours_gray,
)


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/data/plates'
    params = {
        'folder': folder,
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
    }
    return params


def segment_plates(params):
    labels = params['labels']
    metadata = params['metadata']
    folder = params['folder']
    #
    thickness = 3
    color = (255, 0, 0)
    #
    input_folder = f"{folder}/output_plate_segmentation"
    output_folder = f"{folder}/output"
    print("Loading model")
    print("Loading data")
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = pd.read_csv(metadata)
    metadata = metadata.merge(labels, on=['image'], how='left')
    metadata = metadata.assign(idx=range(len(metadata)))
    images = glob.glob(f"{input_folder}/plates*.png")
    images = [cv2.imread(im) for im in images]
    for im in images:
        assert im is not None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    images = [clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in
              images]
    images = [255 - im if has_dark_font(im) else im for im in images]
    # print_named_images(images, output_folder, "images")
    binary_ims = [get_binary_im(im) for im in images]
    edges = [cv2.Canny(im, 100, 200) for im in binary_ims]
    # print_named_images(edges, output_folder, "edges")
    debug_images = [np.concatenate((im, binary, edge), axis=1)
                    for im, binary, edge in zip(images, binary_ims, edges)]
    print_named_images(debug_images, output_folder, "binary_images")
    binary_ims = [print_limits(im) for im in binary_ims]
    # print_named_images(images, output_folder, "images_mask")
    print("Getting contours")
    min_area = 14 ** 2
    max_area = 40 ** 2
    contours = [get_contours_gray(im, min_area, max_area) for im in binary_ims]
    print("Draw contours")
    binary_ims = [cv2.drawContours(
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), c, -1, color, thickness, 8
    ) for im, c in zip(binary_ims, contours)]
    print_named_images(binary_ims, output_folder, "mask_contours")


if __name__ == "__main__":
    print = logger.info
    segment_plates(get_params())
