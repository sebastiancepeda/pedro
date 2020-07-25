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

from cv.image_processing import (
    print_named_images,
    has_dark_font,
    get_binary_im,
    get_contours_binary,
    draw_lines,
)
from io_utils.data_source import (
    load_label_data)


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/data/plates'
    params = {
        'folder': folder,
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
    }
    return params


def segment_plates(logger, params):
    labels = params['labels']
    metadata = params['metadata']
    folder = params['folder']
    #
    thickness = 3
    color = (255, 0, 0)
    min_area = 50 ** 2
    max_area = 1000 ** 2
    #
    input_folder = f"{folder}/output_plate_segmentation"
    output_folder = f"{folder}/output"
    logger.info("Loading model")
    logger.info("Loading data")
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
    images_sobel_y = [cv2.Sobel(im, cv2.CV_8U, 0, 1, ksize=5) for im in images]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    images = [clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in
              images]
    images = [255 - im if has_dark_font(im) else im for im in images]
    # print_named_images(images_sobel_y, output_folder, "images_sobel_y", logger)
    degrees = 90
    g_kernel = cv2.getGaborKernel(
        ksize=(21, 21),
        sigma=1.0,  # 8.0,
        theta=degrees * np.pi / 180,
        lambd=10.0,
        gamma=0.6,  # 0.5,
        psi=0,
        ktype=cv2.CV_32F)
    g_kernel = g_kernel - g_kernel.mean()
    images_gabor = [cv2.filter2D(im, cv2.CV_8UC3, g_kernel) for im in images]
    print_named_images(images_gabor, metadata, output_folder, "images_gabor", logger)
    images_sobel_canny = [cv2.Canny(im, 100, 200) for im in images_sobel_y]
    # print_named_images(images_sobel_canny, output_folder, "images_sobel_canny", logger)
    binary_ims = [get_binary_im(im) for im in images]
    # print_named_images(binary_ims, output_folder, "binary_ims", logger)
    images = [cv2.blur(im, (3, 3)) for im in images]
    edges_set = [cv2.Canny(im, 100, 200) for im in images]
    contours = [get_contours_binary(im, min_area, max_area) for im in
                edges_set]
    debug_contours = [cv2.drawContours(
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), c, -1, color, thickness, 8
    ) for im, c in zip(edges_set, contours)]
    # print_named_images(debug_contours, output_folder, "debug_contours", logger)
    lines_set = [
        cv2.HoughLinesP(
            e, rho=2, theta=1 * np.pi / 180, threshold=100, minLineLength=100,
            maxLineGap=10
        ) for e in edges_set]
    debug_lines = [draw_lines(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), lines) for
                   im, lines in zip(edges_set, lines_set)]
    # print_named_images(debug_lines, output_folder, "debug_lines", logger)
    debug_images = [np.concatenate((im, binary, edge, im_gabor), axis=1)
                    for im, binary, edge, im_gabor in zip(
            images, binary_ims, edges_set, images_gabor)]
    print_named_images(debug_images, metadata, output_folder, "binary_images", logger)
    """
    binary_ims = [print_limits(im) for im in binary_ims]
    # print_named_images(images, output_folder, "images_mask", logger)
    logger.info("Getting contours")
    contours = [get_contours_binary(im, min_area, max_area) for im in
                binary_ims]
    logger.info("Draw contours")
    binary_ims = [cv2.drawContours(
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), c, -1, color, thickness, 8
    ) for im, c in zip(binary_ims, contours)]
    print_named_images(binary_ims, output_folder, "mask_contours", logger)
    """


if __name__ == "__main__":
    segment_plates(logger, get_params())
