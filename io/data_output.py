import numpy as np
from PIL import Image
import os


def save_inference_images(y_pred, metadata, folder):
    output_folder = f"{folder}/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for id_im, im_file in zip(metadata.idx, metadata.image):
        y_im = y_pred[id_im,:,:,0]
        new_shape = (y_im.shape[0], y_im.shape[1])
        y_im = np.reshape(y_im, new_shape)
        y_im = Image.fromarray(y_im)
        y_im = y_im.convert("L")
        im_filename = im_file.split('.')[0]
        filename = f"{output_folder}/{im_filename}_pred.jpg"
        y_im.save(filename)
