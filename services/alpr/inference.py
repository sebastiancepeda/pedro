import glob
import pandas as pd

from cv.tensorflow_models.unet2text3 import (normalize_image_shape)
from cv.tensorflow_models.unet_little import get_model_definition as plate_seg_model_def
from cv.tensorflow_models.unet2text3 import get_model_definition as plate_ocr_model_def
from services.plate_segmentation.inference import plate_segmentation
from services.plate_ocr.inference import image_ocr


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/input/'
    output_folder = f'{path}/plates/alpr'
    dsize = (256, 256)
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    big_shape = (1024, 1024)
    plate_shape = (200, 50)
    color = (255, 0, 0)
    thickness = 3
    debug_level = 1
    min_pct = 0.04
    max_pct = 0.20
    ocr_height, ocr_width = normalize_image_shape(50, 200)
    min_area = (big_shape[0] * min_pct) * (big_shape[1] * min_pct)
    max_area = (big_shape[0] * max_pct) * (big_shape[1] * max_pct)
    train_files = glob.glob(f"{input_folder}/train/*.jpg")
    test_files = glob.glob(f"{input_folder}/test/*.jpg")
    files = train_files + test_files
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'files': files,
        'plate_shape': plate_shape,
        'color': color,
        'thickness': thickness,
        'debug_level': debug_level,
        'dsize': dsize,
        'plate_dsize': (ocr_height, ocr_width),
        'plate_segmentation_model_file': f"{path}/plates/plate_segmentation/model/best_model.h5",
        'plate_ocr_model_file': f"{path}/plates/plate_ocr/model/best_model.h5",
        'alphabet': alphabet,
        'big_shape': big_shape,
        'min_area': min_area,
        'max_area': max_area,
        'plate_ocr_model_params': {
            'img_height': ocr_height,
            'img_width': ocr_width,
            'in_channels': 1,
            'out_channels': len(alphabet),
        },
        'plate_segmentation_model_params': {
            'img_height': 256,
            'img_width': 256,
            'in_channels': 3,
            'out_channels': 2,
        },
    }
    return params


def alpr_inference(params):
    from loguru import logger

    logger.add(f"{params['output_folder']}/logger.log")
    logger.info("Loading model")
    plate_segmentation_model, plate_segmentation_preprocessing = plate_seg_model_def(**params['plate_segmentation_model_params'])
    plate_segmentation_model.load_weights(params['plate_segmentation_model_file'])
    plate_ocr_model, plate_ocr_preprocessing = plate_ocr_model_def(**params['plate_ocr_model_params'])
    plate_ocr_model.load_weights(params['plate_ocr_model_file'])
    logger.info("Loading data")
    files = params['files']
    context = {
        'logger': logger,
        'plate_ocr_model': plate_ocr_model,
        'plate_ocr_preprocessing': plate_ocr_preprocessing,
        'plate_segmentation_model': plate_segmentation_model,
        'plate_segmentation_preprocessing': plate_segmentation_preprocessing,
    }
    context.update(params)
    events = [{'image_file': f, 'ejec_id': ejec_id} for ejec_id, f in enumerate(files)]
    results = map(lambda e: plate_segmentation(event=e, context=context), events)
    results = map(lambda r: {'image': r['image'], 'filename': r['file']}, results)
    results = map(lambda e: image_ocr(event=e, context=context), results)
    results = map(lambda e: {k: e[k] for k in ('filename', 'text')}, results)
    results = pd.DataFrame(results)
    results.to_csv(f"{params['output_folder']}/ocr_events_results.csv")


if __name__ == "__main__":
    alpr_inference(get_params())
