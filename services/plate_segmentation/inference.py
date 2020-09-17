import glob
import cv2
import pandas as pd

from cv.image_processing import (
    get_contours_rgb,
    get_warping,
    warp_image,
    pred2im,
    get_quadrilateral,
    save_image,
    get_rectangle,
)
from cv.tensorflow_models.unet_little import get_model_definition
from io_utils.data_source import (
    get_image,
)
from io_utils.utils import (
    CustomLogger
)


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/input/'
    output_folder = f'{path}/plates/plate_segmentation'
    # dsize = (576, 576)
    dsize = (256, 256)
    # alphabet = '0p'
    alphabet = [' ', 'plate']
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 3
    out_channels = len(alphabet)
    # big_shape = (512, 512)
    big_shape = (1024, 1024)
    plate_shape = (200, 50)
    color = (255, 0, 0)
    thickness = 3
    debug_level = 5
    min_pct = 0.04
    max_pct = 0.20
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
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'labels': f"{input_folder}/labels_plate_text.json",
        'metadata': f"{input_folder}/files.csv",
        'alphabet': alphabet,
        'big_shape': big_shape,
        'min_area': min_area,
        'max_area': max_area,
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


def plate_segmentation(event, context, logger):
    file = event['image_file']
    dsize = context['dsize']
    model = context['model']
    in_channels = context['model_params']['in_channels']
    preprocess_input = context['preprocess_input']
    out_folder = context['output_folder']
    big_shape = context['big_shape']
    min_area = context['min_area']
    max_area = context['max_area']
    debug_level = context['debug_level']
    color = context['color']
    thickness = context['thickness']
    plate_shape = context['plate_shape']
    file_debug_name = file.split('/')[-1]
    file_debug_name = file_debug_name.split('.')[0]
    logger = CustomLogger(file_debug_name, logger)
    x = get_image(file, dsize, in_channels)
    x = pred2im(x, dsize, 0, in_channels)
    logger.info("Pre process input")
    if debug_level > 0:
        x_debug = cv2.resize(x, dsize=big_shape, interpolation=cv2.INTER_CUBIC)
        save_image(x_debug, f"{out_folder}/rectangle_{file_debug_name}_x.png")
    x = preprocess_input(x)
    logger.info("Inference")
    x = x.reshape(1, dsize[0], dsize[0], 3)
    y = (model.predict(x) * 255).round()
    y = pred2im(y, dsize, 0, 3)
    image = get_image(file, big_shape, in_channels)
    image = pred2im(image, big_shape, 0, in_channels)
    image = image.reshape(big_shape[0], big_shape[0], 3)
    y = cv2.resize(y, dsize=big_shape, interpolation=cv2.INTER_CUBIC)
    logger.info("Getting contours")
    contours = get_contours_rgb(y, min_area, max_area)
    if debug_level > 0:
        save_image(255-y, f"{out_folder}/rectangle_{file_debug_name}_y.png")
    if len(contours) > 0:
        # logger.info("Min area bounding box")
        rectangle = get_rectangle(contours)
        # box = get_quadrilateral(contours[0])
        if debug_level > 0:
            logger.info(f"Saving rectangle")
            image_debug = cv2.drawContours(image.copy(), [rectangle], 0, color, thickness)
            save_image(image_debug, f"{out_folder}/rectangle_{file_debug_name}.png")
            # image_debug = cv2.drawContours(image.copy(), [box], 0, color, thickness)
            # logger.info(f"Saving min_area_boxes")
            # save_image(image_debug, f"{out_folder}/min_area_box_{file_debug_name}.png")
        logger.info("Warp images")
        warping = get_warping(rectangle, plate_shape)
        im_pred = warp_image(image, warping, plate_shape)
        logger.info(f"Saving min_area_boxes")
        save_image(im_pred, f"{out_folder}/plate_{file_debug_name}.png")
    else:
        logger.info("Countours not found")
    event_result = {
        'file': file,
        'len_contours': len(contours),
    }
    logger.info(f"event_result: {event_result}")
    return event_result


def segment_plates(params):
    from loguru import logger

    model_file = params['model_file']
    input_folder = params['input_folder']
    logger.info("Loading model")
    model_params = params['model_params']
    model, preprocess_input = get_model_definition(**model_params)
    model.load_weights(model_file)
    logger.info("Loading data")
    files = params['files']
    params_subset = [
        'dsize',
        'model_params',
        'output_folder',
        'big_shape',
        'min_area',
        'max_area',
        'debug_level',
        'color',
        'thickness',
        'plate_shape',
    ]
    context = {
        'model': model,
        'preprocess_input': preprocess_input,
    }
    context.update({k: params[k] for k in params_subset})
    events = [{'image_file': f, 'ejec_id': ejec_id} for ejec_id, f in enumerate(files)]
    events_results = []
    for event in events:
        event_result = plate_segmentation(event, context, logger)
        events_results.append(event_result)
    events_results = pd.DataFrame(events_results)
    events_results.to_csv(f"{params['output_folder']}/events_results.csv")


if __name__ == "__main__":
    segment_plates(get_params())
