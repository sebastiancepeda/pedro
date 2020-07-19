"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import numpy as np

from cv.tensorflow_models.model_definition import get_model_definition
from cv.tensorflow_models.tensorflow_utils import train_model


def test_train_model():
    from loguru import logger

    logger.info(f"Start")
    dsize = (576, 576)
    params = {
        'dsize': dsize,
        'im_channels': 1,
        'epochs': 10,
        'model_file': 'test.model',
        'model_folder': './',
    }
    x_train = np.random.rand(1, dsize[0], dsize[1], 1)
    y_train = np.random.rand(1, dsize[0], dsize[1], 2)
    x_val = np.random.rand(1, dsize[0], dsize[1], 1)
    y_val = np.random.rand(1, dsize[0], dsize[1], 2)
    logger.info(f"Call to train_model")
    model_params = {
        'img_height': dsize[0],
        'img_width': dsize[1],
        'in_channels': 1,
        'out_channels': 2,
    }
    trained_model = train_model(
        x_train, y_train, x_val, y_val, get_model_definition, model_params,
        params, logger)
    assert trained_model is not None
    print(trained_model)


if __name__ == '__main__':
    test_train_model()
