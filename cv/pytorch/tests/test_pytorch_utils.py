"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import torch

from cv.pytorch.pytorch_utils import train_model
from cv.pytorch.unet import (UNet)


def train_model():
    from loguru import logger

    logger.info(f"Start")
    params = {
        'path': './test.model',
        'dsize': (572, 572),
        'im_channels': 1,
    }
    x_train = torch.rand((1, 1, 572, 572))
    y_train = torch.rand((1, 2, 388, 388))
    x_val = torch.rand((1, 1, 572, 572))
    y_val = torch.rand((1, 2, 388, 388))
    logger.info(f"Call to train_model")
    model_params = {
        'in_channels': 1,
        'out_channels': 2,
    }
    trained_model = train_model(x_train, y_train, x_val, y_val, UNet,
                                model_params, params, logger)
    assert trained_model is not None
    print(trained_model)
