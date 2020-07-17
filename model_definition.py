import segmentation_models as sm
from segmentation_models import (Linknet)
from segmentation_models import get_preprocessing
from cv.tensorflow_utils import keras


def get_model_definition():
    backbone = 'mobilenet'
    # backbone = 'seresnet18'
    # backbone = 'resnet18'
    # backbone = 'efficientnetb0'
    n_classes = 2
    lr = 0.001
    activation = 'softmax'
    pre_process_input = get_preprocessing(backbone)
    optimizer = keras.optimizers.Adam(lr)
    metrics = [
        sm.metrics.FScore(threshold=0.5),
    ]
    model = Linknet(backbone, classes=n_classes, activation=activation,
                    encoder_freeze=True)
    if n_classes == 1:
        loss = sm.losses.BinaryFocalLoss()
    else:
        loss = sm.losses.CategoricalFocalLoss()
    model.compile(optimizer, loss, metrics)
    return model, pre_process_input
