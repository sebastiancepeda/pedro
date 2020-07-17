import segmentation_models as sm

from cv.tensorflow.tensorflow_utils import keras


def get_model_definition():
    backbone = 'mobilenet'
    n_classes = 2
    lr = 0.001
    activation = 'softmax'
    pre_process_input = sm.get_preprocessing(backbone)
    optimizer = keras.optimizers.Adam(lr)
    metrics = [
        sm.metrics.FScore(threshold=0.5),
    ]
    model = sm.Linknet(backbone, classes=n_classes, activation=activation,
                       encoder_freeze=True)
    if n_classes == 1:
        loss = sm.losses.BinaryFocalLoss()
    else:
        loss = sm.losses.CategoricalFocalLoss()
    model.compile(optimizer, loss, metrics)
    return model, pre_process_input
