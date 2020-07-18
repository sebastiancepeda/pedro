import torch

from cv.pytorch.unet import (UNet, crop_img)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()


def standardize(tensor, size, im_channels):
    if type(tensor) != torch.Tensor:
        t_shape = tensor.shape
        new_shape = (t_shape[0], t_shape[3], t_shape[1], t_shape[2])
        tensor = torch.tensor(tensor)
        tensor = torch.reshape(tensor, new_shape)
    return tensor


def train_model(x_train, y_train, x_val, y_val, model_definition, model_params,
                params, logger):
    size = params['dsize']
    im_channels = params['im_channels']
    x_train = standardize(x_train, size, im_channels)
    y_train = standardize(y_train, size, im_channels)
    x_val = standardize(x_val, size, im_channels)
    y_val = standardize(y_val, size, im_channels)
    model = model_definition(**model_params)
    epochs = 10
    delta = 1 + (epochs // 10)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    logger.info(f"Iterating through epochs")
    shape1 = x_train.shape[1:4]
    shape2 = y_train.shape[1:4]
    for t in range(epochs):
        for idx in range(len(x_train)):
            x = torch.reshape(x_train[idx, :, :, :], (1, *shape1))
            y = torch.reshape(y_train[idx, :, :, :], (1, *shape2))
            # print(x.shape)
            # print(y.shape)
            y_pred = model(x)
            loss = criterion(y_pred, crop_img(y, y_pred))
            if t % delta == (delta - 1):
                logger.info(f"Loss [{t}]: {loss.item()}")
                save_model(model, params['model_file'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    load_model(model_definition, params['model_file'], **model_params)
    return model


def test_train_model():
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


if __name__ == "__main__":
    test_train_model()
