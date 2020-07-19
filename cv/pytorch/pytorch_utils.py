import torch

from cv.pytorch.unet import (crop_img)


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
            y_pred = model(x)
            loss = criterion(y_pred, crop_img(y, y_pred))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch, image: [{t}, {idx}]")
        if t % delta == (delta - 1):
            logger.info(f"Loss [{t}]: {loss.item()}")
            save_model(model, params['model_file'])
    load_model(model_definition, params['model_file'], **model_params)
    return model
