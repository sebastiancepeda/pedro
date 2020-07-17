import torch

from cv.unet import UNet


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()


def train_model(model, x, y, params, logger):
    epochs = 500
    delta = 1 + (epochs // 10)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    logger.info(f"Iterating through epochs")
    for t in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if t % delta == (delta - 1):
            logger.info(f"Loss [{t}]: {loss.item()}")
            save_model(model, params['path'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


if __name__ == "__main__":
    from loguru import logger

    logger.info(f"Start")
    params = {
        'path': './test.model',
    }
    x = torch.rand((1, 1, 572, 572))
    y = torch.rand((1, 2, 388, 388))
    logger.info(f"Call to train_model")
    trained_model = train_model(UNet(), x, y, params, logger)
