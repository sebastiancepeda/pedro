"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import torch

from cv.pytorch.unet import UNet


def test_unet():
    image = torch.rand((1, 1, 572, 572))
    model = UNet(in_channels=1, out_channels=2)
    y = model(image)
    print(y.size())
