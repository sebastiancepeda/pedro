"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import torch

from cv.pytorch.unet_small import UNetSmall


def test_unet_small():
    image = torch.rand((1, 1, 572, 572))
    model = UNetSmall(in_channels=1, out_channels=2)
    y = model(image)
    print(y.size())
