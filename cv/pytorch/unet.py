"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import torch
import torch.nn as nn
from loguru import logger


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(in_tensor, out_tensor):
    out_size = out_tensor.size()[2]
    in_size = in_tensor.size()[2]
    delta = in_size - out_size
    delta = delta // 2
    result = in_tensor[:, :, delta:in_size - delta, delta:in_size - delta]
    return result


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Down convolutions
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        # Up convolutions
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, image):
        # Encoder
        x1 = self.down_conv_1(image)
        x3 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x3)
        x5 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x5)
        x7 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x7)
        x = self.max_pool_2x2(x7)
        x = self.down_conv_5(x)
        # Decoder
        x = self.up_trans_1(x)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        x = self.out(x)
        return x


if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet(in_channels=1, out_channels=2)
    y = model(image)
    logger.info(y.size())
