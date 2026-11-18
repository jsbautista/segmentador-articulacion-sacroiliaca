
"""U-Net 2D para segmentación de los tres huesos en STIR."""

from collections import OrderedDict

import torch
from torch import nn


def bloque_conv(entrada: int, salida: int) -> nn.Module:
    """Dos convoluciones con GroupNorm y ReLU."""
    capas = nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
        nn.GroupNorm(4, salida),
        nn.ReLU(inplace=True),

        nn.Conv2d(salida, salida, kernel_size=3, padding=1),
        nn.GroupNorm(4, salida),
        nn.ReLU(inplace=True),
    )

    return nn.Sequential(
        OrderedDict([("conv", capas)])
    )


class UNet(nn.Module):
    """U-Net de tres niveles, un canal de entrada y cuatro clases."""

    def __init__(self):
        super().__init__()

        self.enc1 = bloque_conv(1, 16)
        self.enc2 = bloque_conv(16, 32)
        self.enc3 = bloque_conv(32, 64)

        self.centro = bloque_conv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = bloque_conv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = bloque_conv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec1 = bloque_conv(32, 16)

        self.salida = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, imagenes: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(imagenes)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        centro = self.centro(self.pool(e3))

        d3 = self.dec3(
            torch.cat((self.up3(centro), e3), dim=1)
        )

        d2 = self.dec2(
            torch.cat((self.up2(d3), e2), dim=1)
        )

        d1 = self.dec1(
            torch.cat((self.up1(d2), e1), dim=1)
        )

        return self.salida(d1)
