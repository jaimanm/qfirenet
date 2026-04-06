from torch import nn

from models.unet import DoubleConv, Down, OutConv, Up


class CompactUNet(nn.Module):
    """Width-scaled U-Net encoder-decoder with skip connections."""

    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        config = config or {}

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = config.get('base_channels', 48)

        # Scale the full U-Net width from a single config value.
        width_1 = self.base_channels
        width_2 = self.base_channels * 2
        width_3 = self.base_channels * 4
        width_4 = self.base_channels * 8
        factor = 2 if bilinear else 1
        bottleneck_width = (self.base_channels * 16) // factor

        self.inc = DoubleConv(n_channels, width_1)
        self.down1 = Down(width_1, width_2)
        self.down2 = Down(width_2, width_3)
        self.down3 = Down(width_3, width_4)
        self.down4 = Down(width_4, bottleneck_width)

        self.up1 = Up(width_4 * 2, width_4 // factor, bilinear)
        self.up2 = Up(width_3 * 2, width_3 // factor, bilinear)
        self.up3 = Up(width_2 * 2, width_2 // factor, bilinear)
        self.up4 = Up(width_1 * 2, width_1, bilinear)
        self.outc = OutConv(width_1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
