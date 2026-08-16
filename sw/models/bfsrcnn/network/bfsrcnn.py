import math
from torch import nn
import torch.nn.functional as F
from models.bfsrcnn.network.bbcu import BinaryConv2d, BinaryUpConv2d, BinaryDSConv2d

class BFSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=48, s=16, m=8):
        super(BFSRCNN, self).__init__()

        self.scale_factor = scale_factor

        # self.feature_extraction = nn.Sequential(
        #     nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
        #     nn.PReLU(d)
        # )

        # self.shrinking = nn.Sequential(
        #     nn.Conv2d(d, s, kernel_size=1),
        #     nn.PReLU(s)
        # )

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=3, padding=3//2),
            nn.PReLU(d)
        )

        self.shrinking = nn.Sequential(
            # BinaryConv2d(d, 36, 3),

            nn.Conv2d(d, 32, kernel_size=1),
            nn.ReLU(32),

            nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=3//2),
            nn.ReLU(32),
            nn.Conv2d(32, s, kernel_size=1),

            # BinaryDSConv2d(36, s, 3)
        )

        self.mapping = nn.Sequential(
            *[BinaryConv2d(s, s, 3) for _ in range(m)]
            # *[BinaryDSConv2d(s, s, 3) for _ in range(m)]
        )

        self.expanding = nn.Sequential(
            # BinaryUpConv2d(s, 3, scale_factor),

            nn.Conv2d(s, 8 * scale_factor**2, kernel_size=1),
            nn.PReLU(8 * scale_factor**2),
            nn.PixelShuffle(scale_factor),

            nn.Conv2d(8, 1, kernel_size=3, padding=3//2),
            # nn.Conv2d(s, 1, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for part in [
            self.feature_extraction,
            self.shrinking,
            self.mapping,
            self.expanding,
        ]:
            for m in part.modules():
                if isinstance(m, nn.Conv2d):
                    # print('init', m)
                    nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=math.sqrt(
                            2 / (m.out_channels * m.weight[0][0].numel())
                        ),
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):

        residual = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = x * 64
        out = self.feature_extraction(out)
        # print(out, out.shape)
        out = self.shrinking(out)
        # print(out, out.shape)
        out = self.mapping(out)
        # print(out, out.shape)
        out = self.expanding(out)
        # print(out, out.shape)
        out /= 64
        out += residual
        # print(out, out.shape)

        return out
