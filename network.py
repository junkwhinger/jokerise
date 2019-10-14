import os

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """ResBlock for Pix2Pix Generator"""

    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()

        # ReflectionPad2d(1)
        # Conv(dim, dim, 3, 0)
        # InstanceNorm2d
        # ReLU
        # ReflectionPad2d(1)
        # Conv(dim, dim, 3, 0)
        # InstanceNorm2d

        block = [nn.ReflectionPad2d(1),
                 nn.Conv2d(in_dim, out_dim, 3),
                 nn.InstanceNorm2d(out_dim),
                 nn.ReLU(True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(in_dim, out_dim, 3)]

        self.model = nn.Sequential(*block)

    def forward(self, x):
        # skip connection
        x = x + self.model(x)
        return x


class Generator(nn.Module):
    """Pix2Pix Generator"""

    def __init__(self, input_ch, output_ch, ngf, n_blocks):
        super(Generator, self).__init__()

        # part 1. in_conv
        in_conv = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_ch, ngf, 7, 1, 0),
                   nn.InstanceNorm2d(ngf),
                   nn.ReLU(True)]
        self.in_conv = nn.Sequential(*in_conv)

        # part 2. down_1
        down_1 = [nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True)]
        self.down_1 = nn.Sequential(*down_1)

        # part 3. down_2
        down_2 = [nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                  nn.InstanceNorm2d(ngf * 4),
                  nn.ReLU(True)]
        self.down_2 = nn.Sequential(*down_2)

        # part 4. res_blocks
        res_blocks = []
        for i in range(n_blocks):
            res_blocks += [ResBlock(ngf * 4, ngf * 4)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # part 5. up_1
        up_1 = [nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(True)]
        self.up_1 = nn.Sequential(*up_1)

        # part 6. up_2
        up_2 = [nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(True)]
        self.up_2 = nn.Sequential(*up_2)

        # part 7. out_conv
        out_conv = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_ch, 7),
                    nn.Tanh()]
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.res_blocks(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.out_conv(x)
        return x
