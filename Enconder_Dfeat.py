import torch
import torch.nn as nn

def create(deep_lab_v3p, discriminator_type):
    class InvBottleneck(nn.ModuleList):
        def __init__(self, prev_filters, t, c, n, s, initial_dilation=1, dilation=1):
            super().__init__()
            for sub_index in range(n):
                _c0 = prev_filters if sub_index == 0 else c
                _c1 = t * _c0
                _s = s if sub_index == 0 else 1
                _d = initial_dilation if sub_index == 0 else dilation
                self.append(nn.Sequential(
                    nn.Conv2d(_c0, _c1, 1),
                    nn.BatchNorm2d(_c1),
                    nn.ReLU6(),
                    nn.ReplicationPad2d(_d),
                    nn.Conv2d(_c1, _c1, 3, stride=_s, dilation=_d, groups=_c1),
                    nn.BatchNorm2d(_c1),
                    nn.ReLU6(),
                    nn.Conv2d(_c1, c, 1),
                    nn.BatchNorm2d(c)
                ))
        
        def forward(self, x):
            for sub_index, layer in enumerate(self):
                x = layer(x) if sub_index == 0 else layer(x) + x
            return x

    encoder = nn.Sequential(
        deep_lab_v3p.part1,
        deep_lab_v3p.part2,
        deep_lab_v3p.aspp
    )
        
    discriminator_num_output_classes = 1
    discriminator = []
    if discriminator_type == 3:
        discriminator.extend((
            InvBottleneck(256, 6, 64, 3, 1),
            InvBottleneck(64, 6, 16, 3, 1),
            InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
        ))
    elif discriminator_type == 2:
        discriminator.extend((
            InvBottleneck(256, 6, 64, 1, 1),
            InvBottleneck(64, 6, 16, 1, 1),
            InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
        ))
    elif discriminator_type == 1:
        discriminator.extend((
            InvBottleneck(256, 6, discriminator_num_output_classes, 1, 1),
        ))
    else:
        assert discriminator_type == 0
        discriminator.extend((
            ck(256, 256, False),
            ck(256, 256, False),
            nn.Conv2d(256, discriminator_num_output_classes, 1),
        ))
    del discriminator_num_output_classes
    discriminator = nn.Sequential(*discriminator)

    return encoder, descriminator