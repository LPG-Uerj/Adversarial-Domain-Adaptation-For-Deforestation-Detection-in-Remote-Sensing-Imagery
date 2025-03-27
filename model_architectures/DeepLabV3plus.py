import torch
import torch.nn as nn

def create(args):
    patch_size = args.patch_size
    num_classes = args.num_classes
    channels = args.channels

    large_latent_space = args.large_latent_space
    dilation_rates = args.dilation_rates

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
        
    MobileNetv2_part1 = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(channels, 32, 3, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU6(),
        InvBottleneck(32, 1, 16, 1, 1),
        InvBottleneck(16, 6, 24, 2, 2)
    )

    if large_latent_space:
        MobileNetv2_part2 = nn.Sequential(
            InvBottleneck(24, 6, 32, 3, 2),
            InvBottleneck(32, 6, 64, 4, 1, dilation=2),
            InvBottleneck(64, 6, 96, 3, 1, initial_dilation=2, dilation=2),
            InvBottleneck(96, 6, 160, 3, 1, initial_dilation=2, dilation=4),
            InvBottleneck(160, 6, 320, 1, 1, initial_dilation=4)
        )
    else:
        MobileNetv2_part2 = nn.Sequential(
            InvBottleneck(24, 6, 32, 3, 2),
            InvBottleneck(32, 6, 64, 4, 2),
            InvBottleneck(64, 6, 96, 3, 1),
            InvBottleneck(96, 6, 160, 3, 1, dilation=2),
            InvBottleneck(160, 6, 320, 1, 1, initial_dilation=2)
        )
        
    class AtrousSpatialPyramidPooling(nn.ModuleList):
        def __init__(self):
            super().__init__()
            latent_space_size = patch_size // (8 if large_latent_space else 16)
            # global average pooling
            self.append(nn.Sequential(
                nn.AvgPool2d(latent_space_size),
                nn.Conv2d(320, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6(),
                nn.Upsample(latent_space_size)
            ))
            # 1x1 conv
            self.append(nn.Sequential(
                nn.Conv2d(320, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6()
            ))
            # atrous conv
            for d in dilation_rates:
                self.append(nn.Sequential(
                    nn.ReplicationPad2d(d),
                    nn.Conv2d(320, 320, 3, dilation=d, groups=320),
                    nn.Conv2d(320, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU6()
                ))
            
        def forward(self, x):
            results = []
            for layer in self:
                results.append(layer(x))
            return torch.cat(results, 1)
        
    class DeepLabv3p(nn.Module):
        def __init__(self):
            super().__init__()
            self.part1 = MobileNetv2_part1
            self.part2 = MobileNetv2_part2
            self.aspp = nn.Sequential(
                AtrousSpatialPyramidPooling(),
                nn.Conv2d(256*(2+len(dilation_rates)), 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6()
            )
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2 if large_latent_space else 4, mode="bilinear")
            )
            self.skip_connection = nn.Sequential(
                nn.Conv2d(24, 48, 1),
                nn.BatchNorm2d(48),
                nn.ReLU6()
            )
            self.final = nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(304, 304, 3, groups=304),
                nn.Conv2d(304, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6(),
                nn.ReplicationPad2d(1),
                nn.Conv2d(256, 256, 3, groups=256),
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6(),
                nn.Conv2d(256, num_classes, 1),
                nn.Upsample(scale_factor=4, mode="bilinear"),
                # nn.Softmax()
            )
            
        def forward(self, x):
            x = self.part1(x)
            x = torch.cat((self.upsample(self.aspp(self.part2(x))), self.skip_connection(x)), 1)
            return self.final(x)
    return DeepLabv3p()