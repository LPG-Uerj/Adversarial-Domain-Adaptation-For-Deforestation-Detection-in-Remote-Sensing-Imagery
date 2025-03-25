import torch
import torch.nn as nn
import numpy as np
import random


class ck4(nn.Module):
    def __init__(self, i, k, use_normalization):
        super(ck4, self).__init__()
        self.conv_block = self.build_conv_block(i, k, use_normalization)

    def build_conv_block(self, i, k, use_normalization):
        conv_block = []                       
        conv_block += [nn.Conv2d(i, k, kernel_size=4, stride=1, padding=1, output_padding=1)]
        if use_normalization:
            conv_block += [nn.InstanceNorm2d(k, affine=True)]
        conv_block += [nn.LeakyReLU(0.2)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class ck(nn.Module):
    def __init__(self, i, k, s, use_normalization):
        super(ck, self).__init__()
        self.conv_block = self.build_conv_block(i, k, s, use_normalization)

    def build_conv_block(self, i, k, s, use_normalization):
        conv_block = []                       
        conv_block += [nn.Conv2d(i, k, kernel_size=4, stride=s, padding=1)]
        if use_normalization:
            conv_block += [nn.InstanceNorm2d(k, affine=True)]
        conv_block += [nn.LeakyReLU(0.2)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class c7Ak(nn.Module):
    def __init__(self, i, k):
        super(c7Ak, self).__init__()
        self.conv_block = self.build_conv_block(i, k)

    def build_conv_block(self, i, k):
        conv_block = []
        conv_block += [nn.Conv2d(i, k, kernel_size=7)]
        conv_block += [nn.InstanceNorm2d(k, affine=True)]
        conv_block += [nn.ReLU6()]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out
   
class dk(nn.Module):
    def __init__(self, i, k):
        super(dk, self).__init__()
        self.conv_block = self.build_conv_block(i, k)

    def build_conv_block(self, i, k):
        conv_block = []                       
        conv_block += [nn.Conv2d(i, k, kernel_size=3, stride=2, padding=1)]
        conv_block += [nn.InstanceNorm2d(k, affine=True)]
        conv_block += [nn.ReLU6()]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class Rk(nn.Module):
    def __init__(self, i):
        super(Rk, self).__init__()
        self.conv_block = self.build_conv_block(i)

    def build_conv_block(self, i):
        conv_block = []
        dim = i
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim, affine=True),
                       nn.ReLU6()]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim, affine=True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class uk(nn.Module):
    def __init__(self, i, k):
        super(uk, self).__init__()
        self.conv_block = self.build_conv_block(i, k)
        
    def build_conv_block(self, i, k):
        conv_block = []                       
        conv_block += [nn.ConvTranspose2d(i, k, kernel_size=3, stride=2, padding=1, output_padding=1)]
        conv_block += [nn.InstanceNorm2d(k, affine=True)]
        conv_block += [nn.ReLU6()]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        x = self.conv_block(x)
        return x



class modelDiscriminator(nn.Module):
    def __init__(self, name=None, channels = 14):
        super().__init__()
        self.name = name
        self.disc = nn.Sequential(
            ck(channels, 64, 2, False),
            ck(64, 128, 2, True),
            ck(128, 256, 2, True),
            ck(256, 512, 1, True),
            #nn.Conv2d(512, 1, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            #nn.Sigmoid() #decoment for BCE
        )
        
    def forward(self, x):
        #x = x*2 - 1 #decoment for BCE
        x = self.disc(x)
        return x

    
class modelGenerator(nn.Module):
    def __init__(self, name=None, channels = 14):
        super().__init__()
        self.name = name
        self.gen = nn.Sequential(
            nn.ReflectionPad2d(3),
            c7Ak(channels, 32),
            dk(32, 64),
            dk(64, 128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            Rk(128),
            uk(128, 64),
            uk(64, 32),
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, channels, kernel_size=7, stride=1),
            # nn.ReLU6()
            # mog:
            # trocar por linear
        )
        
    def forward(self, x):
        x = self.gen(x)
        return x


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        images = images.cpu().detach().numpy()
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))
        return_images = torch.from_numpy(return_images).float().requires_grad_()
        return return_images