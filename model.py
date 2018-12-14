import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """Enocder network."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5):
        super(Encoder, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, 2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        out_src = self.main(x)
        return out_src, None

class Classifier(nn.Module):
    """Enocder network."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5):
        super(Classifier, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, 2))
        self.main = nn.Sequential(*layers)
        self.conv = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        out_src = self.main(x)
        out_cls = self.conv(out_src)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class EncoderList(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, s_dim=5, c_dim=5):
        super(EncoderList, self).__init__()

        self.style_encoder = Encoder(image_size, conv_dim, s_dim)
        self.char_encoder  = Encoder(image_size, conv_dim, c_dim)

    def forward(self, x):
        style_src, style_cls = self.style_encoder(x)
        char_src,  char_cls  = self.char_encoder(x)

        return style_src, char_src, style_cls, char_cls

class ClassifierList(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, s_dim=5, c_dim=5):
        super(ClassifierList, self).__init__()

        self.style_encoder = Classifier(image_size, conv_dim, s_dim)
        self.char_encoder  = Classifier(image_size, conv_dim, c_dim)

    def forward(self, x):
        style_src, style_cls = self.style_encoder(x)
        char_src,  char_cls  = self.char_encoder(x)

        return style_src, char_src, style_cls, char_cls

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        curr_dim = conv_dim * np.power(2, 2)
        self.mixer = nn.Conv2d(2*curr_dim+2*c_dim, curr_dim, 1)
        layers = []

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, c_from, style, char, c_to):
        # Replicate spatially and concatenate domain information.
        c_from = c_from.view(c_from.size(0), c_from.size(1), 1, 1)
        c_from = c_from.repeat(1, 1, style.size(2), style.size(3))
        c_to = c_to.view(c_to.size(0), c_to.size(1), 1, 1)
        c_to = c_to.repeat(1, 1, style.size(2), style.size(3))

        z = self.mixer(torch.cat([c_from, style, char, c_to], 1))
        return self.main(z)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, s_dim=5, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, s_dim, kernel_size=kernel_size, bias=False)
        self.conv3 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_style = self.conv2(h)
        out_char  = self.conv3(h)
        return out_src.view(out_src.size(0), out_src.size(1)), \
               out_style.view(out_style.size(0), out_style.size(1)),\
               out_char.view(out_char.size(0), out_char.size(1))
