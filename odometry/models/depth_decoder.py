import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict



class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        # 反射填充，以边缘为对称轴进行取图像里面的值填充边界
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthBasicDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthBasicDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([256, 128, 64, 32, 16])

        # decoder
        self.convs = OrderedDict()
        for i in range(len(self.num_ch_dec)):

            # upconv_0
            num_ch_in = self.num_ch_enc[len(num_ch_enc) - 1 - i] if i == 0 else self.num_ch_dec[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i < len(num_ch_enc) - 1:
                num_ch_in += self.num_ch_enc[len(num_ch_enc) - 1 - i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # self.convs[("dispconv")] = Conv3x3(self.num_ch_dec[-1], self.num_output_channels)
        self.convs[("dispconv")] = nn.Conv2d(self.num_ch_dec[-1], self.num_output_channels, kernel_size=1, padding=0, stride=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        num_features_map = len(input_features)

        # decoder
        x = input_features[-1]
        for i in range(len(self.num_ch_dec)):

            x = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]

            if self.use_skips and i < len(self.num_ch_dec) - 1:
                x += [input_features[num_features_map - 1 - i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        # 0~1
        self.output = self.sigmoid(self.convs[("dispconv")](x))

        return self.output

if __name__ == '__main__':
    from resnet_encoder import out, encoder
    print(encoder.num_ch_enc)
    dummy_input = torch.zeros(1,6,256,256)
    decoder = DepthBasicDecoder([64, 64, 128, 256, 512])
    print(decoder)
    print(decoder.convs)
    out1 = decoder(out)
    print(out1)


class UpsamleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio = 2):
        super(UpsamleBlock, self).__init__()

        self.ratio = ratio
        ratio_square = ratio**2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask = nn.Conv2d(out_channels, ratio_square * 9, 1, padding=0)

    def _upsample(self, x, up_mask):
        """ Upsample flow field [H/2, W/2, C] -> [H, W, C] using convex combination """
        N, C, H, W = x.shape
        up_mask = up_mask.view(N, 1, 9, self.ratio, self.ratio, H, W)
        up_mask = torch.softmax(up_mask, dim=2)

        up_x = F.unfold(x, [3, 3], padding=1)
        up_x = up_x.view(N, C, 9, 1, 1, H, W)

        up_x = torch.sum(up_mask * up_x, dim=2)
        up_x = up_x.permute(0, 1, 4, 2, 5, 3)
        return up_x.reshape(N, C, self.ratio * H, self.ratio * W)

    def forward(self, x):
        x = self.stem(x)
        up_mask = self.mask(x)
        up_x = self._upsample(x, up_mask)

        return up_x


if __name__ == '__main__':
    a = UpsamleBlock(512, 128)

    b = torch.rand(10,512,4,4)
    c = a(b)
    print(c.shape)


class DepthUpsampleDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthUpsampleDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([256, 128, 64, 32, 16])

        # decoder
        self.convs = OrderedDict()
        for i in range(len(self.num_ch_dec)):

            # upconv_0
            num_ch_in = self.num_ch_enc[len(num_ch_enc) - 1 - i] if i == 0 else self.num_ch_dec[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = UpsamleBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i < len(num_ch_enc) - 1:
                num_ch_in += self.num_ch_enc[len(num_ch_enc) - 1 - i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs[("dispconv")] = Conv3x3(self.num_ch_dec[-1], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        num_features_map = len(input_features)

        # decoder
        x = input_features[-1]
        for i in range(len(self.num_ch_dec)):

            x = [self.convs[("upconv", i, 0)](x)]

            if self.use_skips and i < len(self.num_ch_dec) - 1:
                x += [input_features[num_features_map - 1 - i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        # 0~1
        self.output = self.sigmoid(self.convs[("dispconv")](x))

        return self.output


class DepthNIGDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthNIGDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([256, 128, 64, 32, 16])

        # decoder
        self.convs = OrderedDict()
        for i in range(len(self.num_ch_dec)):

            # upconv_0
            num_ch_in = self.num_ch_enc[len(num_ch_enc) - 1 - i] if i == 0 else self.num_ch_dec[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i < len(num_ch_enc) - 1:
                num_ch_in += self.num_ch_enc[len(num_ch_enc) - 1 - i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs[("dispconv")] = Conv3x3(self.num_ch_dec[-1], self.num_output_channels * 4)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        # self.sigmoid = nn.Sigmoid()

        self.evidence = nn.Softplus()

    def forward(self, input_features):

        num_features_map = len(input_features)

        # decoder
        x = input_features[-1]
        for i in range(len(self.num_ch_dec)):

            x = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]

            if self.use_skips and i < len(self.num_ch_dec) - 1:
                x += [input_features[num_features_map - 1 - i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        x = self.convs[("dispconv")](x)

        out = {}

        out['mu'], logv, logalpha, logbeta = torch.split(x, split_size_or_sections=1, dim=1)
        out['v'] = self.evidence(logv)
        out['alpha'] = self.evidence(logalpha) + 1
        out['beta'] = self.evidence(logbeta)

        # self.output = torch.concat([mu, v, alpha, beta], dim=1)

        return out