from collections import OrderedDict

import torch
import torch.nn as nn

from .common import *

import torch.nn.utils.spectral_norm as spectral_norm
# class UNet(nn.Module):


#     def __init__(self, in_channels=1, out_channels=1, 
#             init_features=32, act_fun="PReLU", norm_fun="GroupNorm"):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, act_fun, norm_fun, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNet._block(features, features * 2, act_fun, norm_fun, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNet._block(features * 2, features * 4, act_fun, norm_fun, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNet._block(features * 4, features * 8, act_fun, norm_fun, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = UNet._block(features * 8, features * 16, act_fun, norm_fun, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNet._block((features * 8) * 2, features * 8, act_fun, norm_fun, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNet._block((features * 4) * 2, features * 4, act_fun, norm_fun, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNet._block((features * 2) * 2, features * 2, act_fun, norm_fun, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNet._block(features * 2, features, act_fun, norm_fun, name="dec1")

#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         return self.conv(dec1)

#     @staticmethod
#     def _block(in_channels, features, act_fun, norm_fun, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm1", nn_norm(norm_fun, features)),
#                     (name + "relu1", nn_act(act_fun)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm2", nn_norm(norm_fun, features)),
#                     (name + "relu2", nn_act(act_fun)),
#                 ]
#             )
#         )

# class UNet(nn.Module):

#     def __init__(self, in_channels=1, out_channels=1, 
#                      init_features=32, act_fun="ELU", norm_fun=None):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
#         # self.drop1 = nn.Dropout2d(p=0.25)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
#         # self.drop2 = nn.Dropout2d()
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
#         # self.drop3 = nn.Dropout2d()
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
#         # # self.drop4 = nn.Dropout2d()

#         self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=(1,2,2), stride=(1,2,2)
#         )
#         self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#         # self.drop5 = nn.Dropout2d()
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=(1,2,2), stride=(1,2,2)
#         )
#         self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#         # self.drop6 = nn.Dropout2d()
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=(1,2,2), stride=(1,2,2)
#         )
#         self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#         # self.drop7 = nn.Dropout2d()
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=(1,2,2), stride=(1,2,2)
#         )
#         self.decoder1 = UNet._block(features * 2, features, name="dec1")
#         # self.drop8 = nn.Dropout2d()
#         self.conv = nn.Conv3d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         # print(enc1.shape)
#         enc2 = self.encoder2(self.pool1(enc1))
#         # print(enc2.shape)
#         enc3 = self.encoder3(self.pool2(enc2))
#         # print(enc3.shape)
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))
#         # print(bottleneck.shape)

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         # print(dec3.shape)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         # print(dec3.shape)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         out = self.conv(dec1)
#         return x - out#.mean([1,2,3,4]).sum()

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv3d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
                    
#                     (name + "relu1", nn.PRLeu()),
#                     (
#                         name + "conv2",
#                         nn.Conv3d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
                    
#                     (name + "relu2", nn.PRLeu()),
#                 ]
#             )
#         )

# class UNet(nn.Module):

#     def __init__(self, in_channels=1, out_channels=1, 
#                      init_features=32, act_fun="ELU", norm_fun=None):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.drop1 = nn.Dropout2d(p=0.25)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.drop2 = nn.Dropout2d()
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.drop3 = nn.Dropout2d()
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.drop4 = nn.Dropout2d()

#         self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#         # self.drop5 = nn.Dropout2d()
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#         # self.drop6 = nn.Dropout2d()
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#         # self.drop7 = nn.Dropout2d()
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNet._block(features * 2, features, name="dec1")
#         # self.drop8 = nn.Dropout2d()
#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )

#     def forward(self, x):

#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         out = self.conv(dec1)

#         # out = torch.pow(self.conv(dec1), 2)

#         return x - out#.mean([1,2,3]).sum()

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
                    
#                     (name + "relu1", nn.PReLU()),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
                    
#                     (name + "relu2", nn.PReLU()),
#                 ]
#             )
#         )

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, 
            init_features=32, act_fun="PReLU", norm_fun="GroupNorm"):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop1 = nn.Dropout2d(p=0.25)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop2 = nn.Dropout2d()
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop3 = nn.Dropout2d()
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop4 = nn.Dropout2d()

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = spectral_norm(nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        ))
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        # self.drop5 = nn.Dropout2d()
        self.upconv3 = spectral_norm(nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        ))
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        # self.drop6 = nn.Dropout2d()
        self.upconv2 = spectral_norm(nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        ))
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        # self.drop7 = nn.Dropout2d()
        self.upconv1 = spectral_norm(nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        ))
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        # self.drop8 = nn.Dropout2d()
        self.conv = spectral_norm(nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        ))

    def forward(self, x):
        BS, C, H, W = x.shape
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)

        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        spectral_norm(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )),
                    ),
                    (name + "norm1", nn.GroupNorm(num_groups=4, num_channels=features)),
                    (name + "relu1", nn.PReLU()),
                    (
                        name + "conv2",
                        spectral_norm(nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )),
                    ),
                    (name + "relu2", nn.PReLU()),
                ]
            )
        )

import models.basicblock as B
import numpy as np

'''
# ====================
# Residual U-Net
# ====================
citation:
@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
# ====================
'''
class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[32, 64, 128, 256], nb=2, act_mode='S', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # self.m_head = nn.Sequential(
        #     B.conv(in_nc, nc[0]//2, bias=bias, mode='C'),
        #     nn.ELU(),
        #     B.conv(nc[0]//2, nc[0], bias=bias, mode='C')
        # )

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb//2)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb//2)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb//2)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):
       #h, w = x.size()[-2:]
       #paddingBottom = int(np.ceil(h/8)*8-h)
       #paddingRight = int(np.ceil(w/8)*8-w)
       #x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        # print(x0.shape)
        x1 = self.m_head(x0)
        # print(x1.shape)
        x2 = self.m_down1(x1)
        # print(x2.shape)
        x3 = self.m_down2(x2)
        # print(x3.shape)
        x4 = self.m_down3(x3)
        # print(x4.shape)
        x = self.m_body(x4)
        # print(x.shape)
        x = self.m_up3(x+x4)
        # print(x.shape)
        x = self.m_up2(x+x3)
        # print(x.shape)
        x = self.m_up1(x+x2)
        # print(x.shape)
        x = self.m_tail(x+x1)
        # print(x.shape)
        #x = x[..., :h, :w]
        return x.mean([1,2,3]).sum()