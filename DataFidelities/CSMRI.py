import torch
import math
import sys
import decimal
from utils.util import *
from utils.mask_function import *

class MRIClass(nn.Module):
    def __init__(self, config, seed, emParams=None):
        super(MRIClass,self).__init__()

    def grad(self, x, y, mask):
        with torch.no_grad():
            g = self.ftran(self.fmult(x, mask) - y, mask)
        return g

    def fwd_bwd(self, x, mask): #ATAx
        with torch.no_grad():
            g = MRIClass.ftran(MRIClass.fmult(x, mask), mask)
        return g

    @staticmethod
    def fmult(x, mask):
        z = torch.stack([x, torch.zeros_like(x)], -1) # x -> z: tensor, dim: BS， C, H, W, 2 (real + imag) a + bj
        fftz = torch.fft(z, 2)
        y = fftz * mask.to(x.device)
        return y # fourier domain, 

    @staticmethod
    def ftran(y, mask):
        z_hat = torch.ifft(y * mask.to(x.device), 2)
        x = z_hat[:, :, :, 0:1]
        return x

    @staticmethod
    def FFT_Mask_ForBack(x, mask):
        # x: tensor, dim: BS, C, H, W / mask: tensot, dim: 256, 256, 2
        with torch.no_grad():
            z = torch.stack([x, torch.zeros_like(x)], -1) # x -> z: tensor, dim: BS， C, H, W, 2 (real + imag) a + bj
            fftz = torch.fft(z, 2) #(Ax)
            z_hat = torch.ifft(fftz * mask.to(x.device), 2) #(ATAx)
            x = z_hat[:, :, :, 0:1] # z_hat: tensor, dim : BS, C, H, W, 2 -> x: tensor, dim: BS, C, H, W 2
        return x #x: tensor, dim: BS, C, H, W



