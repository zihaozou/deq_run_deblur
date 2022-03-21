import torch
import numpy as np
import scipy.io as sio
from utils.util import *
from utils.mask_function import *

class MRIClass(nn.Module):
    def __init__(self, config, seed, emParams=None):
        super(MRIClass,self).__init__()
        self.mask = emParams['mask']

    def init(self, x): 
        mask = self.mask.to(x.device)
        return mask

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
        y = fftz * mask
        return y # fourier domain, 

    @staticmethod
    def ftran(y,mask):
        z_hat = torch.ifft(y*mask, 2)
        x = z_hat[:, :, :, :, 0]
        return x

    @staticmethod
    def FFT_Mask_ForBack(x, mask):
        # x: tensor, dim: BS, C, H, W / mask: tensot, dim: 256, 256, 2
        with torch.no_grad():
            z = torch.stack([x, torch.zeros_like(x)], -1) # x -> z: tensor, dim: BS， C, H, W, 2 (real + imag) a + bj
            fftz = torch.fft(z, 2) #(Ax)
            y = fftz * mask.to(x.device)
            z_hat = torch.ifft(y, 2) #(ATAx)
            x = z_hat[:, :, :, :, 0] # z_hat: tensor, dim : BS, C, H, W, 2 -> x: tensor, dim: BS, C, H, W 2
        return x, y #x: tensor, dim: BS, C, H, W


