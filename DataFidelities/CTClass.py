'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Mingyang Xie, CIG, WUSTL, 2020
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.util import *
from DataFidelities.pytorch_radon.filters import HannFilter
from DataFidelities.pytorch_radon import Radon, IRadon

class TrainDataset(Dataset):

    def __init__(self, train_ipt:torch.Tensor, 
                       train_gdt:torch.Tensor, 
                       train_y:torch.Tensor,
                ):
          
        super(TrainDataset, self).__init__()
        self.train_ipt = train_ipt
        self.train_gdt = train_gdt
        self.train_y = train_y

    def __len__(self):
        return self.train_gdt.shape[0]

    def __getitem__(self, item):
        return self.train_ipt[item], self.train_gdt[item], self.train_y[item]

class ValidDataset(Dataset):

    def __init__(self, valid_ipt:torch.Tensor, 
                       valid_gdt:torch.Tensor, 
                       valid_y:torch.Tensor,
                       ):
          
        super(ValidDataset, self).__init__()
        self.valid_ipt = valid_ipt
        self.valid_gdt = valid_gdt
        self.valid_y = valid_y
    def __len__(self):
        return self.valid_gdt.shape[0]

    def __getitem__(self, item):
        return self.valid_ipt[item], self.valid_gdt[item], self.valid_y[item]

class TestDataset(Dataset):

    def __init__(self, test_ipt:torch.Tensor, 
                       test_gdt:torch.Tensor,
                       test_y:torch.Tensor,
                       ):
          
        super(Dataset, self).__init__()
        self.test_ipt = test_ipt
        self.test_gdt = test_gdt
        self.test_y = test_y
    def __len__(self):
        return self.test_gdt.shape[0]

    def __getitem__(self, item):
        return self.test_ipt[item], self.test_gdt[item], self.test_y[item]

###################################################
###              Tomography Class               ###
###################################################

class CTClass(nn.Module):

    def __init__(self, sigSize=512, numAngles=45):
        super(CTClass, self).__init__()              

        self.sigSize = sigSize
        # generate angle array
        self.theta = torch.tensor(np.linspace(0., 180, numAngles, endpoint=False),dtype=torch.float)
               
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def grad(self, x, y):
    
        with torch.no_grad():
            gradList = []
            theta_ = self.theta.to(x.device)
            for i in range(x.shape[0]):
                gradList.append(self.ftran(self.fmult(x[i,:].unsqueeze(0), theta_) - y[i,:].unsqueeze(0), theta_))
            g = torch.cat(gradList, 0)

        return g

    def fwd_bwd(self, x):

        gradList = []
        theta_ = self.theta.to(x.device)
        for i in range(x.shape[0]):
            gradList.append(self.ftran(self.fmult(x[i,:].unsqueeze(0), theta_), theta_))
        g = torch.cat(gradList, 0)

        return g

    def gradStoc(self, x, y, meas_list):
        pass

    def fmult(self, x, theta=None):

        device = x.device
        r = Radon(self.sigSize, theta, False, dtype=torch.float, device=device)
        sino = r(x)

        return sino
    
    def ftran(self, z, theta=None):

        device = z.device
        ir = IRadon(self.sigSize, theta, False, dtype=torch.float, use_filter=None, device=device)
        reco_torch = ir(z)

        return reco_torch

    @staticmethod
    def tomoCT(ipt, sigSize, device=None, batchAngles=None, numAngles=180, inputSNR=40):

        if batchAngles is not None:
            print('In the tomoCT function of TorchClass, the batchAngles parameteris currently unsupported.')
            exit()
        device = ipt.device
        # generate angle array
        theta = np.linspace(0., 180, numAngles, endpoint=False)
               
        # convert to torch
        theta = torch.tensor(theta, dtype=torch.float, device=device)
        ipt = ipt.unsqueeze(0).unsqueeze(0).to(device)
        
        # forward project
        r = Radon(sigSize, theta, False, device=device)
        sino = r(ipt.to(device))

        # add white noise to the sinogram
        sino = addwgn_torch(sino, inputSNR)[0]

        # backward project
        ir = IRadon(sigSize, theta, False, use_filter=None, device=device)
        recon_bp = ir(sino)
        # filtered backward project
        ir_hann = IRadon(sigSize, theta, False, dtype=torch.float, use_filter=HannFilter(), device=device)
        reco_fbp_hann = ir_hann(sino)
        reco_fbp_hann[reco_fbp_hann<=0] = 0
        
        return sino, recon_bp, reco_fbp_hann, theta






