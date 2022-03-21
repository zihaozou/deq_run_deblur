import torch
from torch import nn
from utils.util import *
from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self, 
                       test_gdt: torch.Tensor,
                       test_y: torch.Tensor
                       ):
          
        super(TestDataset, self).__init__()
        # self.test_ipt = test_ipt
        self.test_gdt = test_gdt
        self.test_y = test_y
    def __len__(self):
        return self.test_gdt.shape[0]

    def __getitem__(self, item):
        return self.test_gdt[item], self.test_y[item]
        #self.test_ipt[item], self.test_gdt[item], self.test_y[item]

class BlurClass(nn.Module):
    def __init__(self, fwd_set, emParams):
        super(BlurClass, self).__init__()
        self.bk = emParams['bk']
        self.bkt = emParams['bkt']
    def init(self, x): 
        return self.bk.to(x.device), self.bkt.to(x.device)
    def grad(self, x,y,bk,bkt):
        with torch.no_grad():
            delta_g = self.imfilter(self.imfilter(x, bk) - y, bkt)
        return delta_g
    def fwd_bwd(self,x,bk,bkt):
        with torch.no_grad():
            g = self.imfilter(self.imfilter(x, bk), bkt)
        return g
    @staticmethod
    def imfilter(x, k):
        '''
        x: image, NxcxHxW
        k: kernel, cx1xhxw
        '''

        def dim_pad_circular(input, padding, dimension):
            # type: (Tensor, int, int) -> Tensor
            input = torch.cat([input, input[[slice(None)] * (dimension - 1) +
                                            [slice(0, padding)]]], dim=dimension - 1)
            input = torch.cat([input[[slice(None)] * (dimension - 1) +
                                     [slice(-2 * padding, -padding)]], input], dim=dimension - 1)
            return input

        def pad_circular(input, padding):
            # type: (Tensor, List[int]) -> Tensor
            """
            Arguments
            :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
            :param padding: (tuple): m-elem tuple where m is the degree of convolution
            Returns
            :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                             H + 2 * padding[1]], W + 2 * padding[2]))`
            """
            offset = 3
            for dimension in range(input.dim() - offset + 1):
                input = dim_pad_circular(input, padding[dimension], dimension + offset)
            return input

        k = torch.stack([k, k, k], dim=1).squeeze(2).permute(1, 0, 2, 3).type(
            torch.cuda.FloatTensor)
        x = pad_circular(x, padding=((k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2))
        x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
        k = k.cpu()
        torch.cuda.empty_cache()

        return x.detach()