from __future__ import print_function

import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models.dncnn import DnCNN
from models.jacobinNet import jacobinNet
from models.unet import UNetRes
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

def net_model(config: dict):

    if config['network'] == 'skip':
        net = skip( 
                   num_channels_down = [64] * 3,
                   num_channels_up   = [64] * 3,
                   num_channels_skip = [0] * 3,  
                   upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                   need_sigmoid=True, need_bias=True, pad=config['padding'], act_fun=config['act_fun']).type(dtype)
    elif config['network'] == 'UNet':
        net = UNet(in_channels=config['image_channels'], 
                   out_channels=config['image_channels'], 
                   init_features=config['n_channels'],
                   act_fun=config['act_fun'],
                   norm_fun=config['norm_fun'])
        # net =  jacobinNet(net)
    elif config['network'] == 'ResNet':
        net = ResNet(need_sigmoid=True, act_fun='LeakyReLU')
    elif config['network'] == 'DnCNN':
        net = DnCNN(in_channels=config['image_channels'], 
                   out_channels=config['image_channels'], 
                   init_features=64)   
        net =  jacobinNet(net)           
    elif config['network'] == 'UNetRes':
        net = UNetRes()
        net =  jacobinNet(net)
    else:
        assert False
    return net