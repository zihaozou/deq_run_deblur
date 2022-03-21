from cv2 import NORMCONV_FILTER
import torch
from torch import nn

import h5py
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime
from skimage.util import view_as_windows

from utils.util import *
from DataFidelities.MRIClass import *
from utils.mask_function import RandomMaskFunc
#%%
np.random.seed(128)

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def gen_x_y(config, img, emParams, device, set_mode:str):
    IMG_PATCH = config['IMG_Patch']
    imgList,snr_avg = [],0
    for index in range(img.shape[0]):
        img[index] = img[index] - np.amin(img[index])
        img[index] = img[index] / np.amax(img[index])
        img[index] = img[index]
    gdt = torch.from_numpy(img).to(device)  # (72, 1, 90, 90)
    ipt = torch.zeros(size=gdt.shape, dtype=torch.float32)  # (72, 1, 90, 90, 2)
    y   = torch.zeros([gdt.shape[0], 1, IMG_PATCH[0], IMG_PATCH[1], 2], dtype=torch.float32) # (72, 60, 1, 90, 90, 2)
    
    for index in tqdm(range(gdt.shape[0]), 'fBrain MRI %s Dataset'%(set_mode)):

        img_temp = gdt[index].unsqueeze(0) # (1, 90, 90)
        ipt_each, y_each = MRIClass.FFT_Mask_ForBack(img_temp,  emParams['mask'])
        y[index]   = y_each.detach().cpu()
        ipt[index] = ipt_each.detach().cpu()
        snr_avg =  snr_avg + compare_snr(ipt_each.squeeze(), img_temp.squeeze()).item()

    print("%s Dataset: "%(set_mode))
    print('gdt: ', gdt.shape, gdt.dtype)
    print('Avg_inputSNR: ',  snr_avg/gdt.shape[0])

    return gdt, y, ipt

def set2use(data_root, numSet=1, phase='train'):
    
    imgList = []

    if phase == 'train':

        if numSet == 0:
            setName = []
        elif numSet == 1:
            setName = []
        elif numSet == 2:
            setName = []
        elif numSet == 3:
            setName = ['fBrain_dataset_h5_T2']

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList
    elif phase == 'valid':

        if numSet == 0:
            setName = []  
        elif numSet == 1:    
            setName = ['BrainImages_test'] 

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList    

    return imgList

def get_images(datapath, IMG_PATCH, numSet=1, phase='train', tp='file'):
    dtype = torch.FloatTensor #Choose tensor datatype, defult : gpu.float32
    nH, nW = IMG_PATCH
    ###############################################
    #                 Start Loading               #
    ###############################################

    if tp == 'file':
        Training_data = sio.loadmat(datapath)
        img_raw = Training_data['labels']
        BS,H,W = img_raw.shape
        img_raw = np.expand_dims(img_raw, 1)
    elif tp == 'imgs':

        imgList = set2use(datapath, numSet, phase=phase)
        num_scales = [1]
        img_raw = []

        for index in tqdm(imgList):

            for scaling in num_scales:
                
                for m in range(1):
                    img = scaling * uint2single(imread_uint(index, n_channels=1)).squeeze()
                    img_raw.append(img)

        img_raw = np.expand_dims(np.array(img_raw), axis=1)                                                                                                
        print('img_gdt: ', img_raw.shape, img_raw.dtype)

    return img_raw

def get_matrix(cs_ratio):
    mask = sio.loadmat('/export1/project/Jiaming/potential_SDEQ_MRI/data/masks/sampling_matrix/mask_%d.mat'%(cs_ratio), squeeze_me=True)['mask_matrix']
    mask = torch.from_numpy(np.stack([mask, mask], axis=2))
    print(mask.shape)
    return mask

def data_preprocess(config:dict, device=None, set_mode='test'):

    ###############################################
    #                 Start Loading               #
    ###############################################
    # Train
    train_set, valid_set, test_set = {},{},{}
    ###############################################
    #                 Start Loading               #
    ###############################################
    if set_mode=='test':
        pass
    elif set_mode=='train':
        #train
        IMG_PATCH, cs_ratio = config['fwd_train'].IMG_Patch, config['fwd_train'].cs_ratio

        mask = get_matrix(cs_ratio)
        
        emParams = {
            'mask': mask.to(device),# torch.Size([1, 1, 256, 1])
        }
        
        config['train_datapath'] = os.path.join(config['root_path'], config['train_datapath'])

        img_raw = get_images(config['train_datapath'], IMG_PATCH=config['fwd_train'].IMG_Patch, phase='train', tp='file')
        
        num_uesd = int(config['training'].num_train * img_raw.shape[0])
        num_uesd = (num_uesd//config['training'].batch_size)*config['training'].batch_size
        num_uesd = np.random.choice(img_raw.shape[0], num_uesd, replace=False).tolist()
        num_uesd.sort()
        print(num_uesd)
        img_raw = img_raw[num_uesd,:].astype(np.float32)#[8,254,254]
        fwd_train = config['fwd_train']
        
        gdt, y, ipt = gen_x_y(fwd_train, img_raw, emParams, device, 'train')

        train_set = {
                    "train_ipt": ipt,
                    "train_y": y,
                    "train_gdt": gdt,
                    'num_uesd':num_uesd
                    }

        #valid
        IMG_PATCH, cs_ratio = config['fwd_valid'].IMG_Patch, config['fwd_valid'].cs_ratio
        config['valid_datapath'] = os.path.join(config['root_path'], config['valid_datapath'])

        img_raw = get_images(config['valid_datapath'], IMG_PATCH=config['fwd_valid'].IMG_Patch, phase='valid', tp='imgs')

        num_uesd = int(config['validating'].num_valid * img_raw.shape[0])
        num_uesd = (num_uesd//config['validating'].batch_size)*config['validating'].batch_size
        num_uesd = np.random.choice(img_raw.shape[0], num_uesd, replace=False).tolist()
        num_uesd.sort()
        print(num_uesd)
        img_raw = img_raw[num_uesd,:].astype(np.float32)#[8,254,254]
        fwd_valid = config['fwd_valid']
        gdt, y, ipt = gen_x_y(fwd_valid, img_raw, emParams, device, 'valid')
        emParams = {
            'mask': mask.detach().cpu(),# torch.Size([60, 1, 90, 90, 2])
        }
        valid_set = {
                    "valid_ipt": ipt,
                    "valid_y": y,
                    "valid_gdt": gdt,
                    "emParams": emParams,
                    'num_uesd':num_uesd
                    }
        train_set['emParams'] = emParams
    
    return train_set, valid_set, test_set