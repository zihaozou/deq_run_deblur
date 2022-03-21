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
from DataFidelities.BlurClass import *
#%%
np.random.seed(128)

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def gen_x_y(config, img, emParams, device, set_mode:str):
    IMG_PATCH, sigma = config['IMG_Patch'], config['simga']
    imgList,snr_avg = [],0
    for index in range(img.shape[0]):
        img[index] = img[index] - np.amin(img[index])
        img[index] = img[index] / (np.amax(img[index])+1e-9)
        img[index] = img[index]

    gdt = torch.from_numpy(img) # (72, 1, 90, 90)
    y   = torch.zeros_like(gdt, dtype=torch.float32) # (72, 60, 1, 90, 90, 2)
    
    for index in tqdm(range(gdt.shape[0]), 'Color imags %s Dataset'%(set_mode)):

        img_temp = gdt[index].unsqueeze(0).to(device)  # (1, 90, 90)
        imgL = BlurClass.imfilter(img_temp,  emParams['bk'])
        if sigma !=0:
            imgL += torch.FloatTensor(imgL.size()).normal_(mean=0, std=sigma/255.).to(imgL.device)
        y[index]   = imgL.detach().cpu()
        snr_avg =  snr_avg + compare_snr(imgL.squeeze(), img_temp.squeeze()).item()

    print("%s Dataset: "%(set_mode))
    print('gdt: ', gdt.shape, gdt.dtype)
    print('Avg_inputSNR: ',  snr_avg/gdt.shape[0])

    return gdt.detach().cpu(), y.detach().cpu()

def set2use(data_root, numSet=1, phase='train'):
    
    imgList = []

    if phase == 'train':

        if numSet == 0:
            setName = ['CBSD400']
        elif numSet == 1:
            setName = ['CBSD400', 'DIV2K_B']
        elif numSet == 2:
            setName = ['CBSD400', 'DIV2K_A', 'DIV2K_B']
        elif numSet == 3:
            setName = ['CBSD400', 'DIV2K_A', 'DIV2K_B', 'Flickr2K']

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList
    elif phase == 'valid':

        if numSet == 0:
            setName = ['set5']   
        elif numSet == 1:    
            setName = ['set3c'] 

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList    
    elif phase == 'test':

        if numSet == 0:
            setName = ['set5']  
        elif numSet == 1:    
            setName = ['set3c'] 

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList   
    return imgList

def get_images(datapath, IMG_PATCH, numSet=1, phase='train', tp='file'):
    dtype = torch.FloatTensor #Choose tensor datatype, defult : gpu.float32
    nH, nW = IMG_PATCH
    if IMG_PATCH[0] != IMG_PATCH[1]:
        print('Training Requires Square imags !!!')
        exit()
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
                    img = scaling * uint2single(imread_uint(index, n_channels=3)).squeeze()
                    patches = patches_from_image(img, p_size=IMG_PATCH[0], p_overlap=0, p_max=250)
                    patches = patches.transpose(0,3,1,2)
                    img_raw.append(patches)

        img_raw = np.concatenate(img_raw,0)
        print('img_gdt: ', img_raw.shape, img_raw.dtype)

    return img_raw

def get_matrix(config):

    IMG_PATCH, kernel_tp = config['fwd_test'].IMG_Patch, config['fwd_test'].kernel_tp

    # Load kernels
    # kernels = sio.loadmat(config['kernal_datapath'])['kernels'][0]
    # blur_kernel = kernels[kernel_tp].astype(np.float64)
    kernels = sio.loadmat(config['kernal_datapath'], squeeze_me=True)[kernel_tp]
    blur_kernel = kernels.astype(np.float64)
    blur_kernel_trans = blur_kernel[::-1, ::-1]

    # Convert from numpy to torch:
    bk = torch.from_numpy(blur_kernel.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    bkt = torch.from_numpy(blur_kernel_trans.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    return bk, bkt

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
        
        bk, bkt = get_matrix(config)
        
        emParams = {
            'bk': bk.to(device),# torch.Size([1, 1, 256, 1])
            'bkt': bkt.to(device),# torch.Size([1, 1, 256, 1])
        }

        config['test_datapath'] = os.path.join(config['root_path'], config['test_datapath'])

        img_raw = get_images(config['test_datapath'], IMG_PATCH=config['fwd_test'].IMG_Patch, phase='test', tp='imgs')
        img_raw = img_raw.astype(np.float32)#[8,254,254]
        fwd_test = config['fwd_test']
        gdt, y = gen_x_y(fwd_test, img_raw, emParams, device, 'test')

        test_set = {
                    "test_y": y,
                    "test_gdt": gdt,
                    "emParams": emParams,
                    }
        test_set['emParams'] = emParams
    elif set_mode=='train':
        
        bk, bkt = get_matrix(config)
        
        emParams = {
            'bk': bk.to(device),# torch.Size([1, 1, 256, 1])
            'bkt': bkt.to(device),# torch.Size([1, 1, 256, 1])
        }

        config['train_datapath'] = os.path.join(config['root_path'], config['train_datapath'])
        img_raw = get_images(config['train_datapath'], IMG_PATCH=config['fwd_train'].IMG_Patch, numSet=2, phase='train', tp='imgs')


        num_uesd = int(config['training'].num_train * img_raw.shape[0])
        num_uesd = (num_uesd//config['training'].batch_size)*config['training'].batch_size
        num_uesd = np.random.choice(img_raw.shape[0], num_uesd, replace=False).tolist()
        num_uesd.sort()
        print(num_uesd)
        img_raw = img_raw[num_uesd,:].astype(np.float32)#[8,254,254]
        fwd_train = config['fwd_train']
        gdt, y = gen_x_y(fwd_train, img_raw, emParams, device, 'train')

        train_set = {
                    "train_y": y,
                    "train_gdt": gdt,
                    "emParams": emParams,
                    }
        train_set['emParams'] = emParams


        ######################################
        bk, bkt = get_matrix(config)
        
        emParams = {
            'bk': bk.to(device),# torch.Size([1, 1, 256, 1])
            'bkt': bkt.to(device),# torch.Size([1, 1, 256, 1])
        }

        config['valid_datapath'] = os.path.join(config['root_path'], config['valid_datapath'])

        img_raw = get_images(config['valid_datapath'], IMG_PATCH=config['fwd_valid'].IMG_Patch, phase='valid', tp='imgs')
        img_raw = img_raw.astype(np.float32)#[8,254,254]
        fwd_valid = config['fwd_valid']
        gdt, y = gen_x_y(fwd_valid, img_raw, emParams, device, 'valid')

        valid_set = {
                    "valid_y": y,
                    "valid_gdt": gdt,
                    "emParams": emParams,
                    }
        valid_set['emParams'] = emParams        

    return train_set, valid_set, test_set