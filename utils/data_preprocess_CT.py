from skimage.util import view_as_windows

import os
import torch
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm

from utils.util import *
from DataFidelities.CTClass import CTClass
np.random.seed(128)

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))

def read_nii(data_path, csv_file, num_volum=7):
    """read spacings and image indices in CT-ORG"""
    nib.Nifti1Image
    with open('data/{}'.format(csv_file), 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        rownum = 0
        for row in reader:
            if rownum == 0:
                print('{}/volume-{}.nii.gz'.format(data_path, int(row['volume']) - 1))
                img = np.array(nib.load('{}/volume-{}.nii.gz'.format(data_path, int(row['volume']) - 1)).get_data()).astype(np.float32)
                slice_start =  int(row['slice_start'])
                slice_end =  int(row['slice_end'])
                predata = img[:,:,slice_start-1:slice_end-1]
                rownum = rownum + 1
            elif rownum<num_volum :
                print('{}/volume-{}.nii'.format(data_path, int(row['volume']) - 1))
                img = np.array(nib.load('{}/volume-{}.nii.gz'.format(data_path, int(row['volume']) - 1)).get_data()).astype(np.float32)
                slice_start =  int(row['slice_start'])
                slice_end =  int(row['slice_end'])
                predata = np.concatenate([predata, img[:,:,slice_start-1:slice_end-1]], axis=2)
                rownum = rownum + 1
    return predata

def gen_x_y(config, img_raw, device, set_mode:str):

    imgList = []
    num_scales = [1]
    dtype = torch.FloatTensor
    IMG_PATCH = config['IMG_Patch']
    for index in range(img_raw.shape[0]):

        for scaling in num_scales:
            
            for m in range(1):
                img = scaling * img_raw[index].copy()
                # img = data_augmentation(img, m)
                img = np.ascontiguousarray(img)
                img = view_as_windows(img, window_shape=[IMG_PATCH, IMG_PATCH], step=40)
                img = np.ascontiguousarray(img)
                img.shape = [-1] + [IMG_PATCH, IMG_PATCH]
                img = np.rot90(img, k=1, axes=(1,2))
                img = np.expand_dims(img, 1)  # Add channel in the dimension right after batch-dimension.
                imgList.append(img)
    img = torch.from_numpy(np.concatenate(imgList, 0)).type(dtype)
    img_fbp = torch.zeros_like(img)

    sino_temp, _, fbp_temp, _ = CTClass.tomoCT(torch.squeeze(img[0]).to(device), sigSize=IMG_PATCH, 
                                      device=device,numAngles=config['numAngles'],inputSNR=config['inputSNR'])

    y_sino = torch.zeros([img.shape[0], sino_temp.shape[1], sino_temp.shape[2],sino_temp.shape[3]], dtype=torch.float32)

    for index in tqdm(range(img.shape[0]), 'CT %s Dataset'%(set_mode)):
        img_temp = img[index]
        img_temp = (img_temp - torch.min(img_temp)) /(torch.max(img_temp) - torch.min(img_temp))
        img[index] =  img_temp

        y_sino[index], _, img_fbp[index], _ = CTClass.tomoCT(torch.squeeze(img[index]).to(device), sigSize=IMG_PATCH, 
                                        device=device,numAngles=config['numAngles'],inputSNR=config['inputSNR'])

    gdt = img.type(dtype)  # (72, 1, 90, 90)
    y = y_sino.type(dtype)
    ipt = img_fbp.type(dtype)

    print("%s Dataset: "%(set_mode))
    print('gdt: ', gdt.shape, gdt.dtype)
    print('ipt: ', ipt.shape, ipt.dtype)
    print('y: ', y.shape, y.dtype)

    print('Avg_SNR of FBP on %s dataset:  '%(set_mode), compare_snr(ipt, gdt))

    return gdt, y, ipt

def data_preprocess(config:dict, device=None, set_mode='test'):
    train_set, valid_set, test_set = {},{},{}
    ###############################################
    #                 Start Loading               #
    ###############################################
    if set_mode=='test':
        config['test_datapath'] = os.path.join(config['root_path'], config['test_datapath'])
        img_raw = read_nii(data_path=config['test_datapath'], csv_file='CT_ORG_test.csv', num_volum=config['testing'].num_volum)
        img_raw = img_raw.transpose([2,0,1]).astype(np.float32)
        num_uesd = int(config['testing'].num_test * img_raw.shape[0])
        num_uesd = num_uesd - divmod(num_uesd, config['num_gpus'])[1]
        img_raw = img_raw[0:num_uesd,:]#[8,254,254]
        fwd_test = config['fwd_test']
        gdt, y, ipt = gen_x_y(fwd_test, img_raw, device,set_mode)
        test_set = {
                    "test_ipt": ipt,
                    "test_gdt": gdt,
                    "test_y": y
                    }         
    elif set_mode=='train':
        #train
        config['train_datapath'] = os.path.join(config['root_path'], config['train_datapath'])
        img_raw = read_nii(data_path=config['train_datapath'], csv_file='CT_ORG_train.csv', num_volum=config['training'].num_volum)  
        img_raw = img_raw.transpose([2,0,1]).astype(np.float32)
        num_uesd = int(config['training'].num_train * img_raw.shape[0])
        num_uesd = num_uesd - divmod(num_uesd, config['num_gpus'])[1]
        img_raw = img_raw[0:num_uesd,:]#[8,254,254]
        fwd_train = config['fwd_train']
        gdt, y, ipt = gen_x_y(fwd_train, img_raw, device,'train')
        train_set = {
                    "train_ipt": ipt,
                    "train_gdt": gdt,
                    "train_y": y
                    }
        del gdt, y, ipt
        #valid
        config['valid_datapath'] = os.path.join(config['root_path'], config['valid_datapath'])
        img_raw = read_nii(data_path=config['valid_datapath'], csv_file='CT_ORG_valid.csv', num_volum=config['validating'].num_volum)
        img_raw = img_raw.transpose([2,0,1]).astype(np.float32)
        num_uesd = int(config['validating'].num_valid * img_raw.shape[0])
        num_uesd = num_uesd - divmod(num_uesd, config['num_gpus'])[1]
        img_raw = img_raw[0:num_uesd,:]#[8,254,254]
        fwd_valid = config['fwd_valid']
        gdt, y, ipt = gen_x_y(fwd_valid, img_raw, device,'valid')
        valid_set = {
                    "valid_ipt": ipt,
                    "valid_gdt": gdt,
                    "valid_y": y
                    }

    return train_set, valid_set, test_set