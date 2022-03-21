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
from DataFidelities.IDTClass import *

#%%
np.random.seed(128)

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################

def gen_x_y(config, img_raw, emParams, device, set_mode:str):
    IMG_PATCH, NBF_KEEP = config['IMG_Patch'], config['numKeep']
    print(NBF_KEEP)
    imgList = []
    for m in range(1):
        for index in range(img_raw.shape[0]):
            img = img_raw[index].copy()
            # img = data_augmentation(img, m)
            # img = np.ascontiguousarray(img)
            # img = view_as_windows(img, window_shape=[IMG_PATCH, IMG_PATCH], step=30)
            # img = np.ascontiguousarray(img)
            img.shape = [-1] + [IMG_PATCH, IMG_PATCH]
            img = img.astype(np.float32)
            img = np.expand_dims(img, 1)  # Add channel in the dimension right after batch-dimension.

            imgList.append(img)

    img = np.concatenate(imgList, 0)

    for index in range(img.shape[0]):
        img[index] = img[index] - np.amin(img[index])
        img[index] = img[index] / np.amax(img[index])
        img[index] = img[index]

    gdt = torch.from_numpy(img).to(device)  # (72, 1, 90, 90)
    ipt = torch.zeros(size=gdt.shape + (2, ), dtype=torch.float32).to(device)  # (72, 1, 90, 90, 2)
    y   = torch.zeros([gdt.shape[0], NBF_KEEP, 1, IMG_PATCH, IMG_PATCH, 2], dtype=torch.float32) # (72, 60, 1, 90, 90, 2)

    for index in tqdm(range(gdt.shape[0]), 'IDT %s Dataset'%(set_mode)):

        ph_ = gdt[index]  # (1, 90, 90)
        ab_ = torch.zeros_like(ph_)

        # Convert it to complex
        ph_ = torch.stack([ph_, torch.zeros_like(ph_)], -1) # [1 90 90 2]
        ab_ = torch.stack([ab_, torch.zeros_like(ab_)], -1)
        # to complex
        intensityPred = torch.unbind(iFFT(IDTClass.fmult(FFT(ph_), FFT(ab_), emParams)), -1)[0]  # torch.Size([60, 1, 90, 90])

        data = torch.zeros((NBF_KEEP, 1, IMG_PATCH, IMG_PATCH), dtype=torch.float32)
        for j in range(0, NBF_KEEP):
            data[j, 0] = addwgn_torch(intensityPred[j, 0], 20)[0]

        data = torch.stack([data, torch.zeros_like(data)], -1).to(device)
        y_each = FFT(data)  # torch.Size([60, 1, 90, 90, 2])
       
        ipt_each = IDTClass.ftran(y_each, emParams, 'Ph')
        
        y[index]   = y_each
        ipt[index] = ipt_each

    print("%s Dataset: "%(set_mode))
    print('gdt: ', gdt.shape, gdt.dtype)
    print('ipt: ', ipt.shape, ipt.dtype)
    print('y: ', y.shape, y.dtype)

    return gdt, y, ipt

def set2use(data_root, numSet=1, phase='train'):
    
    imgList = []

    if phase == 'train':

        if numSet == 0:
            setName = ['BSD400']
        elif numSet == 1:
            setName = ['BSD400', 'DIV2K']
        elif numSet == 2:
            setName = ['BSD400', 'DIV2K', 'Flickr2K']
        elif numSet == 3:
            setName = ['BreCaHAD_Train']

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList
    elif phase == 'valid':

        if numSet == 0:
            setName = ['Set12']  
        elif numSet == 1:    
            setName = ['BreCaHAD_Test'] 

        for setInd in setName:
            pathList = get_image_paths(os.path.join(data_root, setInd))
            imgList = imgList + pathList    

    return imgList

def get_images(datapath, numSet=3, IMG_PATCH=320, phase='train', tp='matlab'):
    dtype = torch.FloatTensor #Choose tensor datatype, defult : gpu.float32
    g = torch.Generator().manual_seed(40)

    ###############################################
    #                 Start Loading               #
    ###############################################

    if tp == 'matlab':
        data_mat = h5py.File(datapath,'r')
        img_raw = h5py2mat(data_mat['imgs'])    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
        data_mat.close()
    elif tp == 'imgs':     

        imgList = set2use(datapath, numSet, phase=phase)[1:200]
        num_scales = [1]
        img_raw = []

        for index in tqdm(imgList):

            for scaling in num_scales:
                
                for m in range(1):
                    img = scaling * uint2single(imread_uint(index, n_channels=1))
                    img = data_augmentation(img, m)
                    patches = patches_from_image(img, p_size=IMG_PATCH, p_overlap=0, p_max=1000)
                    patches = patches.transpose(0,3,1,2)
                    img_raw.append(patches)

        img_raw = np.concatenate(img_raw, 0)
                                                                                                                
        print('img_gdt: ', img_raw.shape, img_raw.dtype)
        
    return img_raw

def get_matrix(data_path, NBF_KEEP):

    index_sets = {}
    Hreal_list, Himag_list = [], []

    try:
        IDTData = sio.loadmat(data_path)
        Himag_tmp = IDTData['Himag'].astype(np.float32)[:, :, :, :NBF_KEEP]
        Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])
        Hreal_temp = np2torch_complex(IDTData['Hreal'].astype(np.complex64)[:, :, :, :NBF_KEEP]).permute([3, 2, 0, 1, 4])
    except NotImplementedError:

        IDTData = h5py.File(data_path,'r') #sio.loadmat(config['dataset']['measure_path_train']) 

        for data in IDTData["Hreal"]:
            
            data_of_interest_reference = data[0]
            data_of_interest = IDTData[data_of_interest_reference]
            data_of_interest =  np.array(data_of_interest, dtype=np.float32).transpose([1,2,3,4,0])#[:, :, :, :NBF_KEEP]

            data_of_interest = torch.from_numpy(data_of_interest)

            Hreal_list.append(data_of_interest)
            
        # Hreal_list = Hreal_list
        Hreal_temp = torch.cat(Hreal_list)

        print(Hreal_temp.shape)

        for data in IDTData["Himag"]:
            
            data_of_interest_reference = data[0]
            data_of_interest = IDTData[data_of_interest_reference]
            data_of_interest =  np.array(data_of_interest, dtype=np.float32)#[:, :, :, :NBF_KEEP]

            data_of_interest = torch.from_numpy(data_of_interest)

            # print(data_of_interest.dtype)
            # print(data_of_interest.shape)
            # print(type(data_of_interest))

            Himag_list.append(data_of_interest)

        Himag_tmp = torch.cat(Himag_list)

        # Himag_tmp = np.array(IDTData['Himag'],dtype=np.float32).transpose([3, 2, 1, 0]) #Himag h5py_> numpy : [480 1 960 960].transpose([3, 2, 1, 0]) _> [960 960 1 480]
        # Himag_tmp = Himag_tmp[:, :, :, :NBF_KEEP]
        # Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])

        # Hreal_temp = np.array(IDTData['Hreal'], dtype=np.float32)#[:, :, :, :NBF_KEEP]
        # Hreal_temp = Hreal_temp.transpose([1,2,3,4,0])#(Hreal_temp[0,...] + 1j * Hreal_temp[1,...]).transpose([3, 2, 1, 0])
        # Hreal_temp = torch.from_numpy(Hreal_temp[:NBF_KEEP, :, :, :]) 
        
        used_index, angle_index = None, None

        for name in IDTData['record']:
            
            if  name == 'used_index':
               used_index =  np.array(IDTData['record'][name]).squeeze()
               index_sets[name] = used_index
            elif name == 'angle_index':
                angle_index = np.array(IDTData['record'][name]).squeeze().transpose()
                index_sets[name] = angle_index  
        IDTData.close()        
    except Exception as err:
        raise Exception(err)

    return Hreal_temp, Himag_tmp, Hreal_list, Himag_list, index_sets 

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
        config['measure_path'].test_datapath = os.path.join(config['root_path'], config['measure_path'].test_datapath)
        IMG_PATCH, NBF_KEEP = config['fwd_test'].IMG_Patch, config['fwd_test'].numKeep  # Here is based on IDT Data itself, unchangeable.
        IDTData = sio.loadmat(config['measure_path'].test_datapath)
        Himag_tmp = IDTData['Himag'].astype(np.float32)[:, :, :, :NBF_KEEP]
        Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])
        
        emParamsTest = {
            'Hreal': np2torch_complex(IDTData['Hreal'].astype(np.complex64)[:, :, :, :NBF_KEEP]).permute([3, 2, 0, 1, 4]).to(device),# torch.Size([60, 1, 90, 90, 2])
            'Himag': Himag_tmp.to(device),  # torch.Size([60, 1, 90, 90, 2])
            'NBFkeep': NBF_KEEP
        }
        config['data_path'].test_datapath = os.path.join(config['root_path'], config['data_path'].test_datapath)
        data_mat = h5py.File(config['data_path'].test_datapath,'r')
        img_raw = h5py2mat(data_mat['imgs'])    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
        data_mat.close()
        num_uesd = int(config['testing'].num_test * img_raw.shape[0])
        img_raw = img_raw[0:num_uesd,:].astype(np.float32)#[8,254,254]
        fwd_test = config['fwd_test']
        gdt, y, ipt = gen_x_y(fwd_test, img_raw, emParamsTest, device, set_mode)
        test_set = {
                    "test_ipt": ipt,
                    "test_gdt": gdt,
                    "test_y": y,
                    "emParamsTest": emParamsTest
                    }
    elif set_mode=='train':
        #train
        config['measure_path'].train_datapath = os.path.join(config['root_path'], config['measure_path'].train_datapath)
        IMG_PATCH, NBF_KEEP = config['fwd_train'].IMG_Patch, config['fwd_train'].numKeep  # Here is based on IDT Data itself, unchangeable.
        
        Hreal_temp, _, Hreal_list, _, index_sets = get_matrix(config['measure_path'].train_datapath, NBF_KEEP)

        emParamsTrain = {
            'Hreal': Hreal_temp.to(device),# torch.Size([60, 1, 90, 90, 2])
            'NBFkeep': NBF_KEEP, #'Himag': Himag_temp.to(device),  # torch.Size([60, 1, 90, 90, 2]),
            'index_sets' : index_sets
        }

        config['data_path'].train_datapath = os.path.join(config['root_path'], config['data_path'].train_datapath)

        img_raw = get_images(config['data_path'].train_datapath, IMG_PATCH=config['fwd_train'].IMG_Patch, phase='train', tp='imgs')

        num_uesd = int(config['training'].num_train * img_raw.shape[0])
        num_uesd = (num_uesd//config['training'].batch_size)*config['training'].batch_size
        num_uesd = np.random.choice(img_raw.shape[0], num_uesd, replace=False).tolist()
        num_uesd.sort()
        print(num_uesd)
        img_raw = img_raw[num_uesd,:].astype(np.float32)#[8,254,254]
        fwd_train = config['fwd_train']
        gdt, y, ipt = gen_x_y(fwd_train, img_raw, emParamsTrain, device, 'train')

        emParamsTrain['Hreal'] = Hreal_list

        train_set = {
                    "train_ipt": ipt,
                    "train_gdt": gdt,
                    "train_y": y,
                    "emParamsTrain": emParamsTrain,
                    }

        # print('Saving the dataset . . .')
        # with h5py.File('BreCaHAD_HW=512_NBFkeep=%d_left.h5'%(NBF_KEEP), 'w') as hf:
        #     hf.create_dataset("train_ipt",  data=np.float32(ipt.detach().cpu().numpy()))
        #     hf.create_dataset("train_gdt",  data=np.float32(gdt.detach().cpu().numpy()))
        #     hf.create_dataset("train_y",  data=np.float32(y.detach().cpu().numpy()))
        # print('. . . Finished')
        # exit()
        #valid
        config['measure_path'].valid_datapath = os.path.join(config['root_path'], config['measure_path'].valid_datapath)
        IMG_PATCH, NBF_KEEP = config['fwd_valid'].IMG_Patch, config['fwd_valid'].numKeep

        Hreal_temp, _, Hreal_list, _, index_sets = get_matrix(config['measure_path'].valid_datapath, NBF_KEEP)

        emParamsValid = {
            'Hreal': Hreal_temp.to(device),# torch.Size([60, 1, 90, 90, 2])
            'NBFkeep': NBF_KEEP, #'Himag': Himag_temp.to(device),  # torch.Size([60, 1, 90, 90, 2])
            'index_sets' : index_sets
        }
        
        config['data_path'].valid_datapath = os.path.join(config['root_path'], config['data_path'].valid_datapath)

        img_raw = get_images(config['data_path'].valid_datapath, 
                        IMG_PATCH=config['fwd_valid'].IMG_Patch, numSet=1, phase='valid', tp='imgs')

        num_uesd = int(config['validating'].num_valid * img_raw.shape[0])
        num_uesd = (num_uesd//config['validating'].batch_size)*config['validating'].batch_size
        # img_raw = img_raw[0:num_uesd,:].astype(np.float32)#[8,254,254]
        num_uesd = np.random.choice(img_raw.shape[0], num_uesd, replace=False).tolist()
        num_uesd.sort()
        img_raw = img_raw[num_uesd,:].astype(np.float32)#[8,254,254]
        print(img_raw.shape)
        fwd_valid = config['fwd_valid']
        gdt, y, ipt = gen_x_y(fwd_valid, img_raw, emParamsValid, device, 'valid')

        emParamsValid['Hreal'] = Hreal_list

        valid_set = {
                    "valid_ipt": ipt,
                    "valid_gdt": gdt,
                    "valid_y": y,
                    "emParamsValid": emParamsValid
                    
                    }
    return train_set, valid_set, test_set

# ######################################################################################### Nov22/2021
# import torch
# from torch import nn

# import h5py
# import numpy as np
# from tqdm import tqdm
# import scipy.io as sio
# from datetime import datetime
# from skimage.util import view_as_windows

# from utils.util import *
# from DataFidelities.IDTClass import *

# #%%
# np.random.seed(128)

# ################Functions######################
# FFT  = lambda x: torch.fft(x,  signal_ndim=2)
# iFFT = lambda x: torch.ifft(x, signal_ndim=2)
# rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
# ###############################################

# def gen_x_y(config, img_raw, emParams, device, set_mode:str):
#     IMG_PATCH, NBF_KEEP = config['IMG_Patch'], config['numKeep']
#     print(NBF_KEEP)
#     imgList = []
#     for m in range(1):
#         for index in range(img_raw.shape[0]):
#             img = img_raw[index].copy()
#             # img = data_augmentation(img, m)
#             # img = np.ascontiguousarray(img)
#             # img = view_as_windows(img, window_shape=[IMG_PATCH, IMG_PATCH], step=30)
#             # img = np.ascontiguousarray(img)
#             img.shape = [-1] + [IMG_PATCH, IMG_PATCH]
#             img = img.astype(np.float32)
#             img = np.expand_dims(img, 1)  # Add channel in the dimension right after batch-dimension.

#             imgList.append(img)

#     img = np.concatenate(imgList, 0)

#     for index in range(img.shape[0]):
#         img[index] = img[index] - np.amin(img[index])
#         img[index] = img[index] / np.amax(img[index])
#         img[index] = img[index]

#     gdt = torch.from_numpy(img)  # (72, 1, 90, 90)
#     ipt = torch.zeros(size=gdt.shape + (2, ), dtype=torch.float32)  # (72, 1, 90, 90, 2)
#     y   = torch.zeros([gdt.shape[0], NBF_KEEP, 1, IMG_PATCH, IMG_PATCH, 2], dtype=torch.float32) # (72, 60, 1, 90, 90, 2)

#     for index in tqdm(range(gdt.shape[0]), 'IDT %s Dataset'%(set_mode)):

#         ph_ = gdt[index].to(device)  # (1, 90, 90)
#         ab_ = torch.zeros_like(ph_)

#         # Convert it to complex
#         ph_ = torch.stack([ph_, torch.zeros_like(ph_)], -1) # [1 90 90 2]
#         ab_ = torch.stack([ab_, torch.zeros_like(ab_)], -1)
#         # to complex
#         intensityPred = torch.unbind(iFFT(IDTClass.fmult(FFT(ph_), FFT(ab_), emParams)), -1)[0]  # torch.Size([60, 1, 90, 90])

#         data = torch.zeros((NBF_KEEP, 1, IMG_PATCH, IMG_PATCH), dtype=torch.float32)
#         for j in range(0, NBF_KEEP):
#             data[j, 0] = addwgn_torch(intensityPred[j, 0], 20)[0]

#         data = torch.stack([data, torch.zeros_like(data)], -1).to(device)
#         y_each = FFT(data)  # torch.Size([60, 1, 90, 90, 2])
       
#         ipt_each = IDTClass.ftran(y_each, emParams, 'Ph')
        
#         y[index]   = y_each.to('cpu')
#         ipt[index] = ipt_each.to('cpu')

#     print("%s Dataset: "%(set_mode))
#     print('gdt: ', gdt.shape, gdt.dtype)
#     print('ipt: ', ipt.shape, ipt.dtype)
#     print('y: ', y.shape, y.dtype)

#     return gdt, y, ipt

# def set2use(data_root, numSet=1, phase='train'):
    
#     imgList = []

#     if phase is 'train':

#         if numSet == 0:
#             setName = ['BSD400']
#         elif numSet == 1:
#             setName = ['BSD400', 'DIV2K']
#         elif numSet == 2:
#             setName = ['BSD400', 'DIV2K', 'Flickr2K']
#         elif numSet == 3:
#             setName = ['BreCaHAD_Train_subset']

#         for setInd in setName:
#             pathList = get_image_paths(os.path.join(data_root, setInd))
#             imgList = imgList + pathList
#     elif phase is 'valid':

#         if numSet == 0:
#             setName = ['Set12']  
#         elif numSet == 1:    
#             setName = ['BreCaHAD_Test'] 

#         for setInd in setName:
#             pathList = get_image_paths(os.path.join(data_root, setInd))
#             imgList = imgList + pathList    

#     return imgList

# def get_images(datapath, numSet=3, IMG_PATCH=320, phase='train', tp='matlab'):
#     dtype = torch.FloatTensor #Choose tensor datatype, defult : gpu.float32
#     g = torch.Generator().manual_seed(40)

#     ###############################################
#     #                 Start Loading               #
#     ###############################################

#     if tp == 'matlab':
#         data_mat = h5py.File(datapath,'r')
#         img_raw = h5py2mat(data_mat['imgs'])    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#         data_mat.close()
#     elif tp == 'imgs':     

#         imgList = set2use(datapath, numSet, phase=phase)[1:200]
#         num_scales = [1]
#         img_raw = []

#         for index in tqdm(imgList):

#             for scaling in num_scales:
                
#                 for m in range(1):
#                     img = scaling * uint2single(imread_uint(index, n_channels=1))
#                     img = data_augmentation(img, m)
#                     patches = patches_from_image(img, p_size=IMG_PATCH, p_overlap=0, p_max=400)
#                     patches = patches.transpose(0,3,1,2)
#                     img_raw.append(patches)

#         img_raw = np.concatenate(img_raw, 0)
                                                                                                                
#         print('img_gdt: ', img_raw.shape, img_raw.dtype)
        
#     return img_raw

# def get_matrix(data_path, NBF_KEEP):

#     try:
#         IDTData = sio.loadmat(data_path)
#         Himag_tmp = IDTData['Himag'].astype(np.float32)[:, :, :, :NBF_KEEP]
#         Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])
#         Hreal_temp = np2torch_complex(IDTData['Hreal'].astype(np.complex64)[:, :, :, :NBF_KEEP]).permute([3, 2, 0, 1, 4])
#     except NotImplementedError:

#         IDTData = h5py.File(data_path,'r') #sio.loadmat(config['dataset']['measure_path_train']) 
    
#         Himag_tmp = np.array(IDTData['Himag'],dtype=np.float32).transpose([3, 2, 1, 0]) #Himag h5py_> numpy : [480 1 960 960].transpose([3, 2, 1, 0]) _> [960 960 1 480]
#         Himag_tmp = Himag_tmp[:, :, :, :NBF_KEEP]
#         Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])

#         Hreal_temp = np.array(IDTData['Hreal'], dtype=np.float32)#[:, :, :, :NBF_KEEP]
#         Hreal_temp = Hreal_temp.transpose([1,2,3,4,0])#(Hreal_temp[0,...] + 1j * Hreal_temp[1,...]).transpose([3, 2, 1, 0])
#         Hreal_temp = torch.from_numpy(Hreal_temp[:NBF_KEEP, :, :, :]) 
#     except Exception as err:
#         raise Exception(err)

#     return Hreal_temp, Himag_tmp    

# def data_preprocess(config:dict, device=None, set_mode='test'):

#     ###############################################
#     #                 Start Loading               #
#     ###############################################
#     # Train
#     train_set, valid_set, test_set = {},{},{}
#     ###############################################
#     #                 Start Loading               #
#     ###############################################
#     if set_mode=='test':
#         config['measure_path'].test_datapath = os.path.join(config['root_path'], config['measure_path'].test_datapath)
#         IMG_PATCH, NBF_KEEP = config['fwd_test'].IMG_Patch, config['fwd_test'].numKeep  # Here is based on IDT Data itself, unchangeable.
#         IDTData = sio.loadmat(config['measure_path'].test_datapath)
#         Himag_tmp = IDTData['Himag'].astype(np.float32)[:, :, :, :NBF_KEEP]
#         Himag_tmp = torch.stack([torch.from_numpy(Himag_tmp), torch.from_numpy(Himag_tmp)], -1).permute([3, 2, 0, 1, 4])
        
#         emParamsTest = {
#             'Hreal': np2torch_complex(IDTData['Hreal'].astype(np.complex64)[:, :, :, :NBF_KEEP]).permute([3, 2, 0, 1, 4]).to(device),# torch.Size([60, 1, 90, 90, 2])
#             'Himag': Himag_tmp.to(device),  # torch.Size([60, 1, 90, 90, 2])
#             'NBFkeep': NBF_KEEP
#         }
#         config['data_path'].test_datapath = os.path.join(config['root_path'], config['data_path'].test_datapath)
#         data_mat = h5py.File(config['data_path'].test_datapath,'r')
#         img_raw = h5py2mat(data_mat['imgs'])    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#         data_mat.close()
#         num_uesd = int(config['testing'].num_test * img_raw.shape[0])
#         img_raw = img_raw[0:num_uesd,:].astype(np.float32)#[8,254,254]
#         fwd_test = config['fwd_test']
#         gdt, y, ipt = gen_x_y(fwd_test, img_raw, emParamsTest, device, set_mode)
#         test_set = {
#                     "test_ipt": ipt,
#                     "test_gdt": gdt,
#                     "test_y": y,
#                     "emParamsTest": emParamsTest
#                     }
#     elif set_mode=='train':
#         #train
#         config['measure_path'].train_datapath = os.path.join(config['root_path'], config['measure_path'].train_datapath)
#         IMG_PATCH, NBF_KEEP = config['fwd_train'].IMG_Patch, config['fwd_train'].numKeep  # Here is based on IDT Data itself, unchangeable.
        
#         Hreal_temp, Himag_temp = get_matrix(config['measure_path'].train_datapath, NBF_KEEP)

#         emParamsTrain = {
#             'Hreal': Hreal_temp.to(device),# torch.Size([60, 1, 90, 90, 2])
#             'NBFkeep': NBF_KEEP #'Himag': Himag_temp.to(device),  # torch.Size([60, 1, 90, 90, 2])
#         }

#         config['data_path'].train_datapath = os.path.join(config['root_path'], config['data_path'].train_datapath)

#         img_raw = get_images(config['data_path'].train_datapath, phase='train', tp='matlab')

#         num_uesd = int(config['training'].num_train * img_raw.shape[0])
#         img_raw = img_raw[0:num_uesd,:].astype(np.float32)#[8,254,254]
#         fwd_train = config['fwd_train']
#         gdt, y, ipt = gen_x_y(fwd_train, img_raw, emParamsTrain, device, 'train')

#         emParamsTrain['Hreal'] = Hreal_temp

#         train_set = {
#                     "train_ipt": ipt,
#                     "train_gdt": gdt,
#                     "train_y": y,
#                     "emParamsTrain": emParamsTrain
#                     }
#         #valid
#         config['measure_path'].valid_datapath = os.path.join(config['root_path'], config['measure_path'].valid_datapath)
#         IMG_PATCH, NBF_KEEP = config['fwd_valid'].IMG_Patch, config['fwd_valid'].numKeep

#         Hreal_temp, Himag_temp = get_matrix(config['measure_path'].valid_datapath, NBF_KEEP)

#         emParamsValid = {
#             'Hreal': Hreal_temp.to(device),# torch.Size([60, 1, 90, 90, 2])
#             'NBFkeep': NBF_KEEP #'Himag': Himag_temp.to(device),  # torch.Size([60, 1, 90, 90, 2])
#         }
        
#         config['data_path'].valid_datapath = os.path.join(config['root_path'], config['data_path'].valid_datapath)

#         img_raw = get_images(config['data_path'].valid_datapath, phase='valid', tp='matlab')

#         num_uesd = int(config['validating'].num_valid * img_raw.shape[0])
#         img_raw = img_raw[0:num_uesd,:].astype(np.float32)#[8,254,254]
#         fwd_valid = config['fwd_valid']
#         gdt, y, ipt = gen_x_y(fwd_valid, img_raw, emParamsValid, device, 'valid')

#         emParamsValid['Hreal'] = Hreal_temp

#         valid_set = {
#                     "valid_ipt": ipt,
#                     "valid_gdt": gdt,
#                     "valid_y": y,
#                     "emParamsValid": emParamsValid
#                     }
#     return train_set, valid_set, test_set