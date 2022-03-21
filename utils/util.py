import torch
import h5py
import math
import shutil
import cv2
import os
import numpy as np
import torch.nn as nn
import scipy.io as sio
import tifffile as tiff
from typing import List, Dict, Optional, Sequence, Tuple, Union

evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
# Image file extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.h5','.mat']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def h5py2mat(data):
    result = np.array(data,dtype=np.float32)

    if len(result.shape) == 3 and result.shape[0] > result.shape[1]:
        result = result.transpose([1,0,2])
    elif len(result.shape) == 3 and result.shape[1] < result.shape[2]:
        result = result.transpose([2,1,0])
    elif len(result.shape) == 3 and result.shape[1] > result.shape[2]:
        result = result.transpose([2,1,0])     
    elif len(result.shape) == 4:
        result = np.array(data).transpose([0,1,3,2])       
    print(result.shape)    
    return result

def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([res_real, res_imag], -1)

def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths

def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
'''

def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
        w1.append(w-p_size)
        h1.append(h-p_size)
        # print(w1)
        # print(h1)
        for i in w1:
            for j in h1:
                patches.append(img[i:i+p_size, j:j+p_size,:])
    else:
        patches.append(img)

    return np.array(patches, dtype=np.float32)

def imssave(imgs, img_path):
    """ 
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path), img_name+str('_{:04d}'.format(i))+'.png')
        cv2.imwrite(new_path, img)

def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=512, p_overlap=96, p_max=800):
    """
    split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size), 
    and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
    will be splitted.

    Args:
        original_dataroot:
        taget_dataroot:
        p_size: size of small images
        p_overlap: patch size in training is a good choice
        p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
    """
    paths = get_image_paths(original_dataroot)
    for img_path in paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = imread_uint(img_path, n_channels=n_channels)
        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches, os.path.join(taget_dataroot, os.path.basename(img_path)))
        #if original_dataroot == taget_dataroot:
        #del img_path

'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''
# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE # HxW
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

# --------------------------------------------
# get h5 images
# --------------------------------------------

def imread_uint_h5(path):
    #  input: path
    # output: SlicexCoilHxWx2(real and imag)
    file = h5py.File(path, "r")
    kspace = file['kspace']
    kspace = torch.tensor(kspace)[0:10]

    ## crop ####
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1) #[10 20 640 320 2]
    image_4d = ifft2c(kspace)
    if kspace.shape[-2] < 320:
        a = (320 - kspace.shape[-2])//2
        padding = (0, 0, a, a)
        image_4d = F.pad(image_4d, padding, 'constant', 0)
    image_4d = complex_center_crop(image_4d, [320, 320])
    image_3d = rss_complex(image_4d, dim=1)


    ## normlization to 0-1
    # mean = image_3d.mean()
    # mean = image_3d.mean()
    # std = image_3d.std()
    # eps = 1e-10
    # image_3d = (image_3d - mean) / (std + eps)
    # image_3d = (image_3d-image_3d.min())/(image_3d.max()-image_3d.min())

    # sio.savemat('image_3d.mat',{'img':image_3d.detach().cpu().numpy()})
    ### multiple a scale
    # image_3d = image_3d*1000
    img_rss_ori = np.array(file['reconstruction_rss'])[0:10]

    return image_3d, img_rss_ori

# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

# --------------------------------------------
# get single image of size HxWxn_channles (BGR)
# --------------------------------------------
def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(uint)
# numpy(single) <--->  tensor
# numpy(uint)   <--->  tensor
# --------------------------------------------
'''
# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def uint162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint16((img.clip(0, 1)*65535.).round())

###################
# Read Images
###################

def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)

def addwgn_torch(x: torch.Tensor, inputSnr):
    noiseNorm = torch.norm(x.flatten() * 10 ** (-inputSnr / 20))

    noise = torch.randn(x.shape[-2], x.shape[-1]).to(noiseNorm.device)
    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    rec_y = x + noise.to(x.device)#.cuda()

    return rec_y, noise

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))

def compute_rsnr(x, xhat):
    """
    x -> xtrue
    xhat -> xpre
    """
    if len(x.shape) == 2:
        A = np.zeros((2, 2))
        A[0, 0] = np.sum(xhat.flatten('F')**2)
        A[0, 1] = np.sum(xhat.flatten('F'))
        A[1, 0] = A[0, 1]
        A[1, 1] = x.size

        b = np.zeros((2, 1))
        b[0] = np.sum(x.flatten('F') * xhat.flatten('F'))
        b[1] = np.sum(x.flatten('F'))

        try:
            c = np.matmul(np.linalg.inv(A), b)
        except np.linalg.LinAlgError:
            c = [0, 0]
            print('xhat is all zeros.')

        evaluateSnr = lambda xtrue, x: 20 * np.log10(
            np.linalg.norm(xtrue.flatten('F')) / np.linalg.norm(xtrue.flatten('F') - x.flatten('F')))

        avg_snr = evaluateSnr(x, xhat)
        avg_rsnr = evaluateSnr(x, c[0]*xhat+c[1])
    elif len(x.shape) == 3 and x.shape[0] < x.shape[1]:
        rsnr = np.zeros([1,x.shape[0]])
        snr = np.zeros_like(rsnr)
        for num_imgs in range(0,x.shape[0]):
            A = np.zeros((2, 2))
            A[0, 0] = np.sum(xhat.flatten('F')**2)
            A[0, 1] = np.sum(xhat.flatten('F'))
            A[1, 0] = A[0, 1]
            A[1, 1] = x.size
            b = np.zeros((2, 1))
            b[0] = np.sum(x.flatten('F') * xhat.flatten('F'))
            b[1] = np.sum(x.flatten('F'))
            try:
                c = np.matmul(np.linalg.inv(A), b)
            except np.linalg.LinAlgError:  
                c = [0, 0]
                print('xhat is all zeros.')
            evaluateSnr = lambda xtrue, x: 20 * np.log10(
                  np.linalg.norm(xtrue.flatten('F')) / np.linalg.norm(xtrue.flatten('F') - x.flatten('F')))
            snr[:,num_imgs] = evaluateSnr(x, xhat)
            rsnr[:,num_imgs] = evaluateSnr(x, c[0]*xhat+c[1])
        avg_rsnr = np.mean(rsnr)
        avg_snr =  np.mean(snr)
    else:
        avg_rsnr = np.zeros([1,1])
        avg_snr = np.zeros([1,1])
    return avg_rsnr, avg_snr
    
def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        # print(item)
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def data_augmentation(image, mode):
    out = None
    if len(image.shape) ==3:
        out = augment_img_np3(image, mode)
    elif len(image.shape) ==2:
        out = augment_img_np2(image, mode)
    return out    

def augment_img_np2(image, mode):

    out = image

    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out                

def augment_img_np3(img, mode=0):
    out = None
    if mode == 0:
        out = img
    elif mode == 1:
        out = img.transpose(1, 0, 2)
    elif mode == 2:
        out = img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        out = img.transpose(1, 0, 2)
    elif mode == 4:
        out = img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        out = img.transpose(1, 0, 2)
    elif mode == 6:
        img = img[:, ::-1, :]
        out = img[::-1, :, :]
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        out = img.transpose(1, 0, 2)
    
    return out

def get_model_vars(network):
    lst_vars = []
    num_count = 0
    for para in network.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())
        lst_vars.append(para)
    return lst_vars

def pretty(d, indent=0):
    ''' Print dictionary '''
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def total_variation(x):

    """
    total variation regularizer

    Input:
    x ~ input image [bs,x,y,channels]
    type ~ different ways of calculating total variation 'isotropic' & 'anisotropic'

    Output:
    tv_loss ~ total variation loss
    """
    dx = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]   # [1,1,127,127]
    dy = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]   # [1,1,127,127]

    tv_loss = torch.mean(torch.sum(torch.abs(dx) + torch.abs(dy), (1,2,3)))

    return tv_loss 

def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))

def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)

def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)

def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()

def fft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    # data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    # data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    # data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    # data = fftshift(data, dim=[-3, -2])

    return data

# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[:, :, i] -= np.amin(x[:, :, i])
                x[:, :, i] /= np.amax(x[:, :, i])
            #
                x[:, :, i] *= 255

            x = x.astype(np.uint8)
            # x = x.astype(np.float32)
    x = x.astype(np.float32)
    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]