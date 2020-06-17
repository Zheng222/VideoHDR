from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import numpy as np
import os
import torch
from collections import OrderedDict
import cv2
import glob
import math
from torchvision.utils import make_grid


def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def tensor2np(tensor, out_type=np.uint16, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
    elif out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_state_dict(path):
    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


## only for reading LR (np.uint8 images)
def read_img(filename):
    ## read image by cv2, return HWC, BGR, [0, 1]
    if filename == '':
        print('Error' + filename)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = np.float32(img) / 65535.0

    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images

    Returns:
        imgs (Tensor):size (T, C, H, W), RGB, [0, 1]
    '''
    img_path_l = sorted(glob.glob(img_seq_path + '/*.png'))
    img_l = [read_img(v) for v in img_path_l]
    # stack to TCHW
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]  # BGR to RGB
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def tensor2img(tensor, out_type=np.uint16, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit16() WILL NOT round by default.
    elif out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)
