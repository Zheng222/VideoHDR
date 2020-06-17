import torch.utils.data as data
import os.path
from dataloading import common
import cv2


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]  # BGR --> RGB


IMG_EXTENSIONS = [
    '.png'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class color_adjust(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = self.opt.root
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/train_3nd_adjust_color_extracted/train'
        self.dir_hr = os.path.join(self.root, 'HDR_540p_mine')
        self.dir_lr = os.path.join(self.root, 'LDR_540p')

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr)
        return lr_tensor, hr_tensor

    def __len__(self):
        return self.opt.n_train

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        lr = default_loader(self.images_lr[idx])
        hr = default_loader(self.images_hr[idx])
        return lr, hr
