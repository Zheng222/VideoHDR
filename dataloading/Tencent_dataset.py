import torch.utils.data as data
import os
import random
import torch
import numpy as np
import cv2

def _make_dataset(dir):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.
    2D List Structure:
    [[frame00, frame01,...,frameM],  <-- clip0
     [frame00, frame01,...,frameM],  <-- clip1
     ...,
     [frame00, frame01,...,frameM]   <-- clipN
    ]
    Parameters
        dir : string
            root directory containing clips.
    Tips
        read 700 clips, each of them contains 100 frames (only read list)
    """
    framePath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framePath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framePath[index].append(os.path.join(clipsFolderPath, image))
    return framePath


def _make_video_dataset(dir):
    """
    Creates a 1D list of all the frames.
    1D List Structure:
    [frame0, frame1,...,frameN]
    """
    framePath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framePath.append(os.path.join(dir, image))
    return framePath

class RandomCrop(object):
    def __init__(self, video_size, patch_size, scale):
        ih, iw = video_size

        self.tp = patch_size
        self.ip = self.tp // scale

        self.ix = random.randrange(0, iw - self.ip + 1)
        self.iy = random.randrange(0, ih - self.ip + 1)

        self.tx, self.ty = scale * self.ix, scale * self.iy

    def __call__(self, clip, mode='target'):
        if mode == 'target':
            ret = clip[:, self.ty:self.ty + self.tp, self.tx:self.tx + self.tp, :]  # [T, H, W, C]

        else:
            ret = clip[:, self.iy:self.iy + self.ip, self.ix:self.ix + self.ip, :]

        return ret


class Agument(object):
    def __init__(self):
        self.hflip = random.random() < 0.5
        self.vflip = random.random() < 0.5
        self.rot90 = random.random() < 0.5

    def augment(self, video):
        if self.hflip: video = video[:, :, ::-1, :]
        if self.vflip: video = video[:, ::-1, :, :]
        if self.rot90: video = video.transpose(0, 2, 1, 3)  # T(0), H(1), W(2), C(3) --> T, W, H, C
        return video

    def __call__(self, video):
        return self.augment(video)

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]  # BGR --> RGB

class MultiFramesDataset(data.Dataset):
    """
    Youku dataset
    A dataset for loading N samples arranged in this way:
        |-- clip0
            |-- frame00
            |-- frame01
            ...
            |-- frame05
            |-- frame06
        |-- clip1
            |-- frame00
            |-- frame01
            ...
            |-- frame05
            |-- frame06
        ...
        |-- clipN
            |-- frame00
            |-- frame01
            ...
            |-- frame05
            |-- frame06
    Example: 7 frames, N denotes batch_size
    Attributes
    framesPath : list
        List of frames' path in the dataset.
    """

    def __init__(self, opt, mode):
        framesPath = _make_dataset(os.path.join(opt.root, mode, "HDR_540p_generated"))  # HDR_540p_generated
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + opt.root + "\n")
        self.opt = opt
        self.root = opt.root
        self.mode = mode
        self.framesPath = framesPath
        
        # load lr and gt frames
        self.lr_dir = os.path.join(self.root, self.mode, "HDR_540p_generated")
        self.gt_dir = os.path.join(self.root, self.mode, "HDR_4k")


    def __getitem__(self, index):
        current_clip = self.framesPath[index]
        T = len(current_clip)

        current_frame_idx = random.randint(self.opt.nframes // 2,
                                           T - self.opt.nframes // 2 - 1)  # if self.opt.nframes=5, [2, 97] from [0, 99]

        frame_lr = []
        frame_gt = []
        for t in range(current_frame_idx - self.opt.nframes // 2, current_frame_idx + self.opt.nframes // 2 + 1):

            frame_lr.append(default_loader(current_clip[t]))

            if t == current_frame_idx:
                frame_gt.append(default_loader(current_clip[t].replace('HDR_540p_generated', 'HDR_4k')))

        # data crop and augmentation
        if self.mode == 'train':
            # random crop
            get_patch = RandomCrop(frame_lr[0].shape[:2], patch_size=self.opt.patch_size, scale=4)
            augment = Agument()

            clip_lr = np.stack(frame_lr, axis=0)  # T = 7
            clip_gt = np.stack(frame_gt, axis=0)  # T = 1

            clip_lr = get_patch(clip_lr, mode='input')
            clip_gt = get_patch(clip_gt, mode='target')

            if self.opt.geometry_aug:
                clip_lr = augment(clip_lr)
                clip_gt = augment(clip_gt)


        # convert (T, H, W, C) array to (T, C, H, W) tensor
        tensor_hdr_540p = torch.from_numpy(np.float32(np.ascontiguousarray(np.transpose(clip_lr, (0, 3, 1, 2))))) / 65535.0
        tensor_hdr_4k = torch.from_numpy(np.float32(np.ascontiguousarray(np.transpose(np.squeeze(clip_gt), (2, 0, 1))))) / 65535.0

        return tensor_hdr_540p, tensor_hdr_4k  # tensor_hdr_540p --> (T, C, H, W), tensor_hdr_4k --> (C, H, W)

    def __len__(self):
        return len(self.framesPath)