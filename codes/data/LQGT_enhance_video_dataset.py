import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import torch.nn as nn
import pdb
import cv2
import random
import lmdb
import os

class LQGT_video_dataset(data.Dataset):
    def __init__(self, opt):
        super(LQGT_video_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = [], []
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        # 获取视频帧路径
        self.video_folders_GT = get_video_folders(opt['dataroot_GT'])
        self.video_folders_LQ = get_video_folders(opt['dataroot_LQ'])

        for folder_GT, folder_LQ in zip(self.video_folders_GT, self.video_folders_LQ):
            frames_GT = sorted(os.listdir(folder_GT))
            frames_LQ = sorted(os.listdir(folder_LQ))
            self.paths_GT.extend([(folder_GT, os.path.join(folder_GT, f)) for f in frames_GT])
            self.paths_LQ.extend([(folder_LQ, os.path.join(folder_LQ, f)) for f in frames_LQ])

    def __getitem__(self, index):
        folder_GT, GT_path = self.paths_GT[index]
        folder_LQ, LQ_path = self.paths_LQ[index]

        img_GT = util.read_img(None, GT_path, None, dtype=np.uint8)
        img_LQ = util.read_img(None, LQ_path, None, dtype=np.uint16)

        GT_size = self.opt['GT_size']

        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        if self.opt['phase'] == 'val':
            H, W, C = img_LQ.shape
            pad_height = 0
            pad_width = 0
            if self.opt['pad32'] and (H % 32 != 0 or W % 32 != 0):
                pad_height = (32 - H % 32) % 32
                pad_width = (32 - W % 32) % 32
                img_LQ = np.pad(img_LQ, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')
                # img_GT = np.pad(img_GT, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path

        # 添加子文件夹名称到路径中
        GT_path_with_folder = os.path.join(os.path.basename(folder_GT) +  '_' + os.path.basename(GT_path))
        LQ_path_with_folder = os.path.join(os.path.basename(folder_LQ) +  '_' +  os.path.basename(LQ_path))
        # print(GT_path_with_folder)
        # pdb.set_trace()

        if self.opt['phase'] == 'val' and self.opt['pad32']:
            return {'LQ': img_LQ, 'GT': img_GT, 'pad_height': pad_height, 'pad_width': pad_width, 'LQ_path': LQ_path_with_folder, 'GT_path': GT_path_with_folder, 'name': os.path.basename(LQ_path), 'basename': os.path.join(os.path.basename(folder_LQ))}
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path_with_folder, 'GT_path': GT_path_with_folder, 'name': os.path.basename(LQ_path)}

    def __len__(self):
        return len(self.paths_GT)

def get_video_folders(root_dir):
    """获取根目录下所有子文件夹的路径"""
    return [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    #     self.opt = opt
    #     self.data_type = self.opt['data_type']
    #     self.paths_LQ, self.paths_GT = None, None
    #     self.sizes_LQ, self.sizes_GT = None, None
    #     self.LQ_env, self.GT_env = None, None  # environments for lmdb

    #     self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
    #     self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
    #     assert self.paths_GT, 'Error: GT path is empty.'
    #     if self.paths_LQ and self.paths_GT:
    #         assert len(self.paths_LQ) == len(
    #             self.paths_GT
    #         ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
    #             len(self.paths_LQ), len(self.paths_GT))
        
    #     if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
    #         self._init_lmdb()
    # def _init_lmdb(self):
    #     # https://github.com/chainer/chainermn/issues/129
    #     self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
    #                             meminit=False)
    #     self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
    #                             meminit=False)
    
    # def __getitem__(self, index):

    #     GT_path, LQ_path = None, None

    #     # get GT image
    #     GT_path = self.paths_GT[index]
    #     LQ_path = self.paths_LQ[index]
    #     resolution = [int(s) for s in self.sizes_GT[index].split('_')
    #                   ] if self.data_type == 'lmdb' else None

    #     img_GT = util.read_img(self.GT_env, GT_path, resolution, dtype=np.uint8)
    #     img_LQ = util.read_img(self.LQ_env, LQ_path, resolution, dtype=np.uint16)
    #     # print(np.max(img_GT), np.max(img_LQ))
    #     GT_size = self.opt['GT_size']
    #     # scale = self.opt['scale']
        
    #     if self.opt['color']:
    #         img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
    #         img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
        
    #     # augmentation for training
    #     if self.opt['phase'] == 'train':
    #         # if the image size is too small
    #         # H, W, _ = img_GT.shape
    #         # if H < GT_size or W < GT_size:
    #         #     img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
    #         #     # using matlab imresize
    #         #     img_LQ = util.imresize_np(img_GT, 1 / scale, True)
    #         #     if img_LQ.ndim == 2:
    #         #         img_LQ = np.expand_dims(img_LQ, axis=2)

    #         gt_size = self.opt['GT_size']
            
    #         if gt_size < 420:
    #             # padding
    #             img_GT, img_LQ = padding(img_GT, img_LQ, gt_size)
    #             # random crop
    #             img_GT, img_LQ = paired_random_crop(img_GT, img_LQ, gt_size, 1)
    #             # flip, rotation augmentations
    #             img_GT, img_LQ = random_augmentation(img_GT, img_LQ)
            
    #         else:
    #             # flip, rotation
    #             img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
    #                                         self.opt['use_rot'])

    #     # BGR to RGB, HWC to CHW, numpy to tensor
    #     if img_GT.shape[2] == 3:
    #         img_GT = img_GT[:, :, [2, 1, 0]]
    #         img_LQ = img_LQ[:, :, [2, 1, 0]]
        
    #     # print(img_GT.shape)
    #     if self.opt['phase'] == 'train':
    #         H, W, C = img_LQ.shape
    #         if (H%32 !=0 or W%32 !=0):  #  这里有两种处理方式，填充和随机裁剪
    #             if self.opt['pad32']:
    #                 # 计算需要填充的额外像素数
    #                 pad_height = (32 - H % 32) % 32
    #                 pad_width = (32 - W % 32) % 32

    #                 img_LQ = np.pad(img_LQ, ((0, pad_height), (0, pad_width), (0, 0)), 
    #                         mode='reflect')
    #                 img_GT = np.pad(img_GT, ((0, pad_height), (0, pad_width), (0, 0)), 
    #                         mode='reflect')
                    
    #     if self.opt['phase'] == 'val':
    #         H, W, C = img_LQ.shape
    #         pad_height = 0
    #         pad_width = 0
    #         if self.opt['pad32'] and (H%32 !=0 or W%32 !=0):  
    #             # 计算需要填充的额外像素数
    #             pad_height = (32 - H % 32) % 32
    #             pad_width = (32 - W % 32) % 32
    #             img_LQ = np.pad(img_LQ, ((0, pad_height), (0, pad_width), (0, 0)), 
    #                       mode='reflect')
    #             img_GT = np.pad(img_GT, ((0, pad_height), (0, pad_width), (0, 0)), 
    #                         mode='reflect')
                

    #     # print(img_LQ.shape, img_GT.shape)
    #     H, W, _ = img_LQ.shape
    #     img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
    #     img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
   
    #     if LQ_path is None:
    #         LQ_path = GT_path
    #     if self.opt['phase'] == 'val' and self.opt['pad32']:
    #         return {'LQ': img_LQ, 'GT': img_GT,'pad_height': pad_height ,'pad_width': pad_width ,'LQ_path': LQ_path, 'GT_path': GT_path}
    #     return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}
    
    # def __len__(self):
    #     return len(self.paths_GT)
    



def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        # print(gt_path)
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    # if h_lq < lq_patch_size or w_lq < lq_patch_size:
    #     raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
    #                      f'({lq_patch_size}, {lq_patch_size}). '
    #                      f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out