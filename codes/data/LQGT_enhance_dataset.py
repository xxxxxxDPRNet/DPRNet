import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import torch.nn as nn
import pdb
import cv2
import random
import lmdb

class LQGT_enhance_dataset(data.Dataset):
    def __init__(self, opt):
        super(LQGT_enhance_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)
    
    def __getitem__(self, index):

        GT_path, LQ_path = None, None

        # get GT image
        GT_path = self.paths_GT[index]
        LQ_path = self.paths_LQ[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None

        img_GT = util.read_img(self.GT_env, GT_path, resolution, dtype=np.uint8)
        img_LQ = util.read_img(self.LQ_env, LQ_path, resolution, dtype=np.uint16)
        # print(np.max(img_GT), np.max(img_LQ))
        # GT_size = self.opt['GT_size']
        # scale = self.opt['scale']
        
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            # if the image size is too small
            # H, W, _ = img_GT.shape
            # if H < GT_size or W < GT_size:
            #     img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            #     # using matlab imresize
            #     img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            #     if img_LQ.ndim == 2:
            #         img_LQ = np.expand_dims(img_LQ, axis=2)

            # H, W, C = img_LQ.shape
            # LQ_size = GT_size // scale

            # # randomly crop
            # rnd_h = random.randint(0, max(0, H - LQ_size))
            # rnd_w = random.randint(0, max(0, W - LQ_size))
            # img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            # rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            
            # flip, rotation
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        
        # print(img_GT.shape)
        if self.opt['phase'] == 'train':
            H, W, C = img_LQ.shape
            if (H%32 !=0 or W%32 !=0):  #  这里有两种处理方式，填充和随机裁剪
                if self.opt['pad32']:
                    # 计算需要填充的额外像素数
                    pad_height = (32 - H % 32) % 32
                    pad_width = (32 - W % 32) % 32

                    img_LQ = np.pad(img_LQ, ((0, pad_height), (0, pad_width), (0, 0)), 
                            mode='reflect')
                    img_GT = np.pad(img_GT, ((0, pad_height), (0, pad_width), (0, 0)), 
                            mode='reflect')
                    
        if self.opt['phase'] == 'val':
            H, W, C = img_LQ.shape
            pad_height = 0
            pad_width = 0
            if self.opt['pad32'] and (H%32 !=0 or W%32 !=0):  
                # 计算需要填充的额外像素数
                pad_height = (32 - H % 32) % 32
                pad_width = (32 - W % 32) % 32
                img_LQ = np.pad(img_LQ, ((0, pad_height), (0, pad_width), (0, 0)), 
                          mode='reflect')


        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        
        if LQ_path is None:
            LQ_path = GT_path
        if self.opt['phase'] == 'val' and self.opt['pad32']:
            return {'LQ': img_LQ, 'GT': img_GT,'pad_height': pad_height ,'pad_width': pad_width ,'LQ_path': LQ_path, 'GT_path': GT_path}
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}
    
    def __len__(self):
        return len(self.paths_GT)
