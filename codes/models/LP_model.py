import logging
from collections import OrderedDict
import pdb
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, VGGLoss, ColorLossDetaE, MS_SSIM_Loss, SSIM_Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F
logger = logging.getLogger('base')

torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # define pixel loss
            l_pix_type = train_opt['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif l_pix_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(l_pix_type))
            self.l_pix_w = train_opt['pixel_weight']

            # define per loss
            l_fea_type = train_opt['feature_criterion']
            if l_fea_type == 'vgg':
                self.cri_vgg = VGGLoss().to(self.device)
            self.l_fea_w = train_opt['feature_weight']

            # define ssim loss
            l_ssim_type = train_opt['ssim_criterion']
            if l_ssim_type == 'ssim':
                self.cri_ssim = SSIM_Loss().to(self.device)
            elif l_ssim_type == 'mssim':
                self.cri_ssim = MS_SSIM_Loss(3).to(self.device)
            
            self.l_ssim_w = train_opt['ssim_weight']           
            self.l_hf_w = train_opt['hf_weight']           

            self.l_lf_w = train_opt['lf_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            if train_opt['finetune_adafm']:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    v.requires_grad = False
                    if k.find('adafm') >= 0:
                        v.requires_grad = True
                        optim_params.append(v)
                        logger.info('Params [{:s}] will optimize.'.format(k))
            else:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        self.kernel = self.gauss_kernel()


    def feed_data(self, data, need_GT=True, need_cond=False):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
        if need_cond:
            self.cond = data['cond'].to(self.device) # cond
            self.input = [self.var_L, self.cond]
        else:
            self.input = self.var_L

    def feed_data_with_HF(self, data, need_GT=True, need_cond=False):
        self.input = data['LQ'].to(self.device)  # LQ   
        self.real_H = data['GT'].to(self.device)  # GT 

        self.GT_HF = self.pyramid_decom(self.real_H)   # return list, include 
        self.GT_LF = self.guassion_decom(self.real_H)[::-1]  # return list


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        if 'LP3DLUT' in self.opt['name']:
            self.sr_imgs, self.HF = self.netG(self.input, 'train')
            # cul loss
            if self.l_hf_w != 0:
                loss_hf = sum(self.cri_pix(input_hf, gt_hf) for input_hf, gt_hf in zip(self.HF, self.GT_HF))
            else:
                loss_hf = torch.tensor(0.)
            
            loss_lf = sum(self.cri_pix(output_guass, gt_guass) for output_guass, gt_guass in zip(self.sr_imgs, self.GT_LF)) 
            
            del self.HF
            torch.cuda.empty_cache()
            # loss_ssim = self.cri_ssim(self.sr_imgs[-1], self.GT_LF[-1]) + self.cri_ssim(self.sr_imgs[-2], self.GT_LF[-2])
            loss_ssim = self.cri_ssim(self.sr_imgs[-1], self.GT_LF[-1])
            torch.cuda.empty_cache()

            if '480p' in self.opt['name']:   
                loss_vgg = self.cri_vgg(self.sr_imgs[-1], self.real_H)
                # loss_ssim = self.cri_ssim(self.sr_imgs[-1], self.GT_LF[-1])
            else:
                 
                loss_vgg = self.cri_vgg(F.interpolate(self.sr_imgs[-1], scale_factor=0.125), F.interpolate(self.real_H, scale_factor=0.125))   

            del self.sr_imgs, self.real_H
            
            losses =  self.l_lf_w * (self.l_pix_w * loss_lf + self.l_ssim_w * loss_ssim + \
                        self.l_fea_w * loss_vgg) + self.l_hf_w * loss_hf
            
            losses.backward()
            self.optimizer_G.step()

            # set log
            self.log_dict['l_lf'] = loss_lf.item()
            self.log_dict['l_ssim'] = loss_ssim.item()
            self.log_dict['l_vgg'] = loss_vgg.item()
            self.log_dict['l_hf'] = loss_hf.item()
            del losses, loss_lf, loss_ssim, loss_vgg, loss_hf
            torch.cuda.empty_cache()
        
        else:
            self.sr_imgs = self.netG(self.input)
            loss_l1 = self.cri_pix(self.sr_imgs, self.real_H)   

            loss_l1.backward()
            self.optimizer_G.step()
            self.log_dict['l_l1'] = loss_l1.item()
        
    def test(self):
        
        if 'LP3DLUT' in self.opt['name']:
            
            self.netG.eval()
            with torch.no_grad():
                self.fake_H = self.netG(self.input, 'test')

            # torchvision.utils.save_image(HF[0], deflare_path)

            self.netG.train()
        
        else:
            self.netG.eval()
            with torch.no_grad():
                self.sr_imgs = self.netG(self.input)
                self.fake_H = self.sr_imgs
                torch.cuda.empty_cache()
            self.netG.train()  
            
    def test_patches(self, patchsize):
        self.netG.eval()
        import numpy as np
        with torch.no_grad():
            _, _, height, width= self.input.shape
            inp = torch.from_numpy(np.pad(self.input.data.cpu().numpy(), ((0,0),(0,0), (patchsize[0],patchsize[0]), (patchsize[1],patchsize[1])), 'reflect')).cuda()
            output = torch.zeros_like(inp)
            lineWeights = self.centeredCosineWindow(torch.arange(patchsize[0]), patchsize[0]).reshape(-1, 1).repeat(1, patchsize[1])
            columnWeights = self.centeredCosineWindow(torch.arange(patchsize[1]), patchsize[1]).reshape(1, -1).repeat(patchsize[0],1)
            window = (lineWeights * columnWeights).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).cuda()

            for r in range(0, inp.shape[2]-patchsize[0], patchsize[0]//2):
                for c in range(0, inp.shape[3]-patchsize[1], patchsize[1]//2):
                    patch = inp[:,:,r:r+patchsize[0], c:c+patchsize[1]]
                    output_patch, _  = self.netG(patch)
                    output[:,:,r:r+patchsize[0], c:c+patchsize[1]] += output_patch[-1]*window
        self.fake_H = output[:,:,patchsize[0]:height+patchsize[0],patchsize[1]:width+patchsize[1]]
        self.netG.train()
    
    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict
    
    def get_current_visuals_torch(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()
        out_dict['rlt'] = self.fake_H.detach()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()
        return out_dict
    
    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def update(self, new_model_dict):
        if isinstance(self.netG, nn.DataParallel):
            network = self.netG.module
            network.load_state_dict(new_model_dict)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(4):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr[:4]
    
    def guassion_decom(self, img):
        current = img
        pyr = []
        pyr.append(current)
        for _ in range(4):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            pyr.append(down)
            current = down
        return pyr
    
    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    def centeredCosineWindow(self, x, windowSize=16):
        import numpy as np
        y = 1 / 2 - 1 / 2 * torch.cos(2 * torch.tensor(np.pi) * (x + 1 / 2.) / windowSize)
        return y