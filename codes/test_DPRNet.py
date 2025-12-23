import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
import time
import argparse
from collections import OrderedDict
import torch
import lpips
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.TMQI import TMQI
import pdb

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=False, help='Path to options YMAL file.')

opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

lpips_fn_alex = lpips.LPIPS(net='alex').cuda()
metric_color = util.ColorLossDetaE()
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['tmqi'] = []
    test_results['lpips'] = []
    test_results['deltaE'] = []

    for data in test_loader:
        need_GT = True
 
        model.feed_data(data, need_GT=need_GT, need_cond=False)

        img_path = data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]
        if opt['datasets']['val']['patchsize']:
            model.test_patches(opt['datasets']['val']['patchsize'])
        else:
            model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)
        if opt['datasets']['val']['pad32']:
            _, h, w = visuals['rlt'].shape
            ori_h = h - data['pad_height']
            ori_w = w - data['pad_width']
            visuals['rlt'] = visuals['rlt'][:,:ori_h,:ori_w]
            visuals['LQ'] = visuals['LQ'][:,:ori_h,:ori_w]


        sr_img = util.tensor2img(visuals['rlt'])  # uint8
        ori_img = util.tensor2img(visuals['LQ'])
        # pdb.set_trace()
        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            ssim = 0
            psnr = 0
            tmqi = 0
            lpips_val = 0
            deltaE = 0
            psnr = util.calculate_psnr(sr_img, gt_img)
            ssim = util.calculate_ssim(sr_img, gt_img)
            tmqi = TMQI()(ori_img, sr_img)
            lpips_val = lpips_fn_alex(visuals['rlt'].unsqueeze(0).cuda(), visuals['GT'].unsqueeze(0).cuda()).cpu().detach().numpy()[0][0][0][0]
            deltaE = metric_color(visuals['rlt'].unsqueeze(0).cuda(), visuals['GT'].unsqueeze(0).cuda()).cpu().detach().numpy()
            
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['tmqi'].append(tmqi)
            test_results['lpips'].append(lpips_val)
            test_results['deltaE'].append(deltaE)

            logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}, TMQI: {:.6f}, LPIPS: {:.6f}, deltaE: {:.6f}.'.format(img_name, psnr, ssim, tmqi, lpips_val, deltaE))
        else:
            logger.info(img_name)

    if need_GT:  # metrics

        ave_ssim = 0
        ave_tmqi = 0
        ave_lpips = 0
        ave_deltaE = 0
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_tmqi = sum(test_results['tmqi']) / len(test_results['tmqi'])
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        ave_deltaE = sum(test_results['deltaE']) / len(test_results['deltaE'])
        

        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n; TMQI: {:.6f}\n, LPIPS: {:.6F}, DELTAE: {:.6F}'.format(
                test_set_name, ave_psnr, ave_ssim, ave_tmqi, ave_lpips, ave_deltaE))
