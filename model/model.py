import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel

from utils.util import CephLoad

CE = CephLoad()

logger = logging.getLogger('base')

import copy


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.device = torch.device('cuda')
        temp_model = networks.define_G(opt)
        self.netG = self.set_device(temp_model)
        # print('** current', torch.cuda.current_device())
        self.netG = nn.parallel.DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                        output_device=opt['rank'], find_unused_parameters=False)
        self.schedule_phase = None

        ema_scheduler = opt['train']['ema_scheduler'].get('use', False)
        if ema_scheduler:
            self.ema_scheduler = opt['train']['ema_scheduler']
            self.netG_EMA = copy.deepcopy(temp_model)
            self.netG_EMA = nn.parallel.DistributedDataParallel(self.set_device(self.netG_EMA),
                                                                device_ids=[torch.cuda.current_device()],
                                                                output_device=opt['rank'], find_unused_parameters=False)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            optimizer_opt = opt['train']["optimizer"]
            logger.info("** optimizer ** " + optimizer_opt['type'])
            if optimizer_opt['type'] == "adamw":
                # self.optG = torch.optim.Adam(
                #     optim_params, lr=opt['train']["optimizer"]["lr"])
                self.optG = torch.optim.AdamW(optim_params, lr=optimizer_opt['lr'])
            elif optimizer_opt['type'] == "lion":
                from utils.util import Lion
                self.optG = Lion(optim_params, lr=optimizer_opt['lr'])
            elif optimizer_opt['type'] == "adam":
                self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        self.iter = 0

        from data.LRHR_dataset import MemcachedBase
        self.client = MemcachedBase()

        self.clip_norm = self.opt.get('clip_norm', None)
        logger.info('* clip norm %s' % self.clip_norm)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.clip_norm)
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

        if self.ema_scheduler is not None:
            if self.iter > self.ema_scheduler['step_start_ema'] and self.iter % self.ema_scheduler[
                'update_ema_every'] == 0:
                self.EMA.update_model_average(self.netG_EMA, self.netG)
        self.iter += 1

    def test(self, continous=False):
        self.netG.eval()

        pd = 64
        self.data['SR'] = F.pad(self.data['SR'], (pd, pd, pd, pd), mode='reflect')
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()
        self.SR = self.SR[..., pd:-pd, pd:-pd]
        self.data['SR'] = self.data['SR'][..., pd:-pd, pd:-pd]

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train', force=False):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase or force:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        # logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        CE.torchsave(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None, 'selfiter': self.iter}
        opt_state['optimizer'] = self.optG.state_dict()
        CE.torchsave(opt_state, opt_path)

        if self.ema_scheduler is not None:
            network = self.netG_EMA
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                network = network.module
            state_dict = network.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            CE.torchsave(state_dict, gen_path.replace('gen', 'gen_ema'))

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, nn.parallel.DistributedDataParallel):
                network = network.module

            if self.ema_scheduler and self.opt['phase'] != 'train':
                logger.info('** loading EMA model for evaluation.')
                network.load_state_dict(CE.torchload(
                    gen_path.replace('gen', 'gen_ema')), strict=False)
            else:
                network.load_state_dict(CE.torchload(
                    gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(CE.torchload(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'val':
                if os.path.exists(opt_path):
                    opt = CE.torchload(opt_path, map_location='cpu')
                    self.iter = opt['selfiter']
                    logger.info('*-*- selfiter %s' % self.iter)
                else:
                    logger.info('** cannot find opt model -> unknow iters')
            if self.opt['phase'] == 'train':
                # optimizer
                if os.path.exists(opt_path):
                    opt = CE.torchload(opt_path)
                    self.optG.load_state_dict(opt['optimizer'])
                    self.begin_step = opt['selfiter']  # opt['iter']
                    self.begin_epoch = 0  # opt['epoch']
                    self.iter = opt['selfiter']
                logger.info('*-*- selfiter %s' % self.iter)
                if self.ema_scheduler:
                    network = self.netG_EMA
                    if isinstance(self.netG_EMA, nn.DataParallel) or isinstance(self.netG_EMA,
                                                                                nn.parallel.DistributedDataParallel):
                        network = network.module
                    network.load_state_dict(CE.torchload(
                        gen_path.replace('gen', 'gen_ema')), strict=(not self.opt['model']['finetune_norm']))


class DDPM_bnoise(DDPM):
    def __init__(self, opt):
        super(DDPM_bnoise, self).__init__(opt)

    def feed_data(self, data):
        super().feed_data(data)  # move to device and assign to self.data
        temp = self.data['SR']
        maxmin = 2
        sigma = 50. / 255.
        self.data['SR'] = temp + torch.randn_like(temp) * maxmin * sigma
        return


import random
import numpy as np

import torch.nn.functional as F
from data.degradations import USMSharp, filter2D, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from data.diffjpeg import DiffJPEG


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


dopt = {
    'scale': 4,
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.15, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 30],
    'poisson_scale_range': [0.05, 3],
    'gray_noise_prob': 0.4,
    'jpeg_range': [30, 95],

    # the second degradation process
    'second_blur_prob': 0.8,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 25],
    'poisson_scale_range2': [0.05, 2.5],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [30, 95],

    'gt_size': 256,
    'queue_size': 180,
}
dopt1 = {
    'scale': 4,
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.3, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 15],
    'poisson_scale_range': [0.05, 2.0],
    'gray_noise_prob': 0.4,
    'jpeg_range': [60, 95],

    # the second degradation process
    'second_blur_prob': 0.5,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.6, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 12],
    'poisson_scale_range2': [0.05, 1.0],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [60, 100],

    # 'gt_size': 512,
    # 'no_degradation_prob': 0.01,
    'gt_size': 256,
    'queue_size': 180,
}

dopt1gt = dopt1.copy()
dopt1gt.update({
    'gt_size': 256 + 128,
    'queue_size': 181,  # ???
})


class DDPM_realsr(DDPM):
    def __init__(self, opt):
        super(DDPM_realsr, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp(radius=15).cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        self.is_train = opt['phase'] == 'train'
        self.dopt = eval(opt['dopt'])  # degradation options

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            # print(self.lq.shape, self.gt.shape, b, self.queue_ptr)
            if self.queue_size < self.queue_ptr + b:  # avoid size error
                assert self.queue_size - self.queue_ptr <= b
                b = self.queue_size - self.queue_ptr
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()[:b]
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()[:b]
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.dopt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.dopt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.dopt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.dopt['gray_noise_prob']
            if np.random.uniform() < self.dopt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.dopt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.dopt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.dopt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.dopt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.dopt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.dopt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.dopt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.dopt['scale'] * scale), int(ori_w / self.dopt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.dopt['gray_noise_prob2']
            if np.random.uniform() < self.dopt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.dopt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.dopt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.dopt['scale'], ori_w // self.dopt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.dopt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.dopt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.dopt['scale'], ori_w // self.dopt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.dopt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.dopt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            self.lq = F.interpolate(self.lq, scale_factor=self.dopt['scale'], mode='bilinear')  # scale up
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

        self.data = {}
        self.data['SR'] = self.lq
        if self.opt.get('gt_usm', True):
            self.data['HR'] = self.gt_usm  # use sharpen
        else:
            self.data['HR'] = self.gt

        # normalize to -1~1
        for k in self.data.keys():
            self.data[k] = self.data[k] * 2. - 1.
        # print('(** l=q', self.lq.shape, self.gt.shape, self.gt_usm.shape)
        # exit()
