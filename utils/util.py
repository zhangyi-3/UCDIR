import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def patch_forward(noisy, kpn_net, params={}, skip=512, padding=32):
    ''' Args:
        noisy: b c h w
        kpn_net:
    '''
    assert noisy.is_cuda
    pd = min(noisy.shape[-1], noisy.shape[-2])
    pd = skip - pd + padding if pd < skip else padding

    noisy = F.pad(noisy, (pd, pd, pd, pd), mode='reflect')
    denoised = torch.zeros_like(noisy)[:, :3]  # assume output channel == 3

    _, _, H, W = noisy.shape
    shift = skip - padding * 2
    for i in np.arange(0, H, shift):
        for j in np.arange(0, W, shift):
            h_start, h_end = i, i + skip
            w_start, w_end = j, j + skip
            # print('\nidx', h_start, h_end, w_start, w_end)
            if h_end > H:
                h_end = H
                h_start = H - skip
            if w_end > W:
                w_end = W
                w_start = W - skip
            # print('\nidx2', h_start, h_end, w_start, w_end, H, W)
            patch = noisy[..., h_start: h_end, w_start: w_end]
            with torch.no_grad():
                input_var = patch
                out_var = kpn_net(input_var, **params)

            out = out_var
            denoised[..., h_start + padding: h_end - padding, w_start + padding: w_end - padding] = \
                out[..., padding:-padding, padding:-padding]
    return denoised[..., pd:-pd, pd:-pd]


def patch_forward_guide(noisy, kpn_net, params={}, skip=512, padding=32):
    ''' Args:
        noisy: b c h w
        kpn_net:
    '''
    assert noisy.is_cuda
    pd = min(noisy.shape[-1], noisy.shape[-2])
    pd = skip - pd + padding if pd < skip else padding

    noisy = F.pad(noisy, (pd, pd, pd, pd), mode='reflect')
    guide_pad = F.pad(params['guide'], (pd, pd, pd, pd), mode='reflect')

    denoised = torch.zeros_like(noisy)[:, :3]  # assume output channel == 3

    _, _, H, W = noisy.shape
    shift = skip - padding * 2
    for i in np.arange(0, H, shift):
        for j in np.arange(0, W, shift):
            h_start, h_end = i, i + skip
            w_start, w_end = j, j + skip
            # print('\nidx', h_start, h_end, w_start, w_end)
            if h_end > H:
                h_end = H
                h_start = H - skip
            if w_end > W:
                w_end = W
                w_start = W - skip
            # print('\nidx2', h_start, h_end, w_start, w_end, H, W)
            patch = noisy[..., h_start: h_end, w_start: w_end]
            guide_patch = guide_pad[..., h_start: h_end, w_start: w_end]
            params['guide'] = guide_patch
            with torch.no_grad():
                input_var = patch
                out_var = kpn_net(input_var, **params)

            out = out_var
            denoised[..., h_start + padding: h_end - padding, w_start + padding: w_end - padding] = \
                out[..., padding:-padding, padding:-padding]
    return denoised[..., pd:-pd, pd:-pd]


import io, os


class CephLoad(object):
    def __init__(self):
        try:
            from petrel_client.client import Client
            conf_path = '~/petreloss.conf'
            self.client = Client(conf_path)  # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
        except:
            self.client = None

        self.ceph = self.client

    def rename(self, path):
        if 'zhangyi2.vendor' in path:
            url = 's3://zhangyi/' + path.split('zhangyi2.vendor/')[-1]
        else:
            url = path
        return url

    def list(self, url):
        if self.client is None:
            return os.listdir(url)
        url = self.rename(url)
        filelist = list(self.client.list(url))
        return filelist

    def put(self, url):
        url = self.rename(url)
        self.client.put(url)
        assert NotImplementedError

    def delete(self, url):
        url = self.rename(url)
        self.client.put(url)
        assert NotImplementedError

    def ceph_rename(self, path):
        path = os.path.abspath(path)
        if 'share/images/' in path:
            shortpath = path.split('share/images/')[-1]
            img_url = 's3://zy3_imagenet/' + shortpath
            shortpath = path.split('share/')[-1]
            img_url = 's3://zhangyi/' + shortpath
        elif '.vendor/' in path:
            shortpath = path.split('.vendor/')[-1]
            img_url = 's3://zhangyi/' + shortpath
        elif 'zhangyi3/Restormer/' in path:
            shortpath = path.split('zhangyi3/Restormer/')[-1]
            img_url = 's3://Restormer/' + shortpath
        else:
            assert NotImplementedError
        return img_url

    def get_stream(self, path):
        url = self.ceph_rename(path)
        # print(url, path, 'get')
        value = self.ceph.get(url)
        value_buf = io.BytesIO(value)
        value_buf.seek(0)
        return value_buf

    def put_stream(self, path, obj):  # buffer = io.BytesIO()
        url = self.ceph_rename(path)
        # print(url, path, 'put')
        obj.seek(0)
        self.ceph.put(url, obj)
        return

    def torchsave(self, state_dict, path):
        try:
            torch.save(state_dict, path)
        except:
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            self.put_stream(path, buffer)
            return

    def torchload(self, path, map_location='cpu'):
        try:
            return torch.load(path, map_location=map_location)
        except:
            test = self.get_stream(path)
            test = torch.load(test, map_location=map_location)
            return test

    def isdir(self, path):
        url = self.ceph_rename(path)
        return self.ceph.isdir(url)
