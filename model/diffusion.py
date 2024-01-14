import math
import torch
from torch import device, nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger('base')
import lpips


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    logger.info('*** schedule ' + schedule)
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def identity(t, *args, **kwargs):
    return t


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])  # 1e-6 1e-2 2k
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas  # 1- 1e-6 1 - 1e-2 2k
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 1 - 4e-5
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(  # x0 keeped ratio 1 - 0.006
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / (alphas_cumprod + 1e-10))))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / (alphas_cumprod + 1e-10) - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
               self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, kwargs={}):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, **kwargs))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level, **kwargs))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, kwargs={}):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, kwargs=kwargs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, kwargs={}):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, kwargs=kwargs)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x, kwargs=kwargs)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    # based on p_mean_variance
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False, kwargs={}):
        # model_output = self.model(x, t, x_self_cond)

        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        model_output = self.denoise_fn(torch.cat([x_self_cond, x], dim=1), noise_level, **kwargs)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        from collections import namedtuple
        ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, x_in, continous=False, kwargs={}):
        shape = x_in.shape
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        self.ddim_sampling_eta = 1  # 0-> ddim 1 -> ddpm
        self.sampling_timesteps = 5
        self.objective = 'pred_noise'
        batch, total_timesteps, sampling_timesteps, eta = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_in
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True,
                                                             rederive_pred_noise=False, kwargs=kwargs)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not continous else torch.stack(imgs, dim=1)

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# from utils.lpips import LPIPS

class PerceptualGaussianDiffusion(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(PerceptualGaussianDiffusion, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                          schedule_opt)
        self.lpips = lpips.LPIPS(net='vgg', eval_mode=True)  # input range -1, 1
        logger.info('* diffusion ' + self.__class__.__name__)

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)

        factor = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        x_start_recon = x_noisy - ((1 - factor ** 2).sqrt() * x_recon)
        x_start_recon = x_start_recon / factor
        p_loss = self.lpips(x_start_recon, x_start)  # input range -1, 1; output size b, 1, 1, 1

        return loss + 1.0 * p_loss


from model.ucdir import UNetSeeInDark


class ResiGaussianDiffusion(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(ResiGaussianDiffusion, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                    schedule_opt)

        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_in['HR'] - x_init  # residule input
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        return self.p_sample_loop(x_in, continous) + initx


# pass initial predicter to guide dynamic model
class ResiGaussianGuideDY(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(ResiGaussianGuideDY, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                  schedule_opt)

        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_in['HR'] - x_init  # residule input
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # if not self.conditional:
        #     x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        # else:
        kwargs = {'guide': x_init}
        x_recon = self.denoise_fn(torch.cat([x_in['SR'], x_noisy], dim=1),
                                  continuous_sqrt_alpha_cumprod, **kwargs)

        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        self.pre_initx = initx
        kwargs = {'guide': initx}
        return self.p_sample_loop(x_in, continous, kwargs=kwargs) + initx

# degraded guidance
class ResiGaussianGuideDY_de(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(ResiGaussianGuideDY_de, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                  schedule_opt)

        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_in['HR'] - x_init  # residule input
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # if not self.conditional:
        #     x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        # else:
        kwargs = {'guide': x_in['SR']}
        x_recon = self.denoise_fn(torch.cat([x_in['SR'], x_noisy], dim=1),
                                  continuous_sqrt_alpha_cumprod, **kwargs)

        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        self.pre_initx = initx
        kwargs = {'guide': x_in}
        return self.p_sample_loop(x_in, continous, kwargs=kwargs) + initx




class ResiGaussianGuideDY_initxloss(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(ResiGaussianGuideDY_initxloss, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                  schedule_opt)

        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_in['HR'] - x_init  # residule input
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # if not self.conditional:
        #     x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        # else:
        kwargs = {'guide': x_init}
        x_recon = self.denoise_fn(torch.cat([x_in['SR'], x_noisy], dim=1),
                                  continuous_sqrt_alpha_cumprod, **kwargs)

        loss = self.loss_func(noise, x_recon)
        loss = loss + 0.5 * self.loss_func(x_init, x_in['HR'])
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        self.pre_initx = initx
        kwargs = {'guide': initx}
        return self.p_sample_loop(x_in, continous, kwargs=kwargs) + initx

class ResiPercepGaussianDiffusion(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(ResiPercepGaussianDiffusion, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                          schedule_opt)

        self.predictor = UNetSeeInDark()
        # self.lpips = lpips.LPIPS(net='vgg', eval_mode=True)  # input range -1, 1
        logger.info('* diffusion ' + self.__class__.__name__)

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_in['HR'] - x_init  # residule input
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)

        factor = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        x_start_recon = x_noisy - ((1 - factor ** 2).sqrt() * x_recon)
        x_start_recon = x_start_recon / factor  # get the estimated x_start(residule
        # p_loss = self.lpips(x_start_recon + x_init, x_in['HR'])  # input range -1, 1; output size b, 1, 1, 1
        p_loss = 0
        p_loss += self.loss_func(x_start_recon + x_init, x_in['HR'])
        # print(loss.mean(), p_loss.mean())
        return loss + 0.5 * p_loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        return self.p_sample_loop(x_in, continous) + initx


class NoDiffusion(GaussianDiffusion):
    def __init__(self, denoise_fn, image_size, channels=3, loss_type='l1',
                 conditional=True, schedule_opt=None):
        super(NoDiffusion, self).__init__(denoise_fn, image_size, channels, loss_type, conditional,
                                                    schedule_opt)

        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None):
        x_init = self.predictor(x_in['SR'])
        x_start = x_init 
        [b, c, h, w] = x_start.shape
        # t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
           [self.sqrt_alphas_cumprod_prev[1]] * b
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        x_recon = self.denoise_fn(x_start, continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(x_in['HR'], x_recon)
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        x_start = initx 
        [b, c, h, w] = x_start.shape
        # t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
           [self.sqrt_alphas_cumprod_prev[1]] * b
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        x_recon = self.denoise_fn(x_start, continuous_sqrt_alpha_cumprod)
        return x_recon