import random

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from utils.dist_utils import init_dist, get_dist_info


def val_patch(opt, val_loader, diffusion, skip=1024, padding=64):
    logger.info('** val patch inference skip %d pad %d' % (skip, padding))
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        fname = os.path.basename(val_set.sr_path[val_data['Index'][0]]).split('.')[0]
        sr_img_th = torch.zeros_like(val_data['HR'])
        input_sr = val_data['SR']

        _, _, H, W = input_sr.shape
        # skip, padding = 1024, 64
        shift = skip - padding * 2
        print('h w', H, W)
        for hs in range(0, H, skip):
            for ws in range(0, W, skip):
                print(hs, ws)
                hs = hs if hs < H else H - skip
                ws = ws if ws < W else W - skip
                val_data['SR'] = input_sr[..., hs:hs + skip, ws:ws + skip]
                with torch.no_grad():
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()
                sr_img_th[..., hs:hs + skip, ws:ws + skip] = visuals['SR'][-1]

        hr_img = Metrics.tensor2img(val_data['HR'])  # uint8
        lr_img = Metrics.tensor2img(val_data['SR'])  # uint8
        sr_img = Metrics.tensor2img(sr_img_th)

        # grid img
        Metrics.save_jpg(
            sr_img,
            '{}/{}_{}_{}_sr.png'.format(result_path, fname, idx, opt['name']))
        Metrics.save_jpg(
            hr_img, '{}/{}_{}_{}_hr.png'.format(result_path, fname, idx, opt['name']))
        Metrics.save_jpg(
            lr_img, '{}/{}_{}_{}_lr.png'.format(result_path, fname, idx, opt['name']))


def search_params(opt, val_loader, diffusion):
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        idx += 1
        fname = os.path.basename(val_set.sr_path[val_data['Index'][0]]).split('.')[0]

        res = []
        # for tt in [10, 20, 30, 50, 100, 200, 300, 500]:
        #     temp = []
        #     for endlr in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:

        if idx not in [4, 10]: continue  # for test 0.1
        for tt in [10, 50, 200, 500]:
            temp = []
            for endlr in [0.01, 0.05, 0.2, 0.5]:
                print('tt', tt, 'endlr', endlr)
                schedule_opt = opt['model']['beta_schedule'][opt['phase']]
                schedule_opt['n_timestep'] = tt
                schedule_opt['linear_end'] = endlr
                # print(schedule_opt)
                diffusion.set_new_noise_schedule(schedule_opt, schedule_phase=opt['phase'], force=True)
                tail = '-t%03d-%.2f' % (tt, endlr)
                with torch.no_grad():
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()

                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                sr_img_mode = 'grid'
                if sr_img_mode == 'single':
                    # single img series
                    sr_img = visuals['SR']  # uint8
                    sample_num = sr_img.shape[0]
                    for iter in range(0, sample_num):
                        Metrics.save_img(
                            Metrics.tensor2img(sr_img[iter]),
                            '{}/{}_{}_sr_{}.png'.format(result_path, fname, idx, iter))
                else:
                    # grid img
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    Metrics.save_jpg(sr_img,
                                     '{}/{}_{}_{}_sr_process.png'.format(result_path, fname, idx, opt['name']))
                    Metrics.save_jpg(Metrics.tensor2img(visuals['SR'][-1]),
                                     '{}/{}_{}_tt{}_end{}_{}_sr.png'.format(result_path, fname, idx, tt, endlr,
                                                                            opt['name'] + tail))

                # Metrics.save_img(
                #     hr_img, '{}/{}_{}_{}_hr.png'.format(result_path, current_step, idx, opt['name']+tail))
                # Metrics.save_img(
                #     lr_img, '{}/{}_{}_{}_lr.png'.format(result_path, current_step, idx, opt['name']+tail))
                # Metrics.save_img(
                #     fake_img, '{}/{}_{}_{}_inf.png'.format(result_path, current_step, idx, opt['name']+tail))

                temp.append(Metrics.tensor2img(visuals['SR'][-1]))
            res.append(temp)
        res = np.array(res)
        m, n, h, w, c = res.shape
        res = np.transpose(res, (0, 2, 1, 3, 4)).reshape((m * h, n * w, c))
        Metrics.save_jpg(res[..., ::-1], '%s/%s-final%d.png' % (result_path, opt['name'], idx))
        # import cv2
        # cv2.imwrite('%s/%s-final%d.png' % (result_path, opt['name'], idx), res[..., ::-1])


def model_wrapper(
        model,
        noise_schedule,
        model_type="noise",
        model_kwargs={},
        guidance_type="uncond",
        condition=None,
        unconditional_condition=None,
        guidance_scale=1.,
        classifier_fn=None,
        classifier_kwargs={},
):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(torch.cat([cond, x], dim=1), t_input, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output


    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier-free":
            return noise_pred_fn(x, t_continuous, cond=condition)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn

def dpm_solver(opt, val_loader, diffusion):
    from dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    betas = diffusion.netG.module.betas
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    for _, val_data in enumerate(val_loader):
        idx += 1
        fname = os.path.basename(val_set.sr_path[val_data['Index'][0]]).split('.')[0]
        with torch.no_grad():
            diffusion.feed_data(val_data)
            # diffusion.test(continous=True)
        # visuals = diffusion.get_current_visuals()
            initx = diffusion.netG.module.predictor(diffusion.data['SR'])
            model_fn = model_wrapper(
                diffusion.netG.module.denoise_fn,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs={'guide': initx},
                guidance_type="classifier-free",
                condition=diffusion.data['SR'],
                # unconditional_condition=unconditional_condition,
                # guidance_scale=guidance_scale,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

            x_T = torch.randn(diffusion.data['SR'].shape, device=diffusion.data['SR'].device)
            x_sample = dpm_solver.sample(
                x_T,
                steps=20,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )

            x_sample += initx


        hr_img = Metrics.tensor2img(diffusion.data['HR'].detach().float().cpu())  # uint8
        lr_img = Metrics.tensor2img(diffusion.data['SR'].detach().float().cpu())  # uint8
        fake_img = Metrics.tensor2img(diffusion.data['SR'].detach().float().cpu())  # uint8
        Metrics.save_jpg(
                Metrics.tensor2img(x_sample.detach().float().cpu()),
                '{}/{}_{}_sr.png'.format(result_path, fname, opt['name']))

        Metrics.save_jpg(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, fname, opt['name']))
        Metrics.save_jpg(
            lr_img, '{}/{}_{}_lr.png'.format(result_path, fname, opt['name']))
        Metrics.save_jpg(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, fname, opt['name']))
        exit()


def ddim_sampling(opt, val_loader, diffusion):
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        fname = os.path.basename(val_set.sr_path[val_data['Index'][0]]).split('.')[0]
        with torch.no_grad():
            diffusion.feed_data(val_data)
            # diffusion.test(continous=True)
            diffusion.netG.eval()

            initx = diffusion.netG.module.predictor(diffusion.data['SR'])
            diffusion.netG.module.pre_initx = initx
            kwargs = {'guide': initx}
            diffusion.SR = diffusion.netG.module.ddim_sample(diffusion.data['SR'], continous=False, kwargs=kwargs) + initx
            diffusion.netG.train()

        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        if hasattr(diffusion.netG.module, 'pre_initx'):
            fake_img = Metrics.tensor2img(diffusion.netG.module.pre_initx.detach().float().cpu())  # uint8
        else:
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]),
                    '{}/{}_{}_sr_{}.png'.format(result_path, fname, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            # Metrics.save_jpg(
            #     sr_img, '{}/{}_{}_{}_sr_process.png'.format(result_path, fname, idx, opt['name']))
            Metrics.save_jpg(
                Metrics.tensor2img(visuals['SR'][-1]),
                '{}/{}_{}_sr.png'.format(result_path, fname, opt['name']))

        Metrics.save_jpg(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, fname, opt['name']))
        Metrics.save_jpg(
            lr_img, '{}/{}_{}_lr.png'.format(result_path, fname, opt['name']))
        Metrics.save_jpg(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, fname, opt['name']))

        # generation
        eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim

        if wandb_logger and opt['log_eval']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr,
                                       eval_ssim)

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        current_epoch, current_step, avg_psnr, avg_ssim))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-launcher', default='slurm')

    parser.add_argument('--checkpoint', type=str, default=None)


    # parse configs
    args = parser.parse_args()
    # init dist first
    try:
        init_dist(args.launcher, backend='nccl')
    except:
        init_dist(args.launcher, backend='nccl')

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    opt['rank'], opt['world_size'] = get_dist_info()

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> {:s}'.format(
                        current_epoch, current_step, opt['name'])
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if (current_step % opt['train']['val_freq'] == 0 or current_step in [50]):
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        with torch.no_grad():
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img,
                            '{}/{}_{}_{}_hr.png'.format(result_path, current_step + opt['rank'], idx, opt['name']))
                        Metrics.save_img(
                            sr_img,
                            '{}/{}_{}_{}_sr.png'.format(result_path, current_step + opt['rank'], idx, opt['name']))
                        Metrics.save_img(
                            lr_img,
                            '{}/{}_{}_{}_lr.png'.format(result_path, current_step + opt['rank'], idx, opt['name']))
                        Metrics.save_img(
                            fake_img,
                            '{}/{}_{}_{}_inf.png'.format(result_path, current_step + opt['rank'], idx, opt['name']))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step + opt['rank']),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_models'] == 0 and opt['rank'] == 0:
                    diffusion.save_network(current_epoch, current_step)

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and opt['rank'] == 0:
                    logger.info('Saving models and training states.')
                    # diffusion.save_network(current_epoch, current_step)
                    diffusion.save_network('latest', '')

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation. len ' + str(len(val_loader)))
        # search_params(opt, val_loader, diffusion)
        # val_patch(opt, val_loader, diffusion)
        # dpm_solver(opt, val_loader, diffusion)
        # ddim_sampling(opt, val_loader, diffusion)
        # exit()

        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            logger.info('val index %d' % _)
            idx += 1
            fname = "".join(os.path.basename(val_set.sr_path[val_data['Index'][0]]).split('.')[:-1])
            with torch.no_grad():
                diffusion.feed_data(val_data)
                diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            if hasattr(diffusion.netG.module, 'pre_initx'):
                fake_img = Metrics.tensor2img(diffusion.netG.module.pre_initx.detach().float().cpu())  # uint8
            else:
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]),
                        '{}/{}_{}_sr_{}.png'.format(result_path, fname, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                # Metrics.save_jpg(
                #     sr_img, '{}/{}_{}_{}_sr_process.png'.format(result_path, fname, idx, opt['name']))
                Metrics.save_jpg(
                    Metrics.tensor2img(visuals['SR'][-1]),
                    '{}/{}_{}_sr.png'.format(result_path, fname, opt['name']))

            Metrics.save_jpg(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, fname, opt['name']))
            Metrics.save_jpg(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, fname, opt['name']))
            Metrics.save_jpg(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, fname, opt['name']))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr,
                                           eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
