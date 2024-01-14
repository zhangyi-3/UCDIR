import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime
from omegaconf import OmegaConf


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb
    # # remove comments starting with '//'
    # json_str = ''
    # with open(opt_path, 'r') as f:
    #     for line in f:
    #         line = line.split('//')[0] + '\n'
    #         json_str += line
    # opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt = OmegaConf.load(opt_path)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    if args.phase == 'val':
        opt['name'] = 'val_{}'.format(opt['name'])

    fix = ''
    # validation in train phase
    if phase == 'val':

        opt['path']['resume_state'] = args.checkpoint  # "experiments/%s/checkpoint/I_Elatest" % _state
        print('\roverwirte resume sate', args.checkpoint, end='**')

        opt['datasets']['val']["data_args"]['data_len'] = -1

        if 'sr-' in opt['name']:
            opt['datasets']['val']["data_args"]['data_len'] = 5000  # select first 500 for sr val

        opt['datasets']['val']['data_args']['split'] = 'val'

        # modify the evaluation path
        if 'sid' in opt['name']:

            opt['model']['beta_schedule']['val']['n_timestep'] = 50
            opt['model']['beta_schedule']['val']['linear_end'] = 4e-1

        elif 'gop-' in opt['name']:
            opt['datasets']['val']["data_args"]["dataroot"] = {
                "lq": "../Restormer/Motion_Deblurring/Datasets/test/GoPro_sub/input/",
                "gt": "../Restormer/Motion_Deblurring/Datasets/test/GoPro_sub/target/"}

            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq":  "../Restormer/Motion_Deblurring/Datasets/test/HIDE_sub/input/",
            # "gt":  "../Restormer/Motion_Deblurring/Datasets/test/HIDE_sub/target/"}
            # fix += 'hide'

            opt['datasets']['val']["data_args"]["dataroot"] = {
                "lq": "../Restormer/Motion_Deblurring/Datasets/test/GoPro/input/",
                "gt": "../Restormer/Motion_Deblurring/Datasets/test/GoPro/target/"}
            fix += 'full'

            # root = '/mnt/lustre/zhangyi2.vendor/'
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/input/",
            # "gt": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/target/"}
            # fix = 'fhide'

            # # for split hide
            # spname = 'sp1'
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/%s/input/" % spname,
            # "gt": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/%s/target/" % spname}
            # fix = 'fhide-' + spname

            # small hide testing set
            # spname = 'spone0'
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/%s/input/" % spname,
            # "gt": root + "restormer-train/Motion_Deblurring/Datasets/test/HIDE/%s/target/" % spname}
            # fix = 'ssshide-' + spname

            # root = '/mnt/lustre/zhangyi2.vendor/'
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq": root + "restormer-train/Motion_Deblurring/Datasets/test/RealBlur_J/input/",
            # "gt": root + "restormer-train/Motion_Deblurring/Datasets/test/RealBlur_J/target/"}
            # fix = 'freal-J'
            #
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq": root + "restormer-train/Motion_Deblurring/Datasets/test/RealBlur_R/input/",
            # "gt": root + "restormer-train/Motion_Deblurring/Datasets/test/RealBlur_R/target/"}
            # fix = 'freal-R'

            # opt['model']['beta_schedule']['val']['n_timestep'] = 200
            # opt['model']['beta_schedule']['val']['linear_end'] = 1e-1
            opt['model']['beta_schedule']['val']['n_timestep'] = 50
            opt['model']['beta_schedule']['val']['linear_end'] = 4e-1
        elif 'jpg-' in opt['name']:
            root = '../data/'  # gcp
            if not os.path.exists(root + 'images'):
                root = '../../data/'
            opt['datasets']['val']["data_args"]["dataroot"] = {"root": root + "images/val",
                                                               "txt": "./imagenet_val_1k.txt"}

            # opt['datasets']['val']["data_args"]["factor"] = [5, 5]
            # fix += 'fullimage5'

            opt['datasets']['val']["data_args"]["factor"] = [10, 10]
            fix += 'fullimage10'

            # opt['datasets']['val']["datasetname"] = "PairDataset"
            # opt['datasets']['val']["data_args"]["dataroot"] = {
            # "lq":  "experiments/image_samples/imagenet10input",
            # "gt":  "experiments/image_samples/imagenet10gt"}
            # fix += 'image10'

            opt['datasets']['val']['data_args']['crop_size'] = -1
            opt['model']['beta_schedule']['val']['n_timestep'] = 200
            opt['model']['beta_schedule']['val']['linear_end'] = 1e-1
            opt['model']['beta_schedule']['val']['n_timestep'] = 50
            opt['model']['beta_schedule']['val']['linear_end'] = 4e-1

        else:
            assert 'val name not support'

        if opt['train']['ema_scheduler']['use']:
            opt['name'] += '-ema'

    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(get_timestamp(), opt['name']))
    if phase == 'val':
        experiments_root += '_s{}'.format(opt['model']['beta_schedule']['val']['n_timestep'])
        experiments_root += fix

    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    opt['distributed'] = True  # disable gpu list

    # modify batch size
    from utils.dist_utils import get_dist_info
    rank, ws = get_dist_info()
    temp_bs = opt['datasets']['train']['batch_size']
    if phase == 'train':
        assert temp_bs % ws == 0
    opt['datasets']['train']['batch_size'] = temp_bs // ws
    print('**set bs', temp_bs, opt['datasets']['train']['batch_size'], rank, ws)

    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    from utils.dist_utils import get_dist_info
    rank, _ = get_dist_info()
    if rank != 0:
        l.setLevel('ERROR')
    elif log_file is not None:
        # logger.setLevel(log_level)
        l.setLevel(level)
    l.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
