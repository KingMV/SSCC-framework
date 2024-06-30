# -*- coding: utf-8 -*-

import os
import os.path as osp
import random
import torch
from easydict import EasyDict as edict

from cccv.utils.logger import get_logger
from cccv.utils.utils_tool import print_environment, print_configure
from utils.argparser import parse_args

from config.system_config import sys_cfg
from utils.file_process import load_json, store_current_training_args
from ccm.trainer import run_trainer


def get_args():
    # now = time.strftime("%m%d%H%M", time.localtime())
    args = parse_args()  # cmd config
    param = load_json(args.config)  # json cfg

    custom_args = sys_cfg.copy()  # system config
    custom_args.update(param)
    custom_args.update(vars(args))
    custom_args = edict(custom_args)

    exp_name = '{}_{}_{}_{}'.format(custom_args.train_id,
                                    custom_args.method,
                                    custom_args.dataset,
                                    custom_args.note)
    
    custom_args.exp_dir = osp.join(custom_args.exp_root, exp_name)
    custom_args.log_dir = osp.join(custom_args.exp_dir, 'log')
    return custom_args


def _set_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def _set_device(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus.strip()


def main_work():
    args = get_args()  # parse the args from cmd and json file
    store_current_training_args(args.log_dir, args.train_id + '_env.txt', args)  # save the args to the log dir

    _set_random(args.seed)  # set seed
    _set_device(args.gpus)  # set gpu

    logger = get_logger(name='SSCC', log_file=osp.join(args.log_dir, '{0}.log'.format(args.train_id + '_train')))
    # print the env setting and configure  setting
    print_environment(logger)
    print_configure(logger, args)

    # copy the trainer file to the log dir

    # run the trainer
    run_trainer(args, logger)


if __name__ == '__main__':
    main_work()
