# -*- coding: utf-8 -*-
__author__ = 'Watson'

import argparse
import torch
import os
from config.system_config import sys_cfg
from easydict import EasyDict


def attack_parse_args():
    parser = argparse.ArgumentParser(description='SSCC framework')
    parser.add_argument('--network', default='CSRNet', type=str, help='the name of trained model')
    parser.add_argument('--dataset', default='SHHA_eval', type=str, help='the name of trained model')
    parser.add_argument('--model_path', default='./', type=str, help='the name of trained model')
    parser.add_argument('--gpus', default='0', help='assign device')
    parser.add_argument('--eval_mode', default='val', type=str, help='the eval mode of dataset')
    parser.add_argument('--attack_method', default='none', type=str, help='attack method')
    parser.add_argument('--attack_eps', default=0.007, type=float, help='the attacking budget')
    parser.add_argument('--attack_alpha', default=0.007, type=float, help='gpu_id')
    parser.add_argument('--attack_steps', default=0.1, type=int, help='the iteration of attacking')
    parser.add_argument('--num_workers', default=1, type=int, help='the number of workers')
    parser.add_argument('--perturb_method', default='none', type=str, help='perturb method')
    args = parser.parse_args()
    return args


def main_work():
    from ccm.cores.attacker import Attacker
    custom_args = sys_cfg.copy()
    args = attack_parse_args()
    custom_args.update(vars(args))
    custom_args = EasyDict(custom_args)

    os.environ['CUDA_VISIBLE_DEVICES'] = custom_args.gpus.strip()
    print('Total gpus is {0}, using {1} gpus'.format(torch.cuda.device_count(), torch.cuda.device_count()))

    my_eval = Attacker(custom_args, None)
    my_eval.eval_model()


if __name__ == '__main__':
    main_work()
