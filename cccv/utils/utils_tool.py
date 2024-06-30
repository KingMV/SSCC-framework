import time
import os
import shutil
import sys
import torch
import os.path as osp
import subprocess
from collections import defaultdict
import torchvision
import cccv
import ccm
# from config.sys_config_back import sys_cfg
import numpy as np
from PIL import Image
import cv2

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


def logger(exp_path, exp_name, work_dir, exception, resume=False):
    from tensorboardX import SummaryWriter

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path + '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'

    cfg_file = open('./config.py', "r")
    cfg_lines = cfg_file.readlines()

    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path + '/' + exp_name + '/code', exception)

    return writer, log_file


def create_summary_writer(save_dir):
    from tensorboardX import SummaryWriter
    path = osp.join(save_dir, 'logs')
    if not os.path.exists(path):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(path)

    return writer


def copy_cur_env(work_dir, dst_dir, exception_list):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir, filename)
        dst_file = os.path.join(dst_dir, filename)

        copy_flag = True
        for exception in exception_list:
            if exception in filename:
                copy_flag = False

        if os.path.isdir(file) and copy_flag:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file, dst_file)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        # from mmcv.utils.parrots_wrapper import CUDA_HOME
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    env_info['TorchVision'] = torchvision.__version__

    env_info['OpenCV'] = cv2.__version__

    env_info['CCCV'] = cccv.__version__
    env_info['CCM'] = ccm.__version__
    # try:
    #     from mmcv.ops import get_compiler_version, get_compiling_cuda_version
    #     env_info['MMCV Compiler'] = get_compiler_version()
    #     env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()
    # except ImportError:
    #     env_info['MMCV Compiler'] = 'n/a'
    #     env_info['MMCV CUDA Compiler'] = 'n/a'

    return env_info


def print_environment(logger):
    # collect the run environment
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)


def print_configure(logger, sys_cfg):
    cfg_str = '\n'.join([f'{k}: {v}' for k, v in sys_cfg.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('System config info:\n' + dash_line + '\n' + cfg_str + '\n' + dash_line)


def print_dataset_configure(logger):
    from config.csrnet import cfg
    cfg_str = '\n'.join([f'{k}: {v}' for k, v in cfg.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Dataset config info:\n' + dash_line + '\n' + cfg_str + '\n' + dash_line)


def update_crowd_model(net, net_name, optimizer, scheduler, epoch, i_tb, exp_save_dir, scores, train_record,
                       logger=None):
    if len(scores) == 7:
        mae, mse, pmae, pmse, loss, psnr, ssim = scores
        snapshot_name = '%s_ep_%d_mae_%.2f_mse_%.2f_pmae_%.2f_pmse_%.2f_p_%.2f_s_%.2f' % \
                        (net_name, epoch, mae, mse, pmae, pmse, psnr, ssim)
    elif len(scores) == 2:
        mae, mse = scores
        snapshot_name = '%s_ep_%d_mae_%.2f_mse_%.2f' % (net_name, epoch + 1, mae, mse)

    logger.info(snapshot_name)

    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir, exist_ok=True)

    if (2.0 * mse + mae) < (2.0 * train_record['best_mse'] + train_record['best_mae']):
        # if mae < train_record['best_mae'] or mse < train_record['best_mse']:
        train_record['best_model_name'] = snapshot_name
        if logger is not None:
            if len(scores) == 7:
                logger.info('[mae:%.2f mse:%.2f pmae:%.2f pmse:%.2f], [val_loss:%.4f] [psnr %.4f] [ssim %.4f]\n'
                            % (mae, mse, pmae, pmse, loss, psnr, ssim))
            elif len(scores) == 2:
                logger.info('[mae:%.2f mse:%.2f]\n' % (mae, mse))

        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_save_dir, snapshot_name + '.pth'))
        train_record['best_model_path'] = os.path.join(exp_save_dir, snapshot_name + '.pth')
        if mae < train_record['best_mae']:
            train_record['best_mae'] = mae
        if mse < train_record['best_mse']:
            train_record['best_mse'] = mse
    latest_state = {'train_record': train_record, 'CCNet': net.state_dict(), 'optimizer': optimizer.state_dict(), \
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'i_tb': i_tb, 'exp_dir': exp_save_dir}

    torch.save(latest_state, os.path.join(exp_save_dir, 'latest_state.pth'))

    return train_record


def update_tsg_model(net, net_name, optimizer, scheduler, epoch, i_tb, exp_save_dir, scores, train_record,
                     logger=None):
    loss = scores[0]
    snapshot_name = '%s_ep_%d_val_loss%.2f' % (net_name, epoch + 1, loss)
    logger.info(snapshot_name)

    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir, exist_ok=True)

    if loss < train_record['best_loss']:
        train_record['best_model_name'] = snapshot_name
        if logger is not None:
            logger.info('[val_loss:%.4f] \n' % (loss))
        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_save_dir, snapshot_name + '.pth'))
        train_record['best_model_path'] = os.path.join(exp_save_dir, snapshot_name + '.pth')

    if loss < train_record['best_loss']:
        train_record['best_loss'] = loss
    latest_state = {'train_record': train_record, 'CCNet': net.state_dict(), 'optimizer': optimizer.state_dict(), \
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'i_tb': i_tb, 'exp_dir': exp_save_dir}
    torch.save(latest_state, os.path.join(exp_save_dir, 'latest_state.pth'))
    return train_record


def print_summary(train_record, logger):
    dash_line = '-' * 60 + '\n'
    best_model_info = '[best] [model: %s] , [mae %.2f], [mse %.2f]' % (train_record['best_model_name'], \
                                                                       train_record['best_mae'], \
                                                                       train_record['best_mse'])
    logger.info('Best Model Info:\n' + dash_line + best_model_info + '\n' + dash_line)


def debug_density(img, gt_map, s_map, t_map, win_name, save_dir):
    if isinstance(gt_map, torch.Tensor):
        gt_map = gt_map.numpy()
    if isinstance(gt_map, torch.Tensor):
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        # img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if isinstance(s_map, torch.Tensor):
        s_map = s_map.numpy()
    if isinstance(t_map, torch.Tensor):
        t_map = t_map.numpy()

    s_cnt = np.sum(s_map)
    g_cnt = np.sum(gt_map)
    t_cnt = np.sum(t_map)

    normalize_base = np.max(gt_map)

    gt_map = (gt_map - gt_map.min()) / (normalize_base + 1e-20)
    s_map = (s_map - s_map.min()) / (normalize_base + 1e-20)
    t_map = (t_map - t_map.min()) / (normalize_base + 1e-20)
    gt_map = np.squeeze(gt_map)
    s_map = np.squeeze(s_map)
    t_map = np.squeeze(t_map)
    img = np.squeeze(img)

    gt_map = cv2.applyColorMap(np.uint8(255 * gt_map), cv2.COLORMAP_JET)
    gt_map = cv2.resize(gt_map, (img.shape[2], img.shape[1]))

    gt_map = cv2.putText(gt_map, '{0}:{1}'.format('GT', int(g_cnt)), (10, gt_map.shape[0] - 50),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1.2,
                         (255, 255, 255), 2)

    s_map = cv2.applyColorMap(np.uint8(255 * s_map), cv2.COLORMAP_JET)
    s_map = cv2.resize(s_map, (img.shape[2], img.shape[1]))
    s_map = cv2.putText(s_map, '{0}:{1}'.format('EST', int(s_cnt)), (10, s_map.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255), 2)

    t_map = cv2.applyColorMap(np.uint8(255 * t_map), cv2.COLORMAP_JET)
    t_map = cv2.resize(t_map, (img.shape[2], img.shape[1]))

    t_map = cv2.putText(t_map, '{0}:{1}'.format('EST', int(t_cnt)), (10, t_map.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255), 2)

    # cv2.imshow('img', img)
    if save_dir is not None:
        cv2.imwrite(osp.join(save_dir, '{0}_s_map.jpg'.format(win_name)), s_map)
        cv2.imwrite(osp.join(save_dir, '{0}_t_map.jpg'.format(win_name)), t_map)
        cv2.imwrite(osp.join(save_dir, '{0}_gt_map.jpg'.format(win_name)), gt_map)
    # cv2.imshow('{0}_s_map'.format(win_name), s_map)
    # cv2.imshow('{0}_t_map'.format(win_name), t_map)
    # cv2.imshow('{0}_gt_map'.format(win_name), gt_map)

    cv2.waitKey(10)


def vis_features(feature):
    data = feature[0]
    dh, dw = data.shape[-2:]
    f_map = cv2.applyColorMap(np.uint8(255 * data), cv2.COLORMAP_JET)

    f_map = cv2.resize(f_map, (dw, dh))
    cv2.imshow('f_map', f_map)
    cv2.waitKey(5)


def vis_images(name, writer, restore, iter, img, pred_map=None, gt_map=None):
    # print(img.shape)
    pil_input = restore(img)
    pil_input = pil_input.convert('RGB')
    tensor_input = standard_transforms.ToTensor()(pil_input)
    x = []
    if pred_map is not None and gt_map is not None:
        pred_map, gt_map = vis_density_2(pred_map, gt_map, img)
        pred_map = standard_transforms.ToTensor()(pred_map)
        gt_map = standard_transforms.ToTensor()(gt_map)

        x.extend([tensor_input, pred_map, gt_map])
        x = torch.stack(x, 0)
        x = vutils.make_grid(x, nrow=3, padding=5)
        x = (x.numpy() * 255).astype(np.uint8)
        writer.add_image(name, x, iter)
    elif pred_map is not None:
        pred_map = vis_density_1(pred_map, name='EST', img=img)
        x.extend([tensor_input, standard_transforms.ToTensor()(pred_map)])
        x = torch.stack(x, 0)
        x = vutils.make_grid(x, nrow=2, padding=5)
        x = (x.numpy() * 255).astype(np.uint8)
        writer.add_image(name, x, iter)
    else:
        # print('numpy shape {0}'.format(tensor_input.shape))
        writer.add_image(name, tensor_input, iter)


def normalize_data(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input) + 1e-20)


def vis_density_1(map, name, img=None):
    if isinstance(map, torch.Tensor):
        map = map.numpy()
    count = np.sum(map)
    map = normalize_data(map)
    map = np.squeeze(map)
    map = cv2.applyColorMap(np.uint8(255 * map), cv2.COLORMAP_JET)
    map = cv2.resize(map, (img.shape[2], img.shape[1]))
    map = cv2.putText(map, '{0}:{1}'.format(name, int(count)), (10, map.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                      (255, 255, 255), 2)

    map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
    map = Image.fromarray(map)
    map.convert('RGB')
    return map


def vis_map_and_image(data, image, name):
    if len(data.shape) > 2:
        data = np.transpose(data, (1, 2, 0))
    data = (data - data.min()) / (data.max() - data.min())
    data = np.squeeze(data)
    data = cv2.applyColorMap(np.uint8(255 * data), cv2.COLORMAP_JET)
    data = cv2.resize(data, (image.shape[2], image.shape[1]))

    image = np.transpose(image, (1, 2, 0))
    cv2.imshow('{0}'.format(name), data)
    cv2.imshow('{0}_img'.format(name), image)


def vis_mask(data, name):
    if len(data.shape) > 2:
        data = np.transpose(data, (1, 2, 0))
    data = (data - data.min()) / (data.max() - data.min())
    cv2.imshow('{0}'.format(name), data)


def vis_density_2(map1, map2, img=None):
    if isinstance(map1, torch.Tensor):
        map1 = map1.numpy()
    if isinstance(map2, torch.Tensor):
        map2 = map2.numpy()

    count1 = np.sum(map1)
    count2 = np.sum(map2)

    # map1 = normalize_data(map1)
    map1 = (map1 - map1.min()) / (map2.max() + 1e-20)
    map1 = np.squeeze(map1)
    map2 = (map2 - map2.min()) / (map2.max() + 1e-20)
    map2 = np.squeeze(map2)

    map1 = cv2.applyColorMap(np.uint8(255 * map1), cv2.COLORMAP_JET)
    map1 = cv2.resize(map1, (img.shape[2], img.shape[1]))

    map2 = cv2.applyColorMap(np.uint8(255 * map2), cv2.COLORMAP_JET)
    map2 = cv2.resize(map2, (img.shape[2], img.shape[1]))

    map1 = cv2.putText(map1, '{0}:{1}'.format('EST', int(count1)), (10, map1.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2,
                       (255, 255, 255), 2)

    map2 = cv2.putText(map2, '{0}:{1}'.format('GT', int(count2)), (10, map2.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2,
                       (255, 255, 255), 2)
    # print(map.shape)
    map1 = cv2.cvtColor(map1, cv2.COLOR_BGR2RGB)
    map2 = cv2.cvtColor(map2, cv2.COLOR_BGR2RGB)

    map1 = Image.fromarray(map1)
    map1.convert('RGB')

    map2 = Image.fromarray(map2)
    map2.convert('RGB')

    return map1, map2
