from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from .counter import CrowdCounter
from config.sys_config_back import sys_cfg
from cccv.utils.utils_tool import Timer, AverageMeter, create_summary_writer, update_crowd_model, print_summary, \
    copy_cur_env
from . import ramps
import torch
import numpy as np
from tools.metrics.common import caluate_game
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os.path as osp
import math


class UDA_Trainer:
    def __init__(self, args, logger):
        self.logger = logger
        self.loader = dataloader
        self.val_loader = dataloader.create_test_loader()
        self.train_loader = dataloader.create_train_loader()
        self.model = CrowdCounter(model_name=sys_cfg.model_name)
        # set the variable of TNet_ema grad is false
        if sys_cfg.freeze_frontend:
            for p in self.model.CCN.frontend.parameters():
                p.requires_grad = False

        self.optimizer_1 = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.CCN.parameters()),
            lr=sys_cfg.lr,
            weight_decay=1e-4
        )
        self.scheduler_1 = StepLR(
            optimizer=self.optimizer_1,
            step_size=sys_cfg.num_epoch_lr_decay,
            gamma=sys_cfg.lr_decay
        )

        self.mse_fn = torch.nn.MSELoss(reduction='sum').cuda()

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0
        self.global_step = 0
        self.max_epoch = sys_cfg.max_epoch
        self.save_dir = save_dir

        self.writer = create_summary_writer(self.save_dir)

        # save running file
        # if not sys_cfg.resume:
        #     copy_cur_env(sys_cfg.work_dir, osp.join(self.save_dir, 'code'), exception_list=['exp', 'img_save'])

    def train_process(self):
        for e in range(self.epoch, self.max_epoch):
            self.epoch = e
            if e > sys_cfg.num_epoch_lr_decay:
                self.scheduler_1.step(self.epoch)
            # train model
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            self.logger.info('the training time of {0}-th epoch: {1:.2f}s'.format(self.epoch,
                                                                                  self.timer['train time'].diff))
            # validation
            if self.epoch % sys_cfg.val_freq == 0 or self.epoch > sys_cfg.val_dense_start:
                self.timer['val time'].tic()
                self.val()
                self.timer['val time'].toc(average=False)
                self.logger.info('the val  time of {0}-th epoch: {1:.2f}s'.format(self.epoch,
                                                                                  self.timer['val time'].diff))

    def train(self):
        loss_meter = AverageMeter()
        loss_mse_meter = AverageMeter()
        loss_cs_meter = AverageMeter()
        self.model.train()

        unlabel_bs = sys_cfg.batch_size - sys_cfg.batch_label_size
        assert unlabel_bs >= 0
        for i_batch, sampled_batch in enumerate(self.train_loader):

            self.timer['iter time'].tic()
            self.global_step += 1
            volumn_batch_image_label = sampled_batch['image'][unlabel_bs:]
            volumn_batch_target_label = sampled_batch['density'][unlabel_bs:]

            volumn_batch_image_label = volumn_batch_image_label.cuda()
            volumn_batch_target_label = volumn_batch_target_label.cuda()

            volumn_batch_aug_image_unlabel = sampled_batch['aug_image'][:unlabel_bs]
            volumn_batch_aug_image_unlabel = volumn_batch_aug_image_unlabel.cuda()

            unlabel_inputs = sampled_batch['image'][:unlabel_bs]
            unlabel_inputs = unlabel_inputs.cuda()

            output_main = self.model.test_forward(
                torch.cat([volumn_batch_aug_image_unlabel, volumn_batch_image_label], dim=0))

            # output_label = self.model.test_forward(volumn_batch_image_label)
            # output_unlabel = self.model.test_forward(volumn_batch_aug_image_unlabel)

            with torch.no_grad():
                output_unlabel_con = self.model.test_forward(unlabel_inputs)

            loss_hard = self.mse_fn(output_main[unlabel_bs:].squeeze(),
                                    volumn_batch_target_label.squeeze()) / unlabel_bs

            # consistency_weight = cal_soft_weight(self.epoch, init_ep=21)
            consistency_weight = 0.2
            consistency_loss = self.mse_fn(output_main[:unlabel_bs].squeeze(), output_unlabel_con.squeeze()) / \
                               unlabel_bs

            loss = loss_hard + consistency_weight * consistency_loss

            loss_meter.update(loss.item())
            loss_mse_meter.update(loss_hard.item())
            loss_cs_meter.update(consistency_weight * consistency_loss.item())

            self.optimizer_1.zero_grad()
            loss.backward()
            self.optimizer_1.step()

            if i_batch % sys_cfg.log_freq == 0:
                self.i_tb += 1
                self.timer['iter time'].toc(average=False)
                self.logger.info(
                    '[ep %d][it %d][ loss %.4f loss_mse %.4f,loss_cnt %.4f][lr_1 %.4f ][%.2fs]'
                    '[cnt: gt: %.1f pred: %.2f]' % \
                    (self.epoch + 1, i_batch + 1, loss.item(), loss_hard.item(),
                     consistency_weight * consistency_loss.item(),
                     self.optimizer_1.param_groups[0]['lr'] * 10000, self.timer['iter time'].diff,
                     volumn_batch_target_label[-1].sum().data, output_main[-1].sum().data))

            self.writer.add_scalar('train_info/train_loss_sum', loss_meter.avg, self.global_step)
            self.writer.add_scalar('train_info/train_loss_mse', loss_mse_meter.avg, self.global_step)
            self.writer.add_scalar('train_info/train_loss_cnt', loss_cs_meter.avg, self.global_step)
            self.writer.add_scalar('train_info/consistency_weight', consistency_weight, self.global_step)

    def val(self):
        self.model.eval()
        loss_meter = AverageMeter()
        loss_mse_meter = AverageMeter()
        loss_cs_meter = AverageMeter()

        mae_meter = AverageMeter()
        mse_meter = AverageMeter()

        patch_mae_meter = AverageMeter()
        patch_mse_meter = AverageMeter()

        ssim_meter = AverageMeter()
        psnr_meter = AverageMeter()

        for i, data in enumerate(self.val_loader):
            with torch.no_grad():
                self.timer['iter time'].tic()
                img = data['image']
                gt_map = data['density']
                gt_count = data['gt_count']
                #########################
                img = img.cuda()
                gt_map = gt_map.cuda()
                # att_map = Variable(att_map).cuda()
                pred_map = self.model(img, gt_map)

                loss_sum = self.model.sum_loss
                loss_meter.update(loss_sum.item())
                loss_mse, loss_cnt = self.model.all_loss
                loss_mse_meter.update(loss_mse.item())

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img])
                    gt_count = gt_count.item()
                    # gt_count = np.sum(gt_map[i_img])

                    pmae, pmse = caluate_game(np.squeeze(pred_map[i_img]), np.squeeze(gt_map[i_img]),
                                              sys_cfg.patch_level)
                    patch_mae_meter.update(pmae)
                    patch_mse_meter.update(pmse)

                    pred_normalized_map = np.squeeze(pred_map[i_img] / np.max(pred_map[i_img] + 1e-20))
                    gt_normalized_map = np.squeeze(gt_map[i_img] / np.max(gt_map[i_img] + 1e-20))
                    s = ssim(gt_normalized_map, pred_normalized_map)
                    p = psnr(gt_normalized_map, pred_normalized_map)

                    ssim_meter.update(s)
                    psnr_meter.update(p)

                    mae_meter.update(abs(gt_count - pred_cnt))
                    mse_meter.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
        self.writer.add_scalar('val_info/loss_sum', loss_meter.avg, self.epoch + 1)
        self.writer.add_scalar('val_info/loss_mse', loss_mse_meter.avg, self.epoch + 1)

        loss = loss_meter.avg
        mae = mae_meter.avg
        mse = np.sqrt(mse_meter.avg)
        pmae = patch_mae_meter.avg
        pmse = np.sqrt(patch_mse_meter.avg)
        psnr_value = psnr_meter.avg
        ssim_value = ssim_meter.avg

        self.writer.add_scalar('val/MAE', mae, self.epoch + 1)
        self.writer.add_scalar('val/MSE', mse, self.epoch + 1)
        self.writer.add_scalar('val/PMAE', patch_mae_meter.avg, self.epoch + 1)
        self.writer.add_scalar('val/PMSE', patch_mse_meter.avg, self.epoch + 1)
        self.writer.add_scalar('val/PSNR', psnr_value)
        self.writer.add_scalar('val/SSIM', ssim_value)

        self.train_record = update_crowd_model(net=self.model,
                                               net_name='model',
                                               optimizer=self.optimizer_1,
                                               scheduler=self.scheduler_1,
                                               epoch=self.epoch,
                                               i_tb=self.i_tb,
                                               exp_save_dir=osp.join(self.save_dir, 'checkpoint'),
                                               scores=[mae, mse, pmae, pmse, loss, psnr_value, ssim_value],
                                               train_record=self.train_record,
                                               logger=self.logger)
        print_summary(self.train_record, self.logger)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sys_cfg.consistency * ramps.sigmoid_rampup(epoch, sys_cfg.consistency_rampup)


def cal_soft_weight(epoch, init_ep=0, end_ep=1000, init_w=0.0, end_w=1.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    # print('Consistency weight: %f'%weight_cl)
    return weight_cl
