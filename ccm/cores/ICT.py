from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from .counter import CrowdCounter
from config.sys_config_back import sys_cfg
from cccv.utils.utils_tool import Timer, AverageMeter, create_summary_writer, update_crowd_model, print_summary, \
    copy_cur_env, vis_images
from . import ramps
import torch
import numpy as np
from tools.metrics.common import caluate_game
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os.path as osp
import math


class ICTTrainer:
    def __init__(self, dataloader, logger, save_dir, restore_trans=None):
        self.logger = logger
        self.train_loader, self.val_loader = dataloader
        self.model = CrowdCounter(model_name=sys_cfg.model_name)
        self.ema_model = CrowdCounter(model_name=sys_cfg.model_name)
        self.restore_trans = restore_trans

        # set the variable of TNet_ema grad is false
        for p in self.ema_model.parameters():
            p.detach_()

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
        self.update_ema_epoch = 10

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.train_record_ema_net = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
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

    def forward(self):
        for e in range(self.epoch, self.max_epoch):
            self.epoch = e
            if e > sys_cfg.num_epoch_lr_decay:
                self.scheduler_1.step(self.epoch)
            # train model
            self.timer['train time'].tic()
            self.train()
            # 更新ema模型，即Mean-Teacher
            # if self.epoch >= self.update_ema_epoch:
            #     self.model.set_bn_dtype('mix')
                # if self.epoch == self.update_ema_epoch:
                #     update_ema_variables(model=self.model,
                #                          ema_model=self.ema_model,
                #                          alpha=0,
                #                          global_step=self.global_step)
                # else:
                #     update_ema_variables(model=self.model,
                #                          ema_model=self.ema_model,
                #                          alpha=sys_cfg.ema_decay,
                #                          global_step=self.global_step)
            self.timer['train time'].toc(average=False)
            self.logger.info('the training time of {0}-th epoch: {1:.2f}s'.format(self.epoch + 1,
                                                                                  self.timer['train time'].diff))
            # validation
            if self.epoch % sys_cfg.val_freq == 0 or self.epoch > sys_cfg.val_dense_start:
                self.timer['val time'].tic()
                self.val()
                self.val_ema()
                self.timer['val time'].toc(average=False)
                self.logger.info('the val  time of {0}-th epoch: {1:.2f}s'.format(self.epoch,
                                                                                  self.timer['val time'].diff))

    def train(self):
        loss_meter = AverageMeter()
        loss_mse_meter = AverageMeter()
        loss_cs_meter = AverageMeter()

        self.model.train()
        self.ema_model.train()

        unlabel_bs = sys_cfg.batch_size - sys_cfg.batch_label_size
        assert unlabel_bs >= 0
        for i_batch, sampled_batch in enumerate(self.train_loader):

            self.timer['iter time'].tic()

            self.global_step += 1
            volumn_batch_image_label = sampled_batch['image'][unlabel_bs:]
            volumn_batch_target_label = sampled_batch['density'][unlabel_bs:]
            volumn_batch_image_label = volumn_batch_image_label.cuda()
            volumn_batch_target_label = volumn_batch_target_label.cuda()

            ema_inputs = sampled_batch['image'][:unlabel_bs]
            ema_inputs = ema_inputs.cuda()
            # ！！！拆分成两部分，防止模型不收敛；直接合并成一部分训练，训练效果较差？原因未知
            outputs = self.model.test_forward(volumn_batch_image_label)
            loss_hard = self.mse_fn(outputs.squeeze(), volumn_batch_target_label.squeeze()) / outputs.shape[0]
            if self.epoch <= self.update_ema_epoch:
                consistency_loss = torch.FloatTensor([0]).cuda()
            else:
                # ToDo this is ICT code
                # ICT mix factors
                # 3u+3u
                ict_mix_factors = np.random.beta(0.2, 0.2, size=(unlabel_bs // 2, 1, 1, 1))
                ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).cuda()
                torch

                ema_input_0 = ema_inputs[0:unlabel_bs // 2, ...]
                ema_input_1 = ema_inputs[unlabel_bs // 2:, ...]
                batch_ux_mixed = ema_input_0 * (1.0 - ict_mix_factors) + ema_input_1 * ict_mix_factors
                with torch.no_grad():
                    batch_ux_mixed_output = self.model.test_forward(batch_ux_mixed)
                    self.ema_model.test_forward(torch.cat([]))
                    ema_output_ux0 = self.ema_model.test_forward(ema_input_0)
                    ema_output_ux1 = self.ema_model.test_forward(ema_input_1)
                    batch_ema_mixed = ema_output_ux0 * (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
                consistency_loss = self.mse_fn(batch_ux_mixed_output.squeeze(), batch_ema_mixed.squeeze()) \
                                   / batch_ema_mixed.shape[0]
            # consistency_weight = get_current_consistency_weight(self.epoch)
            consistency_weight = 0.2
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
                     volumn_batch_target_label[-1].sum().data, outputs[-1].sum().data))
            if i_batch % 5 and False:
                # unlabel_image = ema_inputs[0].cpu()
                mixed_image = batch_ux_mixed[0].cpu()
                label_image = volumn_batch_image_label[0].cpu().detach()
                label_pred_map = outputs[0].cpu().detach()
                unlabel_pred_map = batch_ema_mixed[0].cpu().detach()
                gt_map = volumn_batch_target_label[0].cpu().detach()
                vis_images(name='train_info/LabelSample',
                           img=label_image,
                           pred_map=label_pred_map,
                           gt_map=gt_map,
                           writer=self.writer,
                           iter=self.global_step,
                           restore=self.restore_trans)
                vis_images(name='train_info/UnlabelSample',
                           img=mixed_image,
                           pred_map=unlabel_pred_map,
                           writer=self.writer,
                           iter=self.global_step,
                           restore=self.restore_trans)

                self.writer.add_scalar('train_info/train_loss_sum', loss_meter.avg, self.global_step)
                self.writer.add_scalar('train_info/train_loss_mse', loss_mse_meter.avg, self.global_step)
                self.writer.add_scalar('train_info/train_loss_cnt', loss_cs_meter.avg, self.global_step)
                self.writer.add_scalar('train_info/consistency_weight', consistency_weight, self.global_step)

    def val(self):
        self.model.eval()
        loss_meter = AverageMeter()
        loss_mse_meter = AverageMeter()
        # loss_cs_meter = AverageMeter()

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
                pred_map = self.model.test_forward(img)

                loss_mse = self.mse_fn(pred_map.squeeze(), gt_map.squeeze()) / gt_map.shape[0]
                loss_meter.update(loss_mse.item())
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

                gt_map = gt_map[0]
                img = img[0].cpu().detach()
                pred_map = pred_map[0]

        # vis_images(name='val_info/LabelSample',
        #            img=img,
        #            pred_map=pred_map,
        #            gt_map=gt_map,
        #            writer=self.writer,
        #            iter=self.global_step,
        #            restore=self.restore_trans)
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
                                               net_name='ICT_model',
                                               optimizer=self.optimizer_1,
                                               scheduler=self.scheduler_1,
                                               epoch=self.epoch,
                                               i_tb=self.i_tb,
                                               exp_save_dir=osp.join(self.save_dir, 'checkpoint'),
                                               scores=[mae, mse, pmae, pmse, loss, psnr_value, ssim_value],
                                               train_record=self.train_record,
                                               logger=self.logger)
        print_summary(self.train_record, self.logger)

    def val_ema(self):
        self.ema_model.eval()
        loss_meter = AverageMeter()
        loss_mse_meter = AverageMeter()

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
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                # att_map = Variable(att_map).cuda()
                pred_map = self.ema_model(img, gt_map)

                loss_sum = self.ema_model.sum_loss
                loss_meter.update(loss_sum.item())
                loss_mse, loss_cnt = self.ema_model.all_loss
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
        self.writer.add_scalar('ema_info/loss_sum', loss_meter.avg, self.epoch + 1)
        self.writer.add_scalar('ema_info/loss_mse', loss_mse_meter.avg, self.epoch + 1)

        loss = loss_meter.avg
        mae = mae_meter.avg
        mse = np.sqrt(mse_meter.avg)
        pmae = patch_mae_meter.avg
        pmse = np.sqrt(patch_mse_meter.avg)
        psnr_value = psnr_meter.avg
        ssim_value = ssim_meter.avg

        self.writer.add_scalar('ema_info/MAE', mae, self.epoch + 1)
        self.writer.add_scalar('ema_info/MSE', mse, self.epoch + 1)
        self.writer.add_scalar('ema_info/PMAE', patch_mae_meter.avg, self.epoch + 1)
        self.writer.add_scalar('ema_info/PMSE', patch_mse_meter.avg, self.epoch + 1)
        self.writer.add_scalar('ema_info/PSNR', psnr_value)
        self.writer.add_scalar('ema_info/SSIM', ssim_value)
        print('ema_net:\tmae:{0}\t mse:{1}'.format(mae, mse))

        # self.train_record = update_crowd_model(CCNet=self.model,
        #                                        net_name='model',
        #                                        optimizer=self.optimizer_1,
        #                                        scheduler=self.scheduler_1,
        #                                        epoch=self.epoch,
        #                                        i_tb=self.i_tb,
        #                                        exp_save_dir=osp.join(self.save_dir, 'checkpoint'),
        #                                        scores=[mae, mse, pmae, pmse, loss, psnr_value, ssim_value],
        #                                        train_record=self.train_record,
        #                                        logger=self.logger)
        # print_summary(self.train_record_ema_net, self.logger)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sys_cfg.consistency * ramps.sigmoid_rampup(epoch, sys_cfg.consistency_rampup)
