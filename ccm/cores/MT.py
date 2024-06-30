from torch import optim
from torch.optim.lr_scheduler import StepLR
from cccv.utils.utils_tool import Timer, AverageMeter
import torch
import numpy as np
from tools.metrics.common import caluate_game
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os.path as osp
import torch.nn as nn
import os

from ccm.dataset.dataloader import make_dataloader
from torch.nn import functional as F
import importlib


def fixed_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def fixed_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.eval()


class MT_Trainer(object):
    """
     MeanTeacher algorithm (https://arxiv.org/abs/1703.01780).
    """

    def __init__(self, args, logger):
        self.logger = logger
        self.args = args

        self.logger.info('Performing MT')
        self.logger.info('[1]-create dataset and dataloader')
        self.train_loader, self.val_loader, self.test_loader = make_dataloader(args)

        # create the model
        net_module = getattr(importlib.import_module('.CCNet.{}'.format(args.network), 'ccm.models'),
                             '{0}'.format(args.network))
        self.model = net_module(args)
        self.ema_model = net_module(args)

        # set the variable of TNet_ema grad is false
        for p in self.ema_model.parameters():
            p.detach_()

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model.to(self._device), device_ids=args.gpus)
                self.ema_model = nn.DataParallel(self.ema_model.to(self._device), device_ids=args.gpus)
            else:
                self.model = self.model.to(self._device)
                self.ema_model = self.ema_model.to(self._device)
        else:
            raise ValueError('no gpu device available')

        # define the optimizer and scheduler
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler_1 = StepLR(
            optimizer=self.optimizer,
            step_size=self.args.num_epoch_lr_decay,
            gamma=self.args.lr_decay
        )

        # define the loss function
        self.mse_fn = torch.nn.MSELoss(reduction='sum').cuda()
        self.mae_fn = torch.nn.L1Loss(reduction='sum').cuda()
        self.bce_loss_fn = torch.nn.BCELoss().cuda()

        # define the running log
        self.ckpt_save_dir = osp.join(args.exp_dir, 'ckpt')
        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        self.train_record = {'best_mae': 1e20, 'best_rmse': 1e20, 'best_model_name': ''}
        self.train_record_ema_net = {'best_mae': 1e20, 'best_rmse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.total_epoch = args.total_epoch
        self.start_epoch = 0
        self.global_step = 0
        self.warmup_epoch = 30

    def train_process(self):
        for e in range(self.start_epoch, self.total_epoch):
            self.epoch = e
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(e, self.total_epoch) + '-' * 5)
            # train model
            # if e > sys_cfg.num_epoch_lr_decay:

            self.timer['train time'].tic()
            self.train_epoch()
            self.timer['train time'].toc(average=False)
            self.logger.info('The training time of {0}-th epoch: {1:.2f}s'.format(self.epoch,
                                                                                  self.timer['train time'].diff))
            torch.cuda.empty_cache()
            self.scheduler_1.step()
            if self.epoch == self.warmup_epoch:
                self.logger.info('warm up finished, start to update the ema model')
                initialize_ema_model(model=self.model, ema_model=self.ema_model)

            # validation
            if self.epoch % self.args.val_freq == 0 and self.epoch > self.args.val_start:
                self.timer['val time'].tic()
                self.val(use_ema_model=True)  # teacher or student for validation
                self.timer['val time'].toc(average=False)
                self.logger.info('epoch: [{}/{}], val time : {:.2f}s'.format(self.epoch,
                                                                             self.total_epoch,
                                                                             self.timer['val time'].diff))

    def train_epoch(self):
        mae_meter = AverageMeter()
        rmse_meter = AverageMeter()
        loss_meter = AverageMeter()

        self.model.train()
        self.ema_model.train()

        unlabeled_bs = self.args.batch_size - self.args.label_batch_size
        assert unlabeled_bs >= 0

        for i, db in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            self.global_step += 1

            img = db['img']
            # aug_img = db['aug_img']
            gt_den = db['den']

            if self.epoch <= self.warmup_epoch:
                img_labeled = img[unlabeled_bs:].to(self._device)
                den_labeled = gt_den[unlabeled_bs:].to(self._device)
                gt_cnt = den_labeled.view(den_labeled.shape[0], -1).sum(1).cpu()
                outputs = self.model(img_labeled)

                sup_loss = self.args.lambda_sup * self.mse_fn(outputs, den_labeled) / outputs.shape[0]
                unsup_loss = torch.FloatTensor([0]).to(self._device)
            else:
                gt_cnt = gt_den.view(gt_den.shape[0], -1).sum(1)
                gt_den = gt_den.to(self._device)

                # if labeled and unlabeled data are not in the same batch, we need to split the batch.
                img = img.to(self._device)
                # aug_img = aug_img.to(self._device)
                if self.args.enable_sbn:  # default: False
                    noise = (torch.randn_like(img[:unlabeled_bs]) * 0.1).clamp(-0.2, 0.2)
                    unlabeled_img = img[:unlabeled_bs] + noise

                    stu_unlabeled_outputs = self.model(unlabeled_img)
                    stu_label_outputs = self.model(img[unlabeled_bs:])

                    stu_outputs = torch.cat([stu_unlabeled_outputs, stu_label_outputs], dim=0)
                else:
                    noise = (torch.randn_like(img[:unlabeled_bs]) * 0.2).clamp(-0.2, 0.2)
                    unlabeled_img = img[:unlabeled_bs] + noise

                    stu_outputs = self.model(torch.cat([unlabeled_img, img[unlabeled_bs:]], dim=0))

                with torch.no_grad():
                    tea_unlable_outputs = self.ema_model(img[:unlabeled_bs])
                N_l = stu_outputs[unlabeled_bs:].shape[0]
                sup_loss = self.args.lambda_sup * self.mse_fn(stu_outputs[unlabeled_bs:],
                                                              gt_den[unlabeled_bs:]) / N_l

                N_u = tea_unlable_outputs.shape[0]
                unsup_loss = self.args.lambda_unsup * self.mse_fn(stu_outputs[:unlabeled_bs],
                                                                  tea_unlable_outputs.detach()) / N_u

                outputs = stu_outputs

            # consistency_weight = get_current_consistency_weight(self.epoch)
            loss = sup_loss + unsup_loss
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epoch > self.warmup_epoch:
                # update ema model at each iteration
                update_ema_model_with_bn(model=self.model,
                                         ema_model=self.ema_model,
                                         alpha=self.args.ema_decay)

            # calculate mae and rmse of training data
            N = outputs.shape[0]
            pred_cnt = torch.sum(outputs.view(N, -1), dim=-1).detach().cpu().numpy()
            res = pred_cnt - gt_cnt.numpy()
            mae_meter.update(np.mean(abs(res)))
            rmse_meter.update(np.mean(res * res))

            if i % self.args.log_freq == 0:
                self.timer['iter time'].toc(average=False)
                self.logger.info(
                    'E-{} iter-{}, Loss[total={:.4f}, sup={:.4f}, unsup={:.4f}], gt={:.1f} pred={:.1f} '
                    'lr={:.4f} cost={:.1f} sec'.format(self.epoch, i,
                                                       loss.item(), sup_loss.item(), unsup_loss.item(),
                                                       gt_cnt[0], pred_cnt[0],
                                                       self.optimizer.param_groups[0]['lr'] * 10000,
                                                       self.timer['train time'].diff))

    def val(self, use_ema_model=False):
        if use_ema_model:
            self.ema_model.eval()
        else:
            self.model.eval()

        epoch_res = []
        for i, data in enumerate(self.test_loader):
            self.timer['iter time'].tic()
            img = data['img']
            gt_den = data['den']
            gt_cnt = gt_den.view(gt_den.shape[0], -1).sum(1)

            img = img.to(self._device)
            gt_den = gt_den.to(self._device)

            with torch.no_grad():
                if use_ema_model:
                    outputs = self.ema_model(img)
                else:
                    outputs = self.model(img)
                res = gt_cnt[0].item() - torch.sum(outputs).item()
            epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mae = np.mean(np.abs(epoch_res))
        rmse = np.sqrt(np.mean(np.square(epoch_res)))

        self.timer['val time'].toc(average=False)
        self.logger.info('Epoch {}, test_mae={:.2f}, test_rmse={:.2f}, '
                         'Cost={:.1f} sec'.format(self.epoch,
                                                  mae,
                                                  rmse,
                                                  self.timer['val time'].diff))
        if True:
            if (2.0 * rmse + mae) < (2.0 * self.train_record['best_rmse'] + self.train_record['best_mae']):
                self.train_record['best_rmse'] = rmse
                self.train_record['best_mae'] = mae

                self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(rmse,
                                                                                         mae,
                                                                                         self.epoch))
                if torch.cuda.device_count() > 1:
                    to_save_dict = self.model.module.state_dict()
                else:
                    to_save_dict = self.model.state_dict()
                if use_ema_model:
                    save_name = 'E-{}_ema_tea_MAE-{:.2f}_RMSE-{:.2f}.pth'.format(self.epoch, mae, rmse)
                else:  # save student model
                    save_name = 'E-{}_stu_MAE-{:.2f}_RMSE-{:.2f}.pth'.format(self.epoch, mae, rmse)
                # save_name = 'E-{}_stu_MAE-{:.2f}_RMSE-{:.2f}.pth'.format(self.epoch, mae, rmse)
                torch.save(to_save_dict, os.path.join(self.ckpt_save_dir, save_name))

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
                img = img.cuda()
                gt_map = gt_map.cuda()
                # att_map = Variable(att_map).cuda()

                # if sys_cfg.use_seg_branch:
                #     pred_map, _ = self.ema_model.test_forward(img)
                # else:
                if sys_cfg.use_seg_branch:
                    pred_map, _ = self.ema_model.test_forward(img)
                else:
                    pred_map = self.ema_model.test_forward(img)

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
        self.logger.info('ema_net:\tmae:{0}\t mse:{1}'.format(mae, mse))

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

    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)


# def update_ema_variables(model, ema_model, alpha):
#     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#         ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_model(model, ema_model, alpha):
    """
    description: update the ema model
    :param model:
    :param ema_model:
    :param alpha:
    :return:
    """
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def initialize_ema_model(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.copy_(param.data)  # initialize
        ema_param.requires_grad = False  # not update by gradient


def update_ema_model_with_bn(model, ema_model, alpha=0.999):
    """
    description: update the ema model with bn statistics
    :param model:
    :param ema_model:
    :param alpha:
    :return:
    """
    params_dict = model.state_dict()
    ema_params_dict = ema_model.state_dict()
    for (k_ema, v_ema), (k_main, v_main) in zip(ema_params_dict.items(), params_dict.items()):
        assert k_main == k_ema, "state_dict names are different!"
        assert v_main.shape == v_ema.shape, "state_dict shapes are different!"

        if 'num_batches_tracked' in k_ema:
            v_ema.copy_(v_main)
        else:
            v_ema.copy_(v_ema * alpha + (1. - alpha) * v_main)


# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return sys_cfg.consistency * ramps.sigmoid_rampup(epoch, sys_cfg.consistency_rampup)


def generate_mask(target, win_size, t1):
    target = build_block(target, size=win_size)
    target[target > t1] = 0.5
    target[target <= t1] = 1

    return F.upsample_nearest(target, scale_factor=win_size)


def multi_layer_loss(input, target, win_size=4):
    input = build_block(input, size=win_size)
    target = build_block(target, size=win_size)
    loss = torch.mean((input - target) ** 2) / target.shape[0]
    return loss


def adaptive_density_loss(input, target, win_size=4):
    squared_loss = (input - target) ** 2
    input = build_block(input, size=win_size)
    target = build_block(target, size=win_size)
    squared_loss = build_block(squared_loss, size=win_size)
    density_factor = (input - target) ** 2
    # print(density_factor.shape)
    # print(squared_loss.shape)

    loss = torch.sum(squared_loss * torch.exp(-density_factor)) / input.shape[0]
    return loss


def build_block(x, size=4):
    x_shape = x.shape
    if list(x_shape).__len__() == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    return F.avg_pool2d(x, size) * size * size
