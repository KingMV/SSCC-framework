from torch import optim
from torch.optim.lr_scheduler import StepLR
from cccv.utils.utils_tool import Timer, AverageMeter
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import os

from ccm.dataset.dataloader import make_dataloader
from torch.nn import functional as F
import importlib
from torch.autograd import Variable


class Bn_Controller:
    """
    Batch Norm controller
    """

    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}


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


def disable_bn_tracking(m):
    if hasattr(m, 'track_running_stats'):
        m.track_running_stats = False


def enable_bn_tracking(m):
    if hasattr(m, 'track_running_stats'):
        m.track_running_stats = True


class VAT_Trainer(object):
    """
    Virtual Adversarial Training algorithm (https://arxiv.org/abs/1704.03976).
    """

    def __init__(self, args, logger):
        self.logger = logger
        self.args = args

        self.logger.info('Performing VAT SSL method')
        self.logger.info('create dataset and dataloader')
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
            weight_decay=5e-4  # it is important for ssls
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
                self.val(use_ema_model=False)
                self.timer['val time'].toc(average=False)
                self.logger.info('epoch: [{}/{}], val time : {:.2f}s'.format(self.epoch,
                                                                             self.total_epoch,
                                                                             self.timer['val time'].diff))

    def vat_loss(self, model, x_u, y_u, xi=1e-6, eps=6, num_iters=1):
        """

        :param model:
        :param x_u:
        :param y_u:
        :param xi: default 1e-6
        :param eps:
        :param num_iters:
        :return:
        """

        d = torch.Tensor(x_u.size()).normal_()

        for i in range(num_iters):
            # d = xi * get_normalized_vector(d).requires_grad_()
            d = xi * self._l2_normalize(d)
            d = Variable(d.cuda(), requires_grad=True)

            y_hat = model(x_u + d)
            adv_distance = self.mse_fn(y_hat, y_u.detach()) / y_hat.size(0)
            adv_distance.backward()

            d = d.grad.data.clone().cpu()
            model.zero_grad()

        d = self._l2_normalize(d)
        d = Variable(d.cuda())
        r_adv = d * eps

        y_hat = model(x_u + r_adv.detach())
        adv_distance = self.mse_fn(y_hat, y_u.detach()) / y_hat.size(0)

        return adv_distance

    def _l2_normalize(self, d):
        # TODO: put this to cuda with torch
        d = d.numpy()
        if len(d.shape) == 4:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        elif len(d.shape) == 3:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def train_epoch(self):
        mae_meter = AverageMeter()
        rmse_meter = AverageMeter()
        loss_meter = AverageMeter()

        self.model.train()
        # self.ema_model.train()

        unlabeled_bs = self.args.batch_size - self.args.label_batch_size
        assert unlabeled_bs >= 0
        for i, db in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            self.global_step += 1

            img = db['img']
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

                # supervised loss
                label_outputs = self.model(img[unlabeled_bs:])
                N_l = label_outputs.shape[0]
                sup_loss = self.args.lambda_sup * self.mse_fn(label_outputs,
                                                              gt_den[unlabeled_bs:]) / N_l

                # unsupervised loss
                self.model.apply(disable_bn_tracking)
                unlabeled_outputs = self.model(img[:unlabeled_bs])
                adv_loss = self.vat_loss(self.model, img[:unlabeled_bs], unlabeled_outputs.detach(),
                                         eps=self.args.eps,
                                         num_iters=self.args.num_iters)
                self.model.apply(enable_bn_tracking)

                unsup_loss = self.args.lambda_unsup * adv_loss

                outputs = torch.cat([unlabeled_outputs, label_outputs], dim=0)

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
