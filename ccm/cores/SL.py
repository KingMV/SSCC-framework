from torch import optim
from torch.optim.lr_scheduler import StepLR
from cccv.utils.utils_tool import Timer, AverageMeter
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import os

from ccm.dataset.dataloader import make_dataloader
import importlib


class SL_Trainer(object):
    def __init__(self, args, logger):
        self.logger = logger
        self.args = args

        self.logger.info('Performing Label-Only Training')
        # self.logger.info('create dataset and dataloader')
        self.train_loader, self.val_loader, self.test_loader = make_dataloader(args)

        # create the model
        net_module = getattr(importlib.import_module('.CCNet.{}'.format(args.network), 'ccm.models'),
                             '{0}'.format(args.network))
        self.model = net_module(args)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model.to(self._device), device_ids=args.gpus)
            else:
                self.model = self.model.to(self._device)
        else:
            raise ValueError('no gpu device available')

        # define the optimizer and scheduler
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = StepLR(
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
            self.scheduler.step()

            # validation
            if self.epoch % self.args.val_freq == 0 and self.epoch > self.args.val_start:
                self.timer['val time'].tic()
                self.val()
                self.timer['val time'].toc(average=False)
                self.logger.info('epoch: [{}/{}], val time : {:.2f}s'.format(self.epoch,
                                                                             self.total_epoch,
                                                                             self.timer['val time'].diff))

    def train_epoch(self):
        mae_meter = AverageMeter()
        rmse_meter = AverageMeter()
        loss_meter = AverageMeter()

        self.model.train()

        unlabeled_bs = self.args.batch_size - self.args.label_batch_size
        assert unlabeled_bs == 0, "unlabeled_bs must be 0"
        for i, db in enumerate(self.train_loader):
            self.timer['iter time'].tic()

            img = db['img']
            gt_den = db['den']

            img_labeled = img[unlabeled_bs:].to(self._device)
            den_labeled = gt_den[unlabeled_bs:].to(self._device)
            gt_cnt = den_labeled.view(den_labeled.shape[0], -1).sum(1).cpu()
            outputs = self.model(img_labeled)
            sup_loss = self.mse_fn(outputs, den_labeled) / outputs.shape[0]

            loss = sup_loss

            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate mae and rmse of training data
            N = outputs.shape[0]
            pred_cnt = torch.sum(outputs.view(N, -1), dim=-1).detach().cpu().numpy()
            res = pred_cnt - gt_cnt.numpy()
            mae_meter.update(np.mean(abs(res)))
            rmse_meter.update(np.mean(res * res))

            if i % self.args.log_freq == 0:
                self.timer['iter time'].toc(average=False)
                self.logger.info(
                    'E-{} iter-{}, Loss[total={:.4f}, sup={:.4f}], gt={:.1f} pred={:.1f} '
                    'lr={:.4f} cost={:.1f} sec'.format(self.epoch, i,
                                                       loss.item(), sup_loss.item(),
                                                       gt_cnt[0], pred_cnt[0],
                                                       self.optimizer.param_groups[0]['lr'] * 10000,
                                                       self.timer['train time'].diff))

    def val(self):
        self.model.eval()

        epoch_res = []
        for i, data in enumerate(self.test_loader):
            self.timer['iter time'].tic()
            img = data['img']
            gt_den = data['den']
            gt_cnt = gt_den.view(gt_den.shape[0], -1).sum(1)

            img = img.to(self._device)
            # gt_den = gt_den.to(self._device)

            with torch.no_grad():
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
                save_name = 'E-{}_stu_MAE-{:.2f}_RMSE-{:.2f}.pth'.format(self.epoch, mae, rmse)
                torch.save(to_save_dict, os.path.join(self.ckpt_save_dir, save_name))
