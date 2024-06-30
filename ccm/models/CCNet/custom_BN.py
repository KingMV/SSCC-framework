import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

debug_enable = False


class MixBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'clean':
            # print('clean bn')
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            # print('mix bn')
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            # input0 = self.aux_bn(input[:batch_size * 1 // 2])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size * 1 // 2])
            input1 = super(MixBatchNorm2d, self).forward(input[batch_size * 1 // 2:])

            input = torch.cat((input0, input1), 0)
        return input

class SplitBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super(SplitBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits

        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                           track_running_stats=track_running_stats) for _ in range(num_splits - 1)])
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            if debug_enable:
                print('adv bn')
            input = self.aux_bn(input)
            return input
        elif self.batch_type == 'clean':
            if debug_enable:
                print('clean bn')
            input = super(SplitBatchNorm2d, self).forward(input)
            return input
        elif self.batch_type == 'mix':
            if debug_enable:
                print('cdn bn')
            assert self.batch_type == 'mix'
            # print('mix bn')
            batch_size = input.shape[0]
            # print(batch_size)
            split_size = batch_size // self.num_splits
            # print(split_size)
            # assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"

            split_input = input.split(split_size)
            x = []
            running_mean_2 = 0
            running_var_2 = 0
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i]))
                running_mean_2 += a.running_mean
                running_var_2 += a.running_var
                unlabel_source = x[-1].permute(0, 2, 3, 1).contiguous().view(-1, self.num_features)
                unlabel_mean = torch.mean(unlabel_source, dim=0)
                unlabel_var = torch.var(unlabel_source, dim=0)
            # for i, a in enumerate(self.aux_bn):
            #     x.append(a(split_input[i]))
            x.append(super(SplitBatchNorm2d, self).forward(split_input[-1]))
            # print(self.running_mean.shape)
            # print(self.running_mean.grad)
            self.running_mean = self.running_mean + 0.1 * unlabel_mean.detach()
            self.running_var = self.running_var + 0.1 * unlabel_var.detach()
            return torch.cat(x, dim=0)
        #     # input0 = self.aux_bn(input[: batch_size // 2])
        #     # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
        #     input0 = self.aux_bn(input[:batch_size // 2])
        #     input1 = super(SplitBatchNorm2d, self).forward(input[batch_size // 2:])
        #     input = torch.cat((input0, input1), 0)
        # return input


class DSBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super(DSBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits

        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                           track_running_stats=track_running_stats) for _ in range(num_splits - 1)])
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            if debug_enable:
                print('adv bn')
            input = self.aux_bn(input)
            return input
        elif self.batch_type == 'clean':
            if debug_enable:
                print('clean bn')
            input = super(DSBN, self).forward(input)
            return input
        elif self.batch_type == 'mix':
            if debug_enable:
                print('DSBN bn')
            assert self.batch_type == 'mix'
            # print('mix bn')
            batch_size = input.shape[0]
            # print(batch_size)
            split_size = batch_size // self.num_splits
            # print(split_size)
            # assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"

            split_input = input.split(split_size)
            x = []
            # running_mean_2 = 0
            # running_var_2 = 0
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i]))
                # running_mean_2 += a.running_mean
                # running_var_2 += a.running_var
                # unlabel_source = x[-1].permute(0, 2, 3, 1).contiguous().view(-1, self.num_features)
                # unlabel_mean = torch.mean(unlabel_source, dim=0)
                # unlabel_var = torch.var(unlabel_source, dim=0)
            x.append(super(DSBN, self).forward(split_input[-1]))
            # print(self.running_mean.shape)
            # print(self.running_mean.grad)
            # self.running_mean = self.running_mean + 0.1 * unlabel_mean.detach()
            # self.running_var = self.running_var + 0.1 * unlabel_var.detach()
            return torch.cat(x, dim=0)
        #     # input0 = self.aux_bn(input[: batch_size // 2])
        #     # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
        #     input0 = self.aux_bn(input[:batch_size // 2])
        #     input1 = super(SplitBatchNorm2d, self).forward(input[batch_size // 2:])
        #     input = torch.cat((input0, input1), 0)
        # return input


class OurBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super(OurBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                           track_running_stats=track_running_stats) for _ in range(num_splits - 1)])
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
            return input
        elif self.batch_type == 'clean':
            input = super(OurBatchNorm2d, self).forward(input)
            return input
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            split_size = batch_size // self.num_splits
            # assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = input.split(split_size)
            x = []
            running_mean_2 = 0
            running_var_2 = 0
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i]))
                running_mean_2 += a.running_mean
                running_var_2 += a.running_var
            print(running_mean_2)
            print(running_var_2)
            x.append(super(OurBatchNorm2d, self).forward(split_input[-1]))
            return torch.cat(x, dim=0)


class _TransNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_TransNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        if self.training and self.track_running_stats:
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]
            z_source = F.batch_norm(
                input_source, self.running_mean_source, self.running_var_source, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
            z_target = F.batch_norm(
                input_target, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
