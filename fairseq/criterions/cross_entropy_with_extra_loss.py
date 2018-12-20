# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion, cross_entropy, label_smoothed_cross_entropy

@register_criterion('cross_entropy_extra_loss')
class CrossEntropyExtraLossCriterions(cross_entropy.CrossEntropyCriterion):
    def __init__(self, args, task):
        super(CrossEntropyExtraLossCriterions, self).__init__(args, task)
        self.alpha = args.extra_loss_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if type(net_output[1]) == dict:
            if 'extra_loss' in net_output[1].keys():
                extra_loss = net_output[1]['extra_loss']
            else:
                extra_loss = 0
                raise ValueError('Extra-loss criterions must be used by model with extra loss appended.')
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        loss += extra_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'extra-loss': utils.item(extra_loss.data) if reduce else extra_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        cross_entropy.CrossEntropyCriterion.add_args(parser)
        parser.add_argument('--extra-loss-weight', default=1., type=float, metavar='D',
                            help='epsilon for extra loss, 0 means take no extra loss into account')

@register_criterion('label_smoothed_cross_entropy_extra_loss')
class LabelSmoothedCrossEntropyExtraLossCriterion(label_smoothed_cross_entropy.LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super(LabelSmoothedCrossEntropyExtraLossCriterion, self).__init__(args, task)
        self.alpha = args.extra_loss_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if type(net_output[1]) == dict:
            if 'extra_loss' in net_output[1].keys():
                extra_loss = net_output[1]['extra_loss']
            else:
                extra_loss = 0
                raise ValueError('Extra-loss criterions must be used by model with extra loss appended.')

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss += extra_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'extra_loss': utils.item(extra_loss.data) if reduce else extra_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # todo : solve this ugly function call.
        label_smoothed_cross_entropy.LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--extra-loss-weight', default=1., type=float, metavar='D',
                            help='epsilon for extra loss, 0 means take no extra loss into account')