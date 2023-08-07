from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

from torch import Tensor

class CrossEntropyLossWeighted(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, n_classes=3):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = n_classes

    def one_hot(self, targets):
        targets_extend=targets.clone()
        targets_extend.unsqueeze_(1) # convert to Nx1xHxW
        one_hot = torch.FloatTensor(targets_extend.size(0), self.n_classes, targets_extend.size(2), targets_extend.size(3)).zero_().to(targets.device)
        one_hot.scatter_(1, targets_extend, 1)
        
        return one_hot
    
    def forward(self, inputs, targets):
        one_hot = self.one_hot(targets)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(one_hot, dim=(2, 3), keepdim=True)/torch.sum(one_hot)
        one_hot = weights*one_hot

        loss = self.ce(inputs, targets).unsqueeze(1) # shape is batch, 1, 256, 256
        loss = loss*one_hot

        return torch.sum(loss)/(torch.sum(weights)*targets.size(0)*targets.size(1))


class ContourRegularizationLoss(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2 * d + 1)

    def forward(self, x):
        # x is the probability maps
        C_d = self.max_pool(x) + self.max_pool(-1*x) # size is batch x 1 x h x w

        loss = torch.norm(C_d, p=2, dim=(2, 3)).mean()
        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device) # batch x h x w x nclasses
        
        label_one_hot = label_one_hot.permute(0, 3, 1, 2) if label_one_hot.dim() == 4 else label_one_hot # batch x nclasses x h x w
        
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

class PPC(nn.Module, ABC):
    def __init__(self, config):
        super(PPC, self).__init__()

        self.config = config
        self.temperature = self.config['temperature']
        self.ignore_label = -1
        if self.config['ce_ignore_index']!=-1:
            self.ignore_label = self.config['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits/self.temperature, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, config):
        super(PPD, self).__init__()

        self.config = config

        self.ignore_label = -1
        if self.config['ce_ignore_index']!=-1:
            self.ignore_label = self.config['ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, config=None):
        super(PixelPrototypeCELoss, self).__init__()

        self.config = config
        if config['use_prototype']:
            self.alpha = 20
        else:
            self.alpha = 1

        self.ignore_index = -1
        if self.config['ce_ignore_index']!=-1:
            self.ignore_index = self.config['ce_ignore_index']

        self.loss_ppc_weight = self.config.get('loss_ppc_weight',0)
        self.loss_ppd_weight = self.config.get('loss_ppd_weight',0)

        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.loss_ppc_weight > 0:
            self.ppc_criterion = PPC(config=config)
        if self.loss_ppd_weight > 0:
            self.ppd_criterion = PPD(config=config)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            loss = self.seg_criterion(seg*self.alpha, target)
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        seg = preds
        loss = self.seg_criterion(seg*self.alpha, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, input, target):
        # N = target.size(0)
        # smooth = 1

        # input_flat = input.view(N, -1)
        # target_flat = target.view(N, -1)

        # intersection = input_flat * target_flat

        # loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / N
        
        smooth = 1

        input_flat = input.flatten()
        target_flat = target.flatten()

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        loss = 1 - loss

        return loss
 
class MultiClassDiceLoss(nn.Module):
    def __init__(self, config=None):
        super(MultiClassDiceLoss, self).__init__()

        self.config = config
        if config['use_prototype']:
            self.alpha = 20
        else:
            self.alpha = 1
        self.num_classes = self.config['num_classes']
        self.ignore_index = -1
        if self.config['dice_ignore_index']!=-1:
            self.ignore_index = self.config['dice_ignore_index']
        self.dice_criterion = DiceLoss()

    def forward(self, preds, target, weights=None):
        target = F.one_hot(target,self.num_classes).permute((0, 3, 1, 2)).float()
        totalLoss = 0
        if isinstance(preds, dict):
            seg = preds['seg']
        else:
            seg = preds
        seg = F.softmax(seg*self.alpha,dim=1)
        count = 0
        for i in range(self.num_classes):
            if i == self.ignore_index:
                continue
            diceLoss = self.dice_criterion(seg[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
            count+=1
        return totalLoss/count