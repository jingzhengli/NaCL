import torch
import torch.nn as nn
from typing import Optional, Sequence
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def supervised_loss(self, inputs, all_labels, la_labels):
    targets_onehot = torch.zeros(inputs.size()).to(self.device)
    for i in range(inputs.size(0)):
        this_idx = all_labels == la_labels[i].float()
        one_tensor = torch.ones(1).to(self.device)
        this_idx = torch.cat((one_tensor == 1, this_idx))
        ones_mat = torch.ones(torch.nonzero(this_idx).size(0)).to(self.device)
        weights = F.softmax(ones_mat, dim=0)
        targets_onehot[i, this_idx] = weights
    # targets_onehot[:, 0] = 0.2
    targets = targets_onehot.detach()
    outputs = F.log_softmax(inputs, dim=1)
    loss = - (targets * outputs)
    loss = loss.sum(dim=1)
    loss = loss.mean(dim=0)
    return loss


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, r_src, r_tgt, pos_matrix, neg_matrix):
        """
            Compute the NCE scores for predicting r_src->r_trg.
            Input:
              r_src    : (n_batch, n_rkhs)
              r_tgt    : (n_keys, n_rkhs)
              pos_matrix : (n_batch, n_keys)
              neg_matrix : (n_batch, n_keys)
            Output:
              query_to_key_loss  : scalar
              contrast_norm_loss : scalar
        """
        # compute src->trg raw scores for batch
        # (n_batch, n_keys)
        raw_scores = torch.mm(r_src, r_tgt.transpose(0, 1)).float()
        raw_scores /= self.temperature

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-1 / self.temperature here)
        '''
        # (n_batch, n_keys)
        pos_scores = (pos_matrix * raw_scores)

        # (n_batch, n_keys)
        neg_scores = (neg_matrix * raw_scores) - ((1. - neg_matrix) / self.temperature)

        '''
        for each set of positive examples P_i, compute the max over scores
        for the set of negative samples N_i that are shared across P_i
        '''
        # (n_batch, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        '''
        compute a "partial, safe sum exp" over each negative sample set N_i,
        to broadcast across the positive samples in P_i which share N_i
        -- size will be (n_batch, 1)
        '''
        neg_sumexp = (neg_matrix * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)

        '''
        use broadcasting of neg_sumexp across the scores in P_i, to compute
        the log-sum-exps for the denominators in the NCE log-softmaxes
        -- size will be (n_batch, n_keys)
        '''
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # compute numerators for the NCE log-softmaxes
        # (n_batch, n_keys)
        pos_shiftexp = pos_scores - neg_maxes

        # compute the final log-softmax scores for NCE...
        # (n_batch, n_keys)
        nce_scores = pos_matrix * (pos_shiftexp - all_logsumexp)

        contrast_loss = -nce_scores.sum() / pos_matrix.sum()

        return contrast_loss
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def info_nce_logits(features):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    # logits = logits / args.temperature
    logits = logits
    return logits, labels



class AdaptiveFeatureNorm(nn.Module):
    r"""
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    We denote parameters of backbone :math:`G` as :math:`\theta_g`, parameters of bottleneck :math:`F_f` as :math:`\theta_f`
    , parameters of classifier head :math:`F_y` as :math:`\theta_y`, and features extracted from sample :math:`x_i` as
    :math:`h(x_i;\theta)`. Full loss is calculated as follows

    .. math::
        L(\theta_g,\theta_f,\theta_y)=\frac{1}{n_s}\sum_{(x_i,y_i)\in D_s}L_y(x_i,y_i)+\frac{\lambda}{n_s+n_t}
        \sum_{x_i\in D_s\cup D_t}L_d(h(x_i;\theta_0)+\Delta_r,h(x_i;\theta))\\

    where :math:`L_y` denotes classification loss, :math:`L_d` denotes norm loss, :math:`\theta_0` and :math:`\theta`
    represent the updated and updating model parameters in the last and current iterations respectively.

    Args:
        delta (float): positive residual scalar to control the feature norm enlargement.

    Inputs:
        - f (tensor): feature representations on source or target domain.

    Shape:
        - f: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    Examples::

        >>> adaptive_feature_norm = AdaptiveFeatureNorm(delta=1)
        >>> f_s = torch.randn(32, 1000)
        >>> f_t = torch.randn(32, 1000)
        >>> norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
    """

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

class BatchSpectralPenalizationLoss(nn.Module):
    r"""Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
    Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>`_.

    Given source features :math:`f_s` and target features :math:`f_t` in current mini batch, singular value
    decomposition is first performed

    .. math::
        f_s = U_s\Sigma_sV_s^T

    .. math::
        f_t = U_t\Sigma_tV_t^T

    Then batch spectral penalization loss is calculated as

    .. math::
        loss=\sum_{i=1}^k(\sigma_{s,i}^2+\sigma_{t,i}^2)

    where :math:`\sigma_{s,i},\sigma_{t,i}` refer to the :math:`i-th` largest singular value of source features
    and target features respectively. We empirically set :math:`k=1`.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self):
        super(BatchSpectralPenalizationLoss, self).__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss
