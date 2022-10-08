from __future__ import print_function
import argparse
import queue
import timm
import common.vision.models as models

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def configure(filename):
    with open(filename, 'r') as f:
        parser = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser


def summary_write_fig(summary_writer, tag, global_step, model, images, labels, domain):
    model.set_bn_domain(domain=domain)
    model.eval()

    with torch.no_grad():
        end_points = model(images)
        figure = plot_classes_predictions(images, labels, end_points['predictions'], end_points['confidences'])

    summary_writer.add_figure(tag=tag,
                              figure=figure,
                              global_step=global_step)
    summary_writer.close()


def plot_classes_predictions(images, labels, predictions, confidences):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    """
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 30))
    for idx in np.arange(min(32, len(images))):
        ax = fig.add_subplot(8, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            predictions[idx],
            confidences[idx] * 100.0,
            labels[idx]),
            color=("green" if predictions[idx] == labels[idx] else "red"))
    return fig


def matplotlib_imshow(image):
    np_image = image.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = np_image * np.array(_STD) + np.array(_MEAN)
    np_image = np.clip(np_image, 0., 1.)
    plt.imshow(np_image)


def summary_write_proj(summary_writer, tag, global_step, model, src_train_loader, tgt_train_loader,
                       num_samples=128):
    total_iteration = num_samples // src_train_loader.data_loader.batch_size
    model.eval()
    with torch.no_grad():
        features_list = []
        class_labels_list = []
        domain_labels_list = []

        for (src_data, tgt_data) in zip(src_train_loader, tgt_train_loader):
            src_inputs, src_labels = src_data['image_1'].cuda(), src_data['true_label'].cuda()
            tgt_inputs, tgt_labels = tgt_data['image_1'].cuda(), tgt_data['true_label'].cuda()
            model.set_bn_domain(domain=0)
            src_end_points = model(src_inputs)
            model.set_bn_domain(domain=1)
            tgt_end_points = model(tgt_inputs)
            src_features = src_end_points[tag]
            tgt_features = tgt_end_points[tag]
            features = torch.cat([src_features, tgt_features], dim=0)
            features_list.append(features)

            class_labels = torch.cat((src_labels, tgt_labels), dim=0)
            class_labels_list.append(class_labels)

            domain_labels = ['S'] * src_labels.size(0) + ['T'] * tgt_labels.size(0)
            domain_labels_list.extend(domain_labels)

            if len(features_list) >= total_iteration:
                break

        all_features = torch.cat(features_list, dim=0)
        all_class_labels = torch.cat(class_labels_list, dim=0)
        all_class_labels = all_class_labels.cpu().numpy()

    summary_writer.add_embedding(all_features,
                                 metadata=all_class_labels,
                                 global_step=global_step,
                                 tag=tag + "_class")
    summary_writer.add_embedding(all_features,
                                 metadata=domain_labels_list,
                                 global_step=global_step,
                                 tag=tag + "_domain")
    summary_writer.close()


class ImageTransform(Dataset):
    def __init__(self, images, transform=None):
        assert len(images) > 0
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


def get_dataset_name(src_name, tgt_name):
    dataset_names = {
        'amazon': 'Office31',
        'dslr': 'Office31',
        'webcam': 'Office31',
        'c': 'image-clef',
        'i': 'image-clef',
        'p': 'image-clef',
        'Art': 'OfficeHome',
        'Clipart': 'OfficeHome',
        'Product': 'OfficeHome',
        'Real_World': 'OfficeHome',
        'train': 'visda-2017',
        'validation': 'visda-2017'
    }
    assert (dataset_names[src_name] == dataset_names[tgt_name])
    return dataset_names[src_name]


class AvgMeter:
    def __init__(self, maxsize=10):
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, item):
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)

    def get(self):
        return self.queue.get()

    def get_average(self):
        sum_all = 0.
        queue_len = self.queue.qsize()
        if queue_len == 0:
            return 0
        while not self.queue.empty():
            sum_all += self.queue.get()
        return round(sum_all / queue_len, 5)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_labels_from_file(file_name):
    image_list = open(file_name).readlines()
    labels = [int(val.split()[1]) for val in image_list]
    return labels


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def compute_accuracy(logits, true_labels, acc_metric='total_mean', print_result=False):
    assert logits.size(0) == true_labels.size(0)
    if acc_metric == 'total_mean':
        predictions = torch.max(logits, dim=1)[1]
        accuracy = 100.0 * (predictions == true_labels).sum().item() / logits.size(0)
        if print_result:
            print(accuracy)
        return accuracy
    elif acc_metric == 'class_mean':
        num_classes = logits.size(1)
        predictions = torch.max(logits, dim=1)[1]
        class_accuracies = []
        for class_label in range(num_classes):
            class_mask = (true_labels == class_label)

            class_count = class_mask.sum().item()
            if class_count == 0:
                class_accuracies += [0.0]
                continue

            class_accuracy = 100.0 * (predictions[class_mask] == class_label).sum().item() / class_count
            class_accuracies += [class_accuracy]
        if print_result:
            print(f'class_accuracies: {class_accuracies}')
            print(f'class_mean_accuracies: {np.mean(class_accuracies)}')
        return np.mean(class_accuracies)
    else:
        raise ValueError(f'acc_metric, {acc_metric} is not available.')


def to_one_hot(label, num_classes):
    identity = torch.eye(num_classes).cuda()
    one_hot = torch.index_select(identity, 0, label)
    return one_hot


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')






import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
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
        # mask-out self-contrast cases 对角线去掉
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
from PythonGraphPers_withCompInfo import PyPers, PyPersCC, PyPersRev, PyPersCCRev, PyPersAll
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import timeit
# import faiss


def pairwise_distance(point_cloud_refer, point_cloud_query):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (num_points, num_dims)
    Returns:
      pairwise distance: (num_points, num_points)
    """
    point_cloud_transpose = torch.transpose(point_cloud_refer, 0, 1)

    point_cloud_inner = torch.matmul(point_cloud_query, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner

    point_cloud_query_square = torch.sum(point_cloud_query**2, dim=-1, keepdim=True)
    point_cloud_refer_square = torch.sum(point_cloud_refer**2, dim=-1, keepdim=True)
    point_cloud_refer_square = torch.transpose(point_cloud_refer_square, 0, 1)

    return point_cloud_query_square + point_cloud_inner + point_cloud_refer_square


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    dists, nn_idx = torch.topk(neg_adj, k=k)
    return nn_idx, dists


def calc_knn_graph(feats_point_cloud, k=2, refer_trunk_size=50000, query_trunk_size=10000):
    """
    Since GPU knn is memory intensive, so we split the query and reference data points into several trunks.
    Each time, we process a trunk of data (in other words, a batch of data).

    refer_trunk_size: The trunk size for the reference points.
    query_trunk_size: The trunk size for the query points.
    """
    with torch.no_grad():
        num_refer_trunk = feats_point_cloud.size(0) // refer_trunk_size
        remain_refer = feats_point_cloud.size(0) - num_refer_trunk * refer_trunk_size

        num_query_trunk = feats_point_cloud.size(0) // query_trunk_size
        remain_query = feats_point_cloud.size(0) - num_query_trunk * query_trunk_size

        knnG = []
        for i in range(num_query_trunk):
            curr_query = feats_point_cloud[i*query_trunk_size:(i+1)*query_trunk_size]

            curr_dist = []
            for j in range(num_refer_trunk):
                curr_refer = feats_point_cloud[j*refer_trunk_size:(j+1)*refer_trunk_size]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            if remain_refer > 0:
                curr_refer = feats_point_cloud[num_refer_trunk * refer_trunk_size:]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            curr_dist = torch.cat(curr_dist, 1)
            knnG.append(torch.topk(curr_dist, k=k+1)[1])

        # if there remain some data points ...
        if remain_query > 0:
            curr_query = feats_point_cloud[num_query_trunk * query_trunk_size:]

            curr_dist = []
            for j in range(num_refer_trunk):
                curr_refer = feats_point_cloud[j * refer_trunk_size:(j + 1) * refer_trunk_size]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            if remain_refer > 0:
                curr_refer = feats_point_cloud[num_refer_trunk * refer_trunk_size:]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            curr_dist = torch.cat(curr_dist, 1)
            knnG.append(torch.topk(curr_dist, k=k + 1)[1])

        knnG = torch.cat(knnG, 0)
        knnG_list = knnG.cpu().numpy().tolist()

    return knnG_list


# -- function for computing topo weights
def calc_topo_weights_with_components_idx(ntrain, prob_all, feats_point_cloud, ori_label, pred_label,
                                          use_log=False, nclass=10, k=2, cp_opt=3,
                                          refer_trunk_size=50000, query_trunk_size=10000):
    """
    Since GPU knn is memory intensive, so we split the query and reference data points into several trunks.
    Each time, we process a trunk of data (in other words, a batch of data).

    refer_trunk_size: The trunk size for the reference points.
    query_trunk_size: The trunk size for the query points.

    nclass: The number of class.
    cp_opt: Should always be set to 3 here. Just use it as a black box. The underlying reason is rooted in the C++ code
        for computing the largest connected component (which was originally written for computing the persistent homology).
    """
    # -- first, compute the knn graph --
    print('computing knn graph')
    start = timeit.default_timer()
    
    knnG_list = calc_knn_graph(feats_point_cloud, k=k, refer_trunk_size=refer_trunk_size, query_trunk_size=query_trunk_size)

    stop = timeit.default_timer()
    print('Finish computing knn graph. Consume time: ', stop - start)

    # -- next, compute phi functions, which is related to persistent homology --
    data_selected = set()  # whether a data has been selected
    tot_num_comp = 0
    tot_comp_nvert = 0
    tot_num_pt2fix = 0

    topo_wt = np.zeros((ntrain, nclass))
    idx_of_small_comps = set()

    start = timeit.default_timer()
    for j in range(nclass):
        tmp_prob_curr = prob_all[:, j]
        tmp_prob_all = prob_all.copy()
        tmp_prob_all[:, j] = -1.0
        tmp_prob_alt = np.amax(tmp_prob_all, axis=1)
        tmp_best_alt = np.argmax(tmp_prob_all, axis=1)
        if use_log:
            phi = np.log(tmp_prob_alt) - np.log(tmp_prob_curr)
        else:
            phi = tmp_prob_alt - tmp_prob_curr

        phi_list = list(phi.ravel())

        # Compute persistence
        skip1D = 1
        levelset_val = 0 + np.finfo('float32').eps
        relevant_vlist = PyPersAll(phi_list, knnG_list, ntrain, levelset_val, skip1D, j, ori_label, pred_label)

        assert len(relevant_vlist) == 6
        assert relevant_vlist[0][0] == len(relevant_vlist[1])

        tot_comp_nvert = tot_comp_nvert + relevant_vlist[0][0]
        tot_num_comp = tot_num_comp + relevant_vlist[0][2]
        tot_num_pt2fix = tot_num_pt2fix + len(relevant_vlist[2 + cp_opt])

        curr_comp_nvert = relevant_vlist[0][0]
        curr_ncomp = relevant_vlist[0][2]

        # relevant_vlist[2] -- comp vert list
        # relevant_vlist[3] -- birth vert list
        # relevant_vlist[4] -- crit vert list
        # relevant_vlist[5] -- rob crit vert list
        assert curr_comp_nvert == len(relevant_vlist[2])
        assert curr_ncomp <= len(relevant_vlist[2])  # less and equal
        assert curr_ncomp == len(relevant_vlist[3])
        assert curr_ncomp <= len(relevant_vlist[4])  # less and equal
        assert curr_ncomp >= len(relevant_vlist[5])

        if curr_ncomp == 0:
            print('WARNING: No extra components, skip to the next label.')
            continue

        selected_vidx = relevant_vlist[2 + cp_opt]
        selected_vidx = list(set(selected_vidx).difference(data_selected))
        data_selected = data_selected.union(set(selected_vidx))

        topo_wt[selected_vidx, j] = -1.0
        topo_wt[selected_vidx, tmp_best_alt[selected_vidx]] = 1.0

        idx_of_small_comps = idx_of_small_comps.union(relevant_vlist[2])

    idx_of_small_comps = list(idx_of_small_comps)

    return topo_wt, idx_of_small_comps
def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone