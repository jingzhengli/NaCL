import math

import torch
import torch.nn.functional as F
import tqdm
from scipy.optimize import linear_sum_assignment

from model.utils import tanh_clip
from utils import to_one_hot


class BasePseudoLabeler(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        raise NotImplementedError


class KMeansPseudoLabeler(BasePseudoLabeler):
    def __init__(self, num_classes, batch_size=4096, eps=0.0005):
        super().__init__(num_classes)
        self.batch_size = batch_size
        self.eps = eps
        self.init_centers = None
        self.centers = None
        self.stop = False

    @staticmethod
    def get_dist(point_a, point_b, cross=False):
        point_a = F.normalize(point_a, dim=1)
        point_b = F.normalize(point_b, dim=1)
        if not cross:
            return 0.5 * (torch.tensor(1.0).cuda() - torch.sum(point_a * point_b, dim=1))
        else:
            assert (point_a.size(1) == point_b.size(1))
            return 0.5 * (torch.tensor(1.0).cuda() - torch.mm(point_a, point_b.transpose(0, 1)))

    def get_src_centers(self, src_features, src_true_labels):
        centers = 0
        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).cuda()
        num_batches = src_features.size(0) // self.batch_size + 1
        src_index = 0
        for i in range(num_batches):
            cur_len = min(self.batch_size, src_features.size(0) - src_index)
            cur_features = src_features.narrow(0, src_index, cur_len)
            cur_true_labels = src_true_labels.narrow(0, src_index, cur_len)

            cur_true_labels = cur_true_labels.unsqueeze(0).expand(self.num_classes, -1)
            mask = (cur_true_labels == refs).unsqueeze(2).float()
            cur_features = cur_features.unsqueeze(0)

            centers += torch.sum(cur_features * mask, dim=1)
            src_index += cur_len

        return centers

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.get_dist(feats, self.centers, cross=True)
        labels = torch.min(dists, dim=1)[1]
        return dists, labels

    def align_centers(self):
        cost = self.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        """
        pseudo label target samples.
        Args:
            src_test_collection['features']: (n_src, n_rkhs)
            tgt_test_collection['features']: (n_tgt, n_rkhs)

        Returns:
            tgt_pseudo_label : (n_tgt, num_classes) contains pseudo label of entire target samples
        """
        src_features = src_test_collection['features']
        tgt_features = tgt_test_collection['features']

        src_true_labels = src_test_collection['true_labels']

        assert src_features.size(1) == tgt_features.size(1)

        src_centers = self.get_src_centers(src_features, src_true_labels)
        self.init_centers = src_centers
        self.centers = src_centers

        centers = None
        self.stop = False

        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).cuda()
        num_samples = tgt_features.size(0)
        num_split = math.ceil(1.0 * num_samples / self.batch_size)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0
            start = 0

            for _ in range(num_split):
                cur_len = min(self.batch_size, num_samples - start)
                cur_feature = tgt_features.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_one_hot = to_one_hot(labels, self.num_classes)
                count += torch.sum(labels_one_hot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).float()
                reshaped_feature = cur_feature.unsqueeze(0)

                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).float()
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center = []
        start = 0
        for N in range(num_split):
            cur_len = min(self.batch_size, num_samples - start)
            cur_feature = tgt_features.narrow(0, start, cur_len)
            cur_dist2center, _ = self.assign_labels(cur_feature)

            dist2center += [cur_dist2center]
            start += cur_len

        tgt_dist2center = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(tgt_features)
        for k in range(num_samples):
            tgt_dist2center[k] = tgt_dist2center[k][cluster2label]

        return torch.tensor(1.0).cuda() - tgt_dist2center #torch.softmax(torch.tensor(1.0).cuda() - tgt_dist2center, dim=1)


class ClassifierPseudoLabeler(BasePseudoLabeler):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        del src_test_collection
        tgt_logits = tgt_test_collection['logits']
        tgt_pseudo_labels = F.softmax(tgt_logits, dim=1)
        # tgt_pseudo_confidences = torch.max(tgt_pseudo_labels, dim=1)[0]
        # return tgt_pseudo_labels, tgt_pseudo_confidences
        return tgt_pseudo_labels


class InfoPseudoLabeler(BasePseudoLabeler):
    def __init__(self, num_classes, batch_size=4096, tclip=20., normalize=True):
        super().__init__(num_classes)
        self.batch_size = batch_size
        self.tclip = tclip
        self.normalize = normalize

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        """
                pseudo label target samples.
                Args:
                    src_test_collection: (n_src, n_rkhs)
                    tgt_test_collection: (n_tgt, n_rkhs)

                Returns:
                    tgt_pseudo_label : (n_tgt, num_classes) contains pseudo label of entire target samples
                """
        src_features = src_test_collection['features']
        tgt_features = tgt_test_collection['features']
        src_true_labels = src_test_collection['true_labels']

        n_src = src_features.size(0)
        n_tgt = tgt_features.size(0)

        assert src_features.size(1) == tgt_features.size(1)
        n_rkhs = src_features.size(1)

        num_batches = (n_src + n_tgt) // self.batch_size + 1
        src_batch_sizes = [n_src // num_batches + 1] * (n_src % num_batches) \
                          + [n_src // num_batches] * (num_batches - n_src % num_batches)
        tgt_batch_sizes = [n_tgt // num_batches + 1] * (n_tgt % num_batches) \
                          + [n_tgt // num_batches] * (num_batches - n_tgt % num_batches)

        src_perm_indices = torch.randperm(n_src).cuda()
        tgt_perm_indices = torch.randperm(n_tgt).cuda()

        tgt_pseudo_labels = torch.zeros(n_tgt, self.num_classes).cuda()

        src_index = 0
        tgt_index = 0

        for i in tqdm.tqdm(range(num_batches), desc='Target pseudo labeling', leave=False, ascii=True):
            src_batch_indices = src_perm_indices[src_index:src_index + src_batch_sizes[i]]
            tgt_batch_indices = tgt_perm_indices[tgt_index:tgt_index + tgt_batch_sizes[i]]
            src_index += src_batch_sizes[i]
            tgt_index += tgt_batch_sizes[i]
            src_batch_features = src_features[src_batch_indices]
            tgt_batch_features = tgt_features[tgt_batch_indices]
            src_batch_true_labels = src_true_labels[src_batch_indices]

            if self.normalize:
                src_batch_features = F.normalize(src_batch_features, dim=1)
                tgt_batch_features = F.normalize(tgt_batch_features, dim=1)
            raw_scores = torch.mm(tgt_batch_features, src_batch_features.transpose(0, 1)).float()

            if self.normalize:
                raw_scores *= self.tclip
            else:
                raw_scores = raw_scores / n_rkhs ** 0.5
                raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

            prop_scores = F.softmax(raw_scores, dim=1)

            src_one_hot_label = to_one_hot(src_batch_true_labels, self.num_classes)

            tgt_batch_pseudo_label = torch.mm(prop_scores, src_one_hot_label)

            # balance class
            src_class_count = torch.sum(src_one_hot_label, dim=0)
            src_class_count = torch.max(src_class_count, torch.ones_like(src_class_count).cuda())
            tgt_batch_pseudo_label /= src_class_count
            tgt_batch_pseudo_label = F.normalize(tgt_batch_pseudo_label, p=1, dim=1)

            tgt_pseudo_labels[tgt_batch_indices] = tgt_batch_pseudo_label

        tgt_pseudo_confidences = torch.max(tgt_pseudo_labels, dim=1)[0]

        return tgt_pseudo_labels, tgt_pseudo_confidences


class PropagatePseudoLabeler(BasePseudoLabeler):
    def __init__(self, num_classes, batch_size=4096, tclip=20., normalize=False):
        super().__init__(num_classes)
        self.batch_size = batch_size
        self.tclip = tclip
        self.normalize = normalize

    def pseudo_label_tgt(self, src_test_collection, tgt_test_collection):
        """
        pseudo label target samples.
        Args:
            src_test_collection: (n_src, n_rkhs)
            tgt_test_collection: (n_tgt, n_rkhs)

        Returns:
            tgt_pseudo_label : (n_tgt, num_classes) contains pseudo label of entire target samples
        """
        src_features = src_test_collection['features']
        tgt_features = tgt_test_collection['features']
        src_true_labels = src_test_collection['true_labels']

        n_src = src_features.size(0)
        n_tgt = tgt_features.size(0)

        assert src_features.size(1) == tgt_features.size(1)
        n_rkhs = src_features.size(1)

        num_batches = (n_src + n_tgt) // self.batch_size + 1
        src_batch_sizes = [n_src // num_batches + 1] * (n_src % num_batches) \
                          + [n_src // num_batches] * (num_batches - n_src % num_batches)
        tgt_batch_sizes = [n_tgt // num_batches + 1] * (n_tgt % num_batches) \
                          + [n_tgt // num_batches] * (num_batches - n_tgt % num_batches)

        src_perm_indices = torch.randperm(n_src).cuda()
        tgt_perm_indices = torch.randperm(n_tgt).cuda()

        tgt_pseudo_labels = torch.zeros(n_tgt, self.num_classes).cuda()

        src_index = 0
        tgt_index = 0

        for i in tqdm.tqdm(range(num_batches), desc='Target pseudo labeling', leave=False, ascii=True):
            src_batch_indices = src_perm_indices[src_index:src_index + src_batch_sizes[i]]
            tgt_batch_indices = tgt_perm_indices[tgt_index:tgt_index + tgt_batch_sizes[i]]
            src_index += src_batch_sizes[i]
            tgt_index += tgt_batch_sizes[i]
            src_batch_features = src_features[src_batch_indices]
            tgt_batch_features = tgt_features[tgt_batch_indices]
            src_batch_true_labels = src_true_labels[src_batch_indices]

            n_src_batch = src_batch_features.size(0)
            n_tgt_batch = tgt_batch_features.size(0)

            batch_features = torch.cat([src_batch_features, tgt_batch_features], dim=0)

            if self.normalize:
                batch_features = F.normalize(batch_features, dim=1)

            raw_scores = torch.mm(batch_features, batch_features.transpose(0, 1)).float()

            if self.normalize:
                raw_scores *= self.tclip
            else:
                raw_scores = raw_scores / n_rkhs ** 0.5
                raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

            prop_scores = F.softmax(raw_scores, dim=1)

            prop_ts = prop_scores[n_src_batch:, :n_src_batch]
            prop_tt = prop_scores[n_src_batch:, n_src_batch:]

            src_one_hot_label = to_one_hot(src_batch_true_labels, self.num_classes)

            # initialize tgt pseudo label
            # tgt_batch_pseudo_label = torch.ones(n_tgt_batch, self.num_classes).cuda() / self.num_classes

            # label propagation
            # for _ in range(self.lp_iterations):
            #     tgt_batch_pseudo_label = torch.mm(prop_ts, src_one_hot_label) \
            #                              + torch.mm(prop_tt, tgt_batch_pseudo_label)

            tgt_batch_pseudo_label = torch.mm(
                torch.inverse(torch.eye(n_tgt_batch).cuda() - prop_tt),
                torch.mm(prop_ts, src_one_hot_label))

            tgt_pseudo_labels[tgt_batch_indices] = tgt_batch_pseudo_label

        tgt_pseudo_confidences = torch.max(tgt_pseudo_labels, dim=1)[0]

        return tgt_pseudo_labels, tgt_pseudo_confidences
