
import os
import os.path as osp
import glob
import torch
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
from data.prepare_data import generate_dataloader_sc
from utils import summary_write_proj, summary_write_fig, AvgMeter, moment_update, compute_accuracy,calc_knn_graph, calc_topo_weights_with_components_idx
from scipy.stats import mode

class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, model_dir, dataset_name,
                 summary_writer, src_file, tgt_file, data_root, contrast_loss,supervised_loss, info_nce_logits,PairEnum,src_memory, tgt_memory, tgt_pseudo_labeler,criterion,adaptive_feature_norm,
                 cw=1.0,
                 thresh=0.95,
                 min_conf_samples=3,
                 num_classes=31,
                 batch_size=36,
                 eval_batch_size=36,
                 num_workers=1,
                 max_iter=100000,
                 iters_per_epoch=100,
                 log_summary_interval=10,
                 log_image_interval=1000,
                 num_proj_samples=384,
                 acc_metric='total_mean',
                 alpha=0.99,transform_type='randomsizedcrop',module='contrastive_only',kcc=2,phase='analysis',momentum_type = 'True', batch_norm = 'False',pseudo_pre = 'True', mcc = 'False'):
        self.model = model
        self.model_ema = model_ema
        self.adaptive_feature_norm = adaptive_feature_norm
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.summary_writer = summary_writer
        self.source = src_file
        self.target = tgt_file
        self.data_root = data_root
        self.contrast_loss = contrast_loss
        self.supervised_loss=supervised_loss
        self.PairEnum = PairEnum
        self.info_nce_logits = info_nce_logits
        self.src_memory = src_memory
        self.tgt_memory = tgt_memory
        self.tgt_pseudo_labeler = tgt_pseudo_labeler
        self.cw = cw
        self.thresh = thresh
        self.min_conf_samples = min_conf_samples
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.iters_per_epoch = iters_per_epoch
        self.log_summary_interval = log_summary_interval
        self.log_image_interval = log_image_interval
        self.num_proj_samples = num_proj_samples
        self.acc_metric = acc_metric
        self.alpha = alpha
        self.transform_type = transform_type
        self.module = module
        self.big_comp = set()
        self.k_cc=kcc
        self.phase = phase
        self.momentum= momentum_type
        self.batch_norm = batch_norm
        self.pseudo_pre = pseudo_pre
        self.mcc = mcc

        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(desc='Iterations', total=self.max_iter, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.acc_dict = {'tgt_best_test_acc': 0.0}
        self.src_train_acc_queue = AvgMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.fixmatch = nn.CrossEntropyLoss(reduction='none').cuda()
        self.criterion = criterion
        self.tgt_conf_pair = None
        self.data_loader = {}
        self.data_iterator = {}

        self.data_loader = generate_dataloader_sc(self.data_root,self.source,self.target,self.batch_size,self.num_workers,self.transform_type)

        self.ntrain = len(self.data_loader['tgt_test'].dataset)
        self.k_outlier = self.ntrain % (5*self.num_classes)

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')
        while self.iteration < self.max_iter:

            self.data_loader = generate_dataloader_sc(self.data_root,self.source,self.target,self.batch_size,self.num_workers,self.transform_type)
            for key in self.data_loader:
                if self.data_loader[key] is not None and key is not 'tgt_data':
                    self.data_iterator[key] = iter(self.data_loader[key])
                else:
                    self.data_iterator[key] = None
            # update target confident dataset
            self.prepare_tgt_conf_dataset()

            # train an epoch
            self.train_epoch()

        self.total_progress_bar.write('Finish training')
        return self.acc_dict['tgt_best_test_acc'], self.acc_dict['tgt_pseudo_acc']

    def train_epoch(self):
        cur_epoch_steps = max(self.iters_per_epoch, len(self.data_loader['tgt_conf']))
        for _ in tqdm.tqdm(range(cur_epoch_steps), desc='Epoch {:4d}'.format(self.epoch), leave=False, ascii=True):
            self.model.train()
            self.model_ema.eval()

            self.train_step()

            if self.iteration % self.log_summary_interval == 0:
                self.log_summary_writer()
        self.epoch += 1
        
    def train_step(self):
        self.lr_scheduler.adjust_learning_rate(self.optimizer, self.iteration)
        if self.batch_norm == 'True':
            self.model.weight_norm()
        (src_inputs_1, src_inputs_2,_), src_labels = self.get_sample('src_train')
        (tgt_inputs_1,tgt_inputs_2,_), tgt_labels = self.get_sample('tgt_train')
        src_inputs_1,src_inputs_2,src_labels, tgt_inputs_1, tgt_inputs_2, tgt_labels\
                = src_inputs_1.cuda(), src_inputs_2.cuda(), src_labels.cuda(),tgt_inputs_1.cuda(),tgt_inputs_2.cuda(),tgt_labels.cuda()
        stu = src_inputs_1.size(0)
        ttu = tgt_inputs_1.size(0)
        if self.momentum == 'True':
            imgs_query = torch.cat([src_inputs_1, tgt_inputs_1], dim=0).cuda()
            query_end_points = self.model(imgs_query)
            logits_u_w_s, logits_u_w_t = torch.split(query_end_points['logits'], [stu, ttu], dim=0)
            self.src_supervised_step(logits_u_w_s, src_labels)
            imgs_key = torch.cat([src_inputs_2, tgt_inputs_2], dim=0).cuda()
            # shuffle BN
            idx = torch.randperm(imgs_key.size(0))
            key_end_points = self.model_ema(imgs_key[idx])
            key_end_points['contrast_features'] = key_end_points['contrast_features'][torch.argsort(idx)]
            feat_key_s, feat_key_t = torch.split(key_end_points['contrast_features'], [stu, ttu], dim=0)
            src_key_features, _ = self.src_memory.get_queue()
            tgt_key_features, _ = self.tgt_memory.get_queue()
            memory_key_features = torch.cat([src_key_features, tgt_key_features], dim=0)
            score_pos = torch.bmm(query_end_points['contrast_features'].unsqueeze(dim=1), key_end_points['contrast_features'].unsqueeze(dim=-1)).squeeze(dim=-1)
            score_neg = torch.mm(query_end_points['contrast_features'], memory_key_features.t().contiguous())

            self.src_memory.store_keys(feat_key_s, src_labels)
            self.tgt_memory.store_keys(feat_key_t, tgt_labels)
            out = torch.cat([score_pos, score_neg], dim=-1)
            # compute loss
            self.losses_dict['contrastive_loss'] = F.cross_entropy(out, torch.zeros(query_end_points['contrast_features'].size(0), dtype=torch.long).cuda())
        else:
            imgs_query = torch.cat([src_inputs_1, src_inputs_2], dim=0).cuda()
            query_end_points = self.model(imgs_query)
            logits_u_w_s, _ = torch.split(query_end_points['logits'], [stu, stu], dim=0)
            self.src_supervised_step(logits_u_w_s, src_labels)
            imgs_key = torch.cat([tgt_inputs_1, tgt_inputs_2], dim=0).cuda()
            # shuffle BN
            key_end_points = self.model(imgs_key)
            logits_u_w_t, _ = torch.split(key_end_points['logits'], [ttu, ttu], dim=0)

            contrastive_logits_s, contrastive_labels_s = self.info_nce_logits(features=query_end_points['contrast_features'])
            self.losses_dict['contrastive_loss_s'] = torch.nn.CrossEntropyLoss()(contrastive_logits_s, contrastive_labels_s)
            contrastive_logits_t, contrastive_labels_t = self.info_nce_logits(features=key_end_points['contrast_features'])
            self.losses_dict['contrastive_loss_t'] = torch.nn.CrossEntropyLoss()(contrastive_logits_t, contrastive_labels_t)

        if self.data_loader['tgt_conf'] is not None:
            (tgt_cho_inputs_1,tgt_cho_inputs_2,_), tgt_pseudo_labels  = self.get_sample('tgt_conf')
            tgt_cho_inputs_1, tgt_cho_inputs_2, tgt_pseudo_labels \
                = tgt_cho_inputs_1.cuda(), tgt_cho_inputs_2.cuda(), tgt_pseudo_labels.cuda() #torch.stack(tgt_data['pseudo_label'],1).cuda()
            btu2 = tgt_cho_inputs_1.size(0)
            imgs_cho_query = torch.cat((tgt_cho_inputs_1, src_inputs_2, tgt_cho_inputs_2),dim = 0)
            labels_cat = torch.cat((src_labels, tgt_pseudo_labels),dim = 0)
            tgt_cho_end_points = self.model(imgs_cho_query)
            tar_features_1, con_features_2 = torch.split(tgt_cho_end_points['contrast_features'], [btu2,stu+btu2], dim=0)
            sur_features_1, _ = torch.split(query_end_points['contrast_features'], [stu, ttu], dim=0)
            con_features_1 = torch.cat((sur_features_1, tar_features_1),dim = 0)
            con_features_s = torch.cat([con_features_1.unsqueeze(1), con_features_2.unsqueeze(1)], dim=1)
            self.losses_dict['sup_con_loss'] = self.criterion(con_features_s, labels=labels_cat)

            # Fixmatch
            tar_logits_1,_, tar_logits_2 = torch.split(tgt_cho_end_points['logits'], [btu2,stu, btu2], dim=0) 
            with torch.no_grad():
                probs = torch.softmax(tar_logits_1, dim=1)
                scores, pseudo_labelss = torch.max(probs, dim=1)
                mask = scores.ge(self.thresh).float()
                probs = probs.detach()
            if self.pseudo_pre == 'True':
                self.losses_dict['fixmatch']  = (self.fixmatch(tar_logits_2, pseudo_labelss) * mask).mean()
            else:
                self.losses_dict['fixmatch'] = 0
        else:
            self.losses_dict['sup_con_loss'] = 0

        if self.module == 'domain_loss':
            if self.momentum == 'True':
                self.losses_dict['total_loss'] = \
                    self.losses_dict['src_classification_loss'] + self.cw*(self.losses_dict['contrastive_loss'] + self.losses_dict['sup_con_loss'])+self.losses_dict['fixmatch']
            else:
                self.losses_dict['total_loss'] = \
                    self.losses_dict['src_classification_loss'] + self.losses_dict['contrastive_loss_s'] + self.losses_dict['contrastive_loss_t']+ self.losses_dict['sup_con_loss']+self.losses_dict['fixmatch']
        self.optimizer.zero_grad()
        self.losses_dict['total_loss'].backward()
        self.optimizer.step()

        moment_update(self.model, self.model_ema, self.alpha)

        self.iteration += 1
        self.total_progress_bar.update(1)

    def src_supervised_step(self, src_logits, src_labels):
        # compute source classification loss
        src_classification_loss = self.class_criterion(src_logits, src_labels)
        self.losses_dict['src_classification_loss'] = src_classification_loss

        # compute source train accuracy
        src_train_accuracy = compute_accuracy(src_logits, src_labels, acc_metric=self.acc_metric)
        self.src_train_acc_queue.put(src_train_accuracy)
    def tgt_supervised_step(self, tgt_logits, tgt_labels):
        # compute source classification loss
        tgt_classification_loss = self.class_criterion(tgt_logits, tgt_labels)
        self.losses_dict['tgt_classification_loss'] = tgt_classification_loss

        # compute target train accuracy
        tgt_train_accuracy = compute_accuracy(tgt_logits, tgt_labels, acc_metric=self.acc_metric)
        self.src_train_acc_queue.put(tgt_train_accuracy)

    def contrastive_step(self, src_end_points, stu,src_labels, feats_u_w=None, tgt_pseudo_labels=None):
        if feats_u_w is not None:
            batch_features = torch.cat([src_end_points['contrast_features'][:stu], feats_u_w],
                                       dim=0)
            batch_labels = torch.cat([src_labels, tgt_pseudo_labels], dim=0)
        else:
            batch_features = src_end_points['contrast_features'][:stu]
            batch_labels = src_labels
        assert batch_labels.lt(0).sum().cpu().numpy() == 0

        src_key_features, src_key_labels = self.src_memory.get_queue()
        tgt_key_features, tgt_key_labels = self.tgt_memory.get_queue()

        key_features = torch.cat([src_key_features, tgt_key_features], dim=0)
        key_labels = torch.cat([src_key_labels, tgt_key_labels], dim=0)

        # (batch_size, key_size)
        pos_matrix = (key_labels == batch_labels.unsqueeze(1)).float()

        # (batch_size, key_size)
        neg_matrix = (key_labels != batch_labels.unsqueeze(1)).float()

        contrast_loss = self.contrast_loss(batch_features, key_features, pos_matrix, neg_matrix)
        self.losses_dict['contrast_loss'] = contrast_loss

    def sigmoid_rampup(self, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            self.epoch = np.clip(self.epoch, 0.0, rampup_length)
            phase = 1.0 - self.epoch / rampup_length

            return float(np.exp(-5.0 * phase * phase))

    def prepare_tgt_conf_dataset(self):
        
        # collect target samples and source samples
        src_test_collection = self.collect_samples('src_test')
        tgt_test_collection = self.collect_samples('tgt_test')
        tgt_pseudo_probabilities = self.tgt_pseudo_labeler.pseudo_label_tgt(src_test_collection, tgt_test_collection) #probs = torch.softmax(tgt_pseudo_probabilities, dim=1)  
        tgt_pseudo_acc = compute_accuracy(tgt_pseudo_probabilities, tgt_test_collection['true_labels'],
                                          acc_metric=self.acc_metric, print_result=False)
        self.acc_dict['tgt_pseudo_acc'] = tgt_pseudo_acc
        # estimate the results
        self.eval_tgt(tgt_test_collection)
        # compute the pseudo labels and the confidence of target data
        _, tgt_pseudo_labels = torch.max(tgt_pseudo_probabilities, dim=1)
        tgt_pseudo_labels = tgt_pseudo_labels.cpu()
        print('\n>> Computing Big Components <<', 'white')
        labels_all = torch.zeros(self.ntrain,self.num_classes)
        labels_all[torch.arange(self.ntrain), tgt_pseudo_labels] = 1.0
        labels_all = labels_all.numpy()
        import numpy as np
        train_gt_labels = tgt_pseudo_labels.numpy().astype(np.int64).tolist()
        train_pred_labels = np.squeeze(tgt_pseudo_labels.numpy().astype(np.int64)).ravel().tolist()
        tgt_test_collection['features']= tgt_test_collection['features'].cpu()
        _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(self.ntrain, labels_all, tgt_test_collection['features'],
                                                                    train_gt_labels, train_pred_labels, k=self.k_cc,
                                                                    use_log=False, cp_opt=3, nclass=self.num_classes)
        # --- update largest connected component ---
        big_comp = set()
        cur_big_comp = list(set(range(self.ntrain)) - set(idx_of_comp_idx2))
        big_comp = self.big_comp.union(set(cur_big_comp))
        # big_comp = big_comp.union(set(cur_big_comp))
        print(">> The accuracy of abandon: {}".format(np.sum(tgt_test_collection['true_labels'][idx_of_comp_idx2].cpu().numpy() == tgt_pseudo_labels[idx_of_comp_idx2].numpy())
                                                        / float(len(idx_of_comp_idx2))), 'red')
        print("=> number of cur_big_comp and all samples  %d and %d" %(len(cur_big_comp),len(self.data_loader['tgt_test'].dataset)))
        print("=> number of big_comp and all samples  %d and %d" %(len(big_comp),len(self.data_loader['tgt_test'].dataset)))
        print(">> The accuracy of LCC: {}".format(np.sum(tgt_test_collection['true_labels'][cur_big_comp].cpu().numpy() == tgt_pseudo_labels[cur_big_comp].numpy())
                                                        / float(len(cur_big_comp))), 'red')                                                
        # --- remove outliers in largest connected component ---
        big_com_idx = list(big_comp)

        feats_big_comp = tgt_test_collection['features'][big_com_idx]
        labels_big_comp = np.array(train_gt_labels)[big_com_idx]

        knnG_list = calc_knn_graph(feats_big_comp, k=self.k_cc)

        knnG_list = np.array(knnG_list)
        knnG_shape = knnG_list.shape
        knn_labels = labels_big_comp[knnG_list.ravel()]
        knn_labels = np.reshape(knn_labels, knnG_shape)

        majority, counts = mode(knn_labels, axis=1)
        majority = majority.ravel()
        counts = counts.ravel()
        non_outlier_idx = np.where(majority == labels_big_comp)[0]
        outlier_idx = np.where(majority != labels_big_comp)[0]
        outlier_idx = np.array(list(big_comp))[outlier_idx]
        print(">> majority == labels_big_comp -> size: ", len(non_outlier_idx))
        print(">> The number of outliers: {}".format(len(outlier_idx)), 'red')

        print(">> The accuracy of outliers: {}".format(np.sum(tgt_test_collection['true_labels'][outlier_idx].cpu().numpy() == tgt_pseudo_labels[outlier_idx].cpu().numpy())
                                                        / float(len(outlier_idx))), 'red')
        big_comp = np.array(list(big_comp))[non_outlier_idx]
        big_comp = set(big_comp.tolist())
        selected_pseudo_probabilities = [tgt_pseudo_labels[index] for index in big_comp]

        conf_images = [self.data_loader['tgt_data'].samples[index] for index in big_comp]
        self.data_loader['tgt_data'].samples = conf_images
        self.data_loader['tgt_data'].targets = selected_pseudo_probabilities
        cluster_class,counts = np.unique(self.data_loader['tgt_data'].targets,return_counts=True)
        print("unique labels:{}".format(counts))
        print("cluster_class_length:{}".format(len(cluster_class)))
        self.data_loader['tgt_conf'] = torch.utils.data.DataLoader(
            self.data_loader['tgt_data'], batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=False
        )

        if self.data_loader['tgt_conf'] is None:
            self.data_iterator['tgt_conf'] = None
        else:
            self.data_iterator['tgt_conf'] = iter(self.data_loader['tgt_conf'])
        
        print("=> number of selected samples and all samples  %d and %d" %(len(self.data_loader['tgt_conf'].dataset),len(self.data_loader['tgt_test'].dataset)))

    def eval_tgt(self, tgt_test_collection):
        tgt_test_acc = compute_accuracy(tgt_test_collection['logits'], tgt_test_collection['true_labels'],
                                        acc_metric=self.acc_metric, print_result=False)
        tgt_test_acc = round(tgt_test_acc, 3)
        self.acc_dict['tgt_test_acc'] = tgt_test_acc
        # self.acc_dict['tgt_best_test_acc'] = max(self.acc_dict['tgt_best_test_acc'], tgt_test_acc)
        if self.acc_dict['tgt_test_acc'] > self.acc_dict['tgt_best_test_acc']:
            self.acc_dict['tgt_best_test_acc'] = self.acc_dict['tgt_test_acc']
            self.save_checkpoint()
            torch.save(self.model_ema.state_dict(), os.path.join(self.model_dir,'checkpoint.pth'))
        self.print_acc()

    def collect_samples(self, data_name):
        assert 'src' in data_name or 'tgt' in data_name
        if 'src' in data_name:
            domain = 0
        else:
            domain = 1

        self.model.eval()
        with torch.no_grad():

            sample_collection = {}
            sample_features = []
            sample_logits = []
            sample_true_labels = []

            for i, (batch_inputs,batch_true_labels,_) in enumerate(tqdm.tqdm(self.data_loader[data_name])):
                with torch.no_grad():
                    batch_inputs = batch_inputs.cuda()
                    batch_true_labels = batch_true_labels.cuda()
                    batch_end_points = self.model(batch_inputs)
                sample_features += [batch_end_points['features'].detach()]
                sample_logits += [batch_end_points['logits'].detach()]
                sample_true_labels += [batch_true_labels]

            sample_collection['features'] = torch.cat(sample_features, dim=0)
            sample_collection['logits'] = torch.cat(sample_logits, dim=0)
            sample_collection['true_labels'] = torch.cat(sample_true_labels, dim=0)

        return sample_collection

    def get_sample(self, data_name):
        try:
            (x_s, x_s_aug,x_s_aug2), labels_s, _ = next(self.data_iterator[data_name])
        except StopIteration:
            self.data_iterator[data_name] = iter(self.data_loader[data_name])
            (x_s, x_s_aug,x_s_aug2), labels_s, _ = next(self.data_iterator[data_name])
        except TypeError:
            assert self.data_loader[data_name] is None
            return None
        return (x_s, x_s_aug,x_s_aug2), labels_s

    def log_summary_writer(self):
        self.summary_writer.add_scalars('losses', self.losses_dict, global_step=self.iteration)
        self.summary_writer.add_scalars('accuracies', self.acc_dict, global_step=self.iteration)
        self.summary_writer.close()

    def log_image_writer(self):
        src_data = next(iter(self.data_loader['src_embed']))
        src_inputs, src_labels = src_data['image_1'].cuda(), src_data['true_label'].cuda()
        tgt_data = next(iter(self.data_loader['tgt_embed']))
        tgt_inputs, tgt_labels = tgt_data['image_1'].cuda(), tgt_data['true_label'].cuda()
        summary_write_fig(self.summary_writer, tag='Source predictions vs. true labels',
                          global_step=self.iteration,
                          model=self.model, images=src_inputs, labels=src_labels, domain=0)
        summary_write_fig(self.summary_writer, tag='Target predictions vs. true labels',
                          global_step=self.iteration,
                          model=self.model, images=tgt_inputs, labels=tgt_labels, domain=1)
        summary_write_proj(self.summary_writer, tag='features', global_step=self.iteration,
                           model=self.model,
                           src_train_loader=self.data_loader['src_embed'],
                           tgt_train_loader=self.data_loader['tgt_embed'],
                           num_samples=self.num_proj_samples)
        summary_write_proj(self.summary_writer, tag='logits', global_step=self.iteration,
                           model=self.model,
                           src_train_loader=self.data_loader['src_embed'],
                           tgt_train_loader=self.data_loader['tgt_embed'],
                           num_samples=self.num_proj_samples)

    def print_acc(self):
        # show the latest eval_result
        self.acc_dict['src_train_accuracy'] = self.src_train_acc_queue.get_average()
        self.total_progress_bar.write('Iteration {:6d}: '.format(self.iteration) + str(self.acc_dict))

    def save_checkpoint(self):
        # delete previous checkpoint
        prev_checkpoints_list = glob.glob(os.path.join(self.model_dir, '*.weights'))
        for prev_checkpoint in prev_checkpoints_list:
            try:
                os.remove(prev_checkpoint)
            except OSError:
                print('Error while deleting previous checkpoint weights.')

        # save new checkpoint weights
        checkpoint_weights = os.path.join(self.model_dir, 'checkpoint_%d_%d.weights' % (self.epoch, self.iteration))
        torch.save({'weights': self.model.state_dict()}, checkpoint_weights)

