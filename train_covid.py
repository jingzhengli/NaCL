import os
import glob
import torch
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
from data.prepare_data import generate_dataloader_sc
from data.dataset import penuDataset, covidDataset_target, covid_target
from utils import summary_write_proj, summary_write_fig, AvgMeter, moment_update, set_bn_train, compute_accuracy,calc_knn_graph, calc_topo_weights_with_components_idx
from scipy.stats import mode
import torchvision.transforms as transforms
# from preprocess import transform as T
# from preprocess.randaugment import RandomAugment
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
import copy
# from tqdm import tqdm

class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, model_dir,
                 summary_writer, src, tgt, label_file_source, label_file_target, contrast_loss,supervised_loss,src_memory, tgt_memory, tgt_pseudo_labeler,criterion,
                 cw=1.0,
                 min_conf_samples=3,
                 num_classes=2,
                 batch_size=36,
                 eval_batch_size=36,
                 num_workers=1,
                 max_iter=100000,
                 iters_per_epoch=100,
                 log_summary_interval=10,
                 log_image_interval=1000,
                 acc_metric='total_mean',
                 alpha=0.99,module='contrastive_only',kcc=1):
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_dir = model_dir
        self.summary_writer = summary_writer
        self.source = src
        self.target = tgt
        self.label_file_source =label_file_source
        self.label_file_target = label_file_target
        self.contrast_loss = contrast_loss
        self.supervised_loss=supervised_loss
        self.src_memory = src_memory
        self.tgt_memory = tgt_memory
        self.tgt_pseudo_labeler = tgt_pseudo_labeler
        self.cw = cw
        self.min_conf_samples = min_conf_samples
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.iters_per_epoch = iters_per_epoch
        self.log_summary_interval = log_summary_interval
        self.log_image_interval = log_image_interval
        self.acc_metric = acc_metric
        self.alpha = alpha
        self.module = module
        self.big_comp = set()
        self.k_cc = kcc
        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(desc='Iterations', total=self.max_iter, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.acc_dict = {'tgt_best_test_acc': 0.0}
        self.src_train_acc_queue = AvgMeter(maxsize=100)
        # self.tgt_train_acc_queue = AvgMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        self.tgt_conf_pair = None
        # self.tgt_non_conf_indices = list(range(self.tgt_size))
        self.data_loader = {}
        self.data_iterator = {}

        self.source_train_cluster = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4859, 0.4859, 0.4859), (0.0820, 0.0820, 0.0820)),  # grayscale mean/std
        ])
        self.target_train_cluster = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5002, 0.5002, 0.5002), (0.0893, 0.0893, 0.0893)),  # grayscale mean/std
        ])
        # self.source_train = T.Compose([
        #     T.Resize((256, 256)),
        #     T.PadandRandomCrop(border=4, cropsize=(224, 224)),
        #     T.RandomHorizontalFlip(p=0.5),
        #     RandomAugment(2, 10),
        #     T.Normalize((0.4859, 0.4859, 0.4859), (0.0820, 0.0820, 0.0820)),
        #     T.ToTensor(),
        # ])

        self.source_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize((0.4859, 0.4859, 0.4859), (0.0820, 0.0820, 0.0820)),
        ])  

        self.target_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize((0.5002, 0.5002, 0.5002), (0.0893, 0.0893, 0.0893)),
        ])  

        dataset_source = penuDataset(self.source, self.label_file_source, train=True, transform=self.source_train)

        import numpy as np
        from torch.utils.data import WeightedRandomSampler
        cluster_class,counts = np.unique(dataset_source.train_labels,return_counts=True)
        print("unique labels:{}".format(counts))
        print("cluster_class_length:{}".format(len(cluster_class))) 
        class_weights = [1]*self.num_classes 
        for c in range(len(cluster_class)):
            class_weights[cluster_class[c]] = sum(counts) / counts[c] 
        # class_weights = [sum(counts) / c for c in counts]
        example_weights = [class_weights[e] for e in dataset_source.train_labels]
        sampler = WeightedRandomSampler(example_weights,len(dataset_source.train_labels),replacement=True)
        self.data_loader['src_train'] = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=self.batch_size,
            sampler = sampler,
            drop_last=True,
            num_workers=2
        )
        # load unlabeled target domain data
        self.data_loader['tgt_train'] = covidDataset_target(self.target, self.label_file_target, train=True, transform=self.target_train)
        self.data_loader['tgt_train_cluster'] = covidDataset_target(self.target, self.label_file_target, train=True, transform=self.target_train_cluster)
        self.data_loader['tgt_train_unlabeled'] = torch.utils.data.DataLoader(
            dataset=self.data_loader['tgt_train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        self.data_loader['tgt_train_unlabeled_cluster'] = torch.utils.data.DataLoader(
            dataset=self.data_loader['tgt_train_cluster'],
            batch_size=1000,
            shuffle=False,
            num_workers=2
        )
        # load labeled target domain data
        dataset_target_labeled = covidDataset_target(self.target, self.label_file_target, semi=True, train=True,
                                                    transform=self.target_train)
        dataset_target_labeled_cluster = covidDataset_target(self.target, self.label_file_target, semi=True, train=True,
                                                    transform=self.target_train_cluster)
        self.data_loader['tgt_train_labeled'] = torch.utils.data.DataLoader(
            dataset=dataset_target_labeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        self.data_loader['tgt_train_labeled_cluster'] = torch.utils.data.DataLoader(
            dataset=dataset_target_labeled_cluster,
            batch_size=1000,
            shuffle=False,
            num_workers=2
        )
        test_dataset = covid_target(self.target, self.label_file_target, train=False, transform=self.target_train_cluster)
        self.data_loader['test_dataset'] = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        for key in self.data_loader:
            if self.data_loader[key] is not None and key is not 'tgt_train' and key is not 'tgt_train_cluster':
                self.data_iterator[key] = iter(self.data_loader[key])
            else:
                self.data_iterator[key] = None
        self.ntrain = len(self.data_loader['tgt_train_unlabeled_cluster'].dataset)
        self.k_outlier = self.ntrain % (5*self.num_classes)
    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            self.data_iterator['tgt_train'] = covidDataset_target(self.target, self.label_file_target, train=True, transform=self.target_train)
            self.prepare_tgt_conf_dataset()
            # train an epoch
            self.train_epoch()

        self.total_progress_bar.write('Finish training')
        return self.acc_dict['tgt_best_test_acc']

    def train_epoch(self):
        cur_epoch_steps = 100
        for _ in tqdm.tqdm(range(cur_epoch_steps), desc='Epoch {:4d}'.format(self.epoch), leave=False, ascii=True):
            self.model.train()
            self.model_ema.eval()
            self.model_ema.apply(set_bn_train)

            self.train_step()

            if self.iteration % self.log_summary_interval == 0:
                self.log_summary_writer()
        self.epoch += 1
        
    def train_step(self):
        # update the learning rate of the optimizer
        self.lr_scheduler.adjust_learning_rate(self.optimizer, self.iteration)
        self.optimizer.zero_grad()

        # prepare source batch
        (src_inputs_1), src_labels = self.get_sample('src_train')
        (tgt_inputs_1), tgt_labels = self.get_sample('tgt_train_labeled')

        src_inputs_1,src_labels \
            = src_inputs_1.cuda(), src_labels.cuda()
        tgt_inputs_1,tgt_labels \
            = tgt_inputs_1.cuda(), tgt_labels.cuda()
        imgs_s = torch.cat([src_inputs_1, tgt_inputs_1], dim=0).cuda()
        all_labels = torch.cat([src_labels, tgt_labels], dim=0).cuda()

        # model inference
        src_end_points = self.model(imgs_s)
        stu = src_inputs_1.size(0)
        #supervised loss
        self.src_supervised_step(src_end_points['logits'], all_labels)

        # update key memory
        with torch.no_grad(): 
            src_end_points_ema = self.model_ema(imgs_s)
            self.src_memory.store_keys(src_end_points_ema['contrast_features'], all_labels)

        if self.data_loader['tgt_conf'] is not None:
            (tgt_input), tgt_pseudo_labels  = self.get_sample('tgt_conf')
            # tgt_inputs, tgt_pseudo_labels = tgt_data['image'].cuda(), tgt_data['pseudo_label'].cuda()
            tgt_input, tgt_pseudo_labels \
                = tgt_input.cuda(), tgt_pseudo_labels.cuda()#torch.stack(tgt_data['pseudo_label'],1).cuda()
            btu = tgt_input.size(0)

            # model inference

            tgt_end_points = self.model(tgt_input)

            # update key memory
            with torch.no_grad():
                tgt_end_points_ema = self.model_ema(tgt_input)
                self.tgt_memory.store_keys(tgt_end_points_ema['contrast_features'], tgt_pseudo_labels)
            # class contrastive alignment
            self.contrastive_step(src_end_points, stu,src_labels, tgt_end_points['contrast_features'], tgt_pseudo_labels)
        else:
            self.losses_dict['contrast_loss'] = 0.

        if self.module == 'contrastive_loss':
            # warm-up
            if self.epoch >10:
                self.losses_dict['total_loss'] = \
                self.losses_dict['src_classification_loss'] + self.losses_dict['contrast_loss']
            else:
                self.losses_dict['total_loss'] = self.losses_dict['src_classification_loss']

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
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            self.epoch = np.clip(self.epoch, 0.0, rampup_length)
            phase = 1.0 - self.epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def prepare_tgt_conf_dataset(self):
        
        # collect target samples and source samples
        src_test_collection = self.collect_samples('tgt_train_labeled_cluster')
        tgt_test_collection = self.collect_samples('tgt_train_unlabeled_cluster')
        test_collection = self.collect_samples('test_dataset')
        tgt_pseudo_probabilities = self.tgt_pseudo_labeler.pseudo_label_tgt(src_test_collection, tgt_test_collection) #probs = torch.softmax(tgt_pseudo_probabilities, dim=1)  
        tgt_pseudo_acc = compute_accuracy(tgt_pseudo_probabilities, tgt_test_collection['true_labels'],
                                          acc_metric=self.acc_metric, print_result=False)
        self.acc_dict['tgt_pseudo_acc'] = tgt_pseudo_acc
        self.eval_tgt(test_collection)
        # compute the pseudo labels and the confidentce of target data
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
        # tgt_test_collection['features']
        # tgt_test_collection['true_labels']
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
        print("=> number of cur_big_comp and all samples  %d and %d" %(len(cur_big_comp),len(self.data_loader['tgt_train_unlabeled'].dataset)))
        print("=> number of big_comp and all samples  %d and %d" %(len(big_comp),len(self.data_loader['tgt_train_unlabeled'].dataset)))
        print(">> The accuracy of LCC: {}".format(np.sum(tgt_test_collection['true_labels'][cur_big_comp].cpu().numpy() == tgt_pseudo_labels[cur_big_comp].numpy())
                                                        / float(len(cur_big_comp))), 'red')                                                
        # --- remove outliers in largest connected component ---
        big_com_idx = list(big_comp)

        feats_big_comp = tgt_test_collection['features'][big_com_idx]
        labels_big_comp = np.array(train_gt_labels)[big_com_idx]

        knnG_list = calc_knn_graph(feats_big_comp, k=5)

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
        # noisy_data_indices = list(set(range(self.ntrain)) - big_comp)
        selected_pseudo_probabilities = [tgt_pseudo_labels[index] for index in big_comp]

        conf_images = [self.data_iterator['tgt_train'].train_data[index] for index in big_comp]
        self.data_iterator['tgt_train'].train_data = conf_images
        self.data_iterator['tgt_train'].train_labels = selected_pseudo_probabilities
        cluster_class,counts = np.unique(self.data_iterator['tgt_train'].train_labels,return_counts=True)
        print("unique labels:{}".format(counts))
        print("cluster_class_length:{}".format(len(cluster_class))) 

        self.data_loader['tgt_conf'] = torch.utils.data.DataLoader(
            self.data_iterator['tgt_train'], batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True
        )
        if self.data_loader['tgt_conf'] is None:
            self.data_iterator['tgt_conf'] = None
        else:
            self.data_iterator['tgt_conf'] = iter(self.data_loader['tgt_conf'])
        
        print("=> number of selected samples and all samples  %d and %d" %(len(self.data_loader['tgt_conf'].dataset),len(self.data_loader['tgt_train_unlabeled'].dataset)))

    def eval_tgt(self, test_collection):
        tgt_test_acc = compute_accuracy(test_collection['logits'], test_collection['true_labels'],
                                        acc_metric=self.acc_metric, print_result=True)
        tgt_test_acc = round(tgt_test_acc, 3)
        self.acc_dict['tgt_test_acc'] = tgt_test_acc

        p_list=copy.deepcopy(F.softmax(test_collection['logits'], dim=1).cpu().numpy()[:,1])
        t_list=copy.deepcopy(test_collection['true_labels'].cpu().numpy())
        self.acc_dict['AUROC'] = metrics.ranking.roc_auc_score(t_list, p_list)
        # logger.info('AUROC: %.4f' % AUROC)

        p_list[p_list >= 0.5] = 1
        p_list[p_list < 0.5] = 0

        t_open, f_narrow, f_open, t_narrow = metrics.confusion_matrix(t_list, p_list).ravel()
        self.acc_dict['F1'] = f1_score(t_list, p_list)
        self.acc_dict['accuracy'] = (t_narrow+t_open) / (t_narrow+t_open+f_narrow+f_open)
        self.acc_dict['precision'] = t_narrow / (t_narrow+f_narrow)
        self.acc_dict['recall'] = t_narrow / (t_narrow+f_open)
        # self.acc_dict['tgt_best_test_acc'] = max(self.acc_dict['tgt_best_test_acc'], tgt_test_acc)
        if self.acc_dict['tgt_test_acc'] > self.acc_dict['tgt_best_test_acc']:
            self.acc_dict['tgt_best_test_acc'] = self.acc_dict['tgt_test_acc']
            self.save_checkpoint()
        self.print_acc()

    def collect_samples(self, data_name):
        # assert 'src' in data_name or 'tgt' in data_name

        self.model_ema.eval()
        with torch.no_grad():
            sample_collection = {}
            sample_features = []
            sample_logits = []
            sample_true_labels = []

            # for sample_data in tqdm.tqdm(self.data_loader[data_name], desc=data_name, leave=False, ascii=True):
            for i, (batch_inputs,batch_true_labels) in enumerate(self.data_loader[data_name]):
                with torch.no_grad():
                    batch_inputs = batch_inputs.cuda()
                    batch_true_labels = batch_true_labels.cuda()
                    batch_end_points = self.model_ema(batch_inputs)
                sample_features += [batch_end_points['features'].detach()]
                # sample_features += [batch_end_points['contrast_features']]
                sample_logits += [batch_end_points['logits'].detach()]
                sample_true_labels += [batch_true_labels]

            sample_collection['features'] = torch.cat(sample_features, dim=0)
            sample_collection['logits'] = torch.cat(sample_logits, dim=0)
            sample_collection['true_labels'] = torch.cat(sample_true_labels, dim=0)

        return sample_collection

    def get_sample(self, data_name):
        try:
            (x_s), labels_s= next(self.data_iterator[data_name])
        except StopIteration:
            # if data_name == 'src_train' or data_name == 'tgt_conf':
            #     self.data_loader[data_name].construct_data_loader()
            self.data_iterator[data_name] = iter(self.data_loader[data_name])
            (x_s), labels_s = next(self.data_iterator[data_name])
        except TypeError:
            assert self.data_loader[data_name] is None
            return None
        return (x_s), labels_s

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
        # self.acc_dict['tgt_train_accuracy'] = self.tgt_train_acc_queue.get_average()
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

