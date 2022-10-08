import argparse
import csv
import json
import os
import shutil
import copy

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from lr_schedule import InvScheduler
from model.contrastive_loss import supervised_loss,InfoNCELoss,GaussianKernel
from model.key_memory import KeyMemory

from model.model import ImageClassifier

from pseudo_labeler import KMeansPseudoLabeler
from train_covid import Train
from utils import configure, get_dataset_name, moment_update, str2bool,SupConLoss
import utils
parser = argparse.ArgumentParser()

# dataset configurations
parser.add_argument('--config',
                    type=str,
                    default='config/config.yml',
                    help='Dataset configuration parameters')

# datapath
parser.add_argument('--src', default='./all_data_pneumonia/', type=str, help='data path of source data')
parser.add_argument('--label_file_source', default='./COVID-DA-master/data/pneumonia_task_for_python3.pkl', type=str,
                    help='the pkl file that records the source data list and labels')
parser.add_argument('--tgt', default='./all_data_covid/', type=str, help='data path of target data')
parser.add_argument('--label_file_target', default='./COVID-DA-master/data/COVID-19_task_for_python3.pkl', type=str,
                    help='the pkl file that records the target data list and labels')

# training configurations
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size for both training and evaluation')
parser.add_argument('--eval_batch_size',
                    type=int,
                    default=32,
                    help='Batch size for both training and evaluation')
parser.add_argument('--pseudo_batch_size',
                    type=int,
                    default=4096,
                    help='Batch size for pseudo labeling')
parser.add_argument('--max_iterations',
                    type=int,
                    default=20000,
                    help='Maximum number of iterations')
parser.add_argument('--num_classes',
                    type=int,
                    default=2,
                    help='num of classes')

# logging configurations
parser.add_argument('--log_dir',
                    type=str,
                    default='logs',
                    help='Parent directory for log files')
parser.add_argument('--log_summary_interval',
                    type=int,
                    default=100,
                    help='Logging summaries frequency')
parser.add_argument('--log_image_interval',
                    type=int,
                    default=1000,
                    help='Logging images frequency')
parser.add_argument('--acc_file',
                    type=str,
                    default='hyper_search.csv',  # 'result.txt'
                    help='File where accuracies are wrote')

# resource configurations

parser.add_argument('--num_workers',
                    type=int,
                    default=1,
                    help='Number of workers')

# InfoNCE loss configurations
parser.add_argument('--temperature',
                    type=float,
                    default=0.07,
                    help='Temperature parameter for InfoNCE loss')

# hyper-parameters
parser.add_argument('--cw',
                    type=float,
                    default=1,
                    help='Weight for NCE contrast loss')

parser.add_argument('--max_key_size',
                    type=int,
                    default=200,
                    help='Maximum number of key feature size computed in the model')
parser.add_argument('--min_conf_samples',
                    type=int,
                    default=1,
                    help='Minimum number of samples per confident target class')
parser.add_argument('--kcc',
                    type=int,
                    default=3,
                    help='the lcc')

# model configurations
parser.add_argument('--network',
                    type=str,
                    default='resnet18',  # resnet101
                    help='Base network architecture')
parser.add_argument('--contrast_dim',
                    type=int,
                    default=128,
                    help='contrast layer dimension')
parser.add_argument('--alpha',
                    type=float,
                    default=0.9,
                    help='momentum coefficient for model ema')

# optimizer configurations
parser.add_argument('--optimizer',
                    type=str,
                    default='sgd',
                    help='Optimizer type')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='Initial learning rate')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='Optimizer parameter, momentum')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0005,
                    help='Optimizer parameter, weight decay')
parser.add_argument('--nesterov',
                    type=str2bool,
                    default=False,  # True
                    help='Optimizer parameter, nesterov')

# learning rate scheduler configurations
parser.add_argument('--lr_scheduler',
                    type=str,
                    default='inv',
                    help='Learning rate scheduler type')
parser.add_argument('--gamma',
                    type=float,
                    default=0.001,
                    help='Inv learning rate scheduler parameter, gamma')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.75,  # 
                    help='Inv learning rate scheduler parameter, decay rate')
parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
parser.add_argument("--module", type=str, default='contrastive_loss', choices=['contrastive_loss', 'source_only'],
                        help="When module is 'contrastive_loss', it is our method.")
    
def main():
    args = parser.parse_args()
    print(args)
    config = configure(args.config)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    # define model name
    setup_list = ['covid19',
        args.module,
        args.network,
        f"contrast_dim_{args.contrast_dim}",
        f"maxiter_{args.max_iterations}",
        f"batchsize_{args.batch_size}",
        f"augment_{args.cw}",
        f"kcc_{args.kcc}"
    ]
    model_name = "_".join(setup_list)
    print(colored(f"Model name: {model_name}", 'green'))
    model_dir = os.path.join(args.log_dir, model_name)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    summary_writer = SummaryWriter(model_dir)

    # save parsed arguments
    with open(os.path.join(model_dir, 'parsed_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    criterion = SupConLoss(temperature=0.07)

    backbone = utils.get_model(args.network)
    pool_layer =  None
    model = ImageClassifier(backbone, args.num_classes, bottleneck_dim=args.contrast_dim,
                                 pool_layer=pool_layer).cuda()
    backbone_ema = utils.get_model(args.network)                             
    model_ema = ImageClassifier(backbone_ema, args.num_classes, bottleneck_dim=args.contrast_dim,
                                 pool_layer=pool_layer).cuda()

    moment_update(model, model_ema, 0)

    model = model.cuda()
    model_ema = model_ema.cuda()

    contrast_loss = InfoNCELoss(temperature=args.temperature).cuda()
    max_key_size=args.max_key_size*args.num_classes
    src_memory = KeyMemory(max_key_size, args.contrast_dim,args.num_classes).cuda()
    tgt_memory = KeyMemory(max_key_size, args.contrast_dim,args.num_classes).cuda()

    tgt_pseudo_labeler = KMeansPseudoLabeler(num_classes=args.num_classes,
                                             batch_size=args.pseudo_batch_size)

    parameters = model.get_parameter_list()
    group_ratios = [parameter['lr'] for parameter in parameters]

    optimizer = torch.optim.SGD(parameters,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    assert args.lr_scheduler == 'inv'
    lr_scheduler = InvScheduler(gamma=args.gamma,
                                decay_rate=args.decay_rate,
                                group_ratios=group_ratios,
                                init_lr=args.lr)
    
    trainer = Train(model, model_ema, optimizer, lr_scheduler, model_dir,
                    summary_writer, args.src, args.tgt, args.label_file_source,args.label_file_target,contrast_loss,supervised_loss,src_memory, tgt_memory, tgt_pseudo_labeler,criterion,
                    cw=args.cw,
                    min_conf_samples=args.min_conf_samples,
                    num_classes=args.num_classes,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    max_iter=args.max_iterations,
                    iters_per_epoch=100,
                    log_summary_interval=args.log_summary_interval,
                    log_image_interval=args.log_image_interval,
                    acc_metric='total_mean',
                    alpha=args.alpha,module=args.module,kcc=args.kcc)

    tgt_best_acc = trainer.train()

    # write to text file
    # with open(args.acc_file, 'a') as f:
    #     f.write(model_name + '     ' + str(tgt_best_acc) + '\n')
    #     f.close()

    # write to xlsx file
    write_list = [
        args.src,
        args.tgt,
        args.network,
        args.contrast_dim,
        args.temperature,
        args.alpha,
        args.cw,
        args.max_key_size,
        args.module,
        args.batch_size,
        args.min_conf_samples,
        args.gpu,
        tgt_best_acc
    ]
    with open(args.acc_file, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(write_list)
if __name__ == '__main__':
    main()
