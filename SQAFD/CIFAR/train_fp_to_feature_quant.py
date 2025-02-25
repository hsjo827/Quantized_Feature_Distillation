import argparse
import logging
import os
import random
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn

import utils

from models.custom_models_resnet import *
from models.custom_models_vgg import * 
from models.feature_quant_module import *

from utils import *
import utils
from utils import printRed

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders


start_time = time.time()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")

# data and model
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--arch', type=str, default='resnet20_fp', help='model architecture')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
parser.add_argument('--lr_m', type=float, default=1e-3, help='learning rate for model parameters')
parser.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--lr_q_end', type=float, default=0.0, help='final learning rate for quantizer parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--decay_schedule_q', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--lr_scheduler_q', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

# arguments for quantization
parser.add_argument('--QWeightFlag', type=str2bool, default=False, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=False, help='do activation quantization')
parser.add_argument('--baseline', type=str2bool, default=True, help='training with STE')
parser.add_argument('--bkwd_scaling_factorF', type=float, default=0.0, help='scaling factor for feature')
parser.add_argument('--use_hessian', type=str2bool, default=False, help='update scsaling factor using Hessian trace')
parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')
parser.add_argument('--quan_method', type=str, default='EWGS', help='training with different quantization methods')


parser.add_argument('--train_mode', type=str, default='teacher', choices=['fp', 'teacher', 'student'], help='training mode: fp model training / teacher model training / knowledge distillation')
parser.add_argument('--QFeatureFlag', type=str2bool, default=True, help='add feature quantizer to model')
parser.add_argument('--feature_levels', type=int, default=2, help='number of feature quantization levels')
parser.add_argument('--train_last_conv', type=str2bool, default=False, help='train last convolution layer of teacher model')
parser.add_argument('--use_student_quant_params', type=str2bool, default=False, help='Enable the use of student quantization parameters during teacher quantization')
parser.add_argument('--use_adapter', type=str2bool, default=False, help='Enable the use of adapter(connector) for Student')
parser.add_argument('--use_map_norm', type=str2bool, default=False, help='Enable the use of feature normalization')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/Teacher_Quant/T1bit_EWGS/')
parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='./results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth', help='path for pretrained full-preicion model')

args = parser.parse_args()
arg_dict = vars(args)


### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)

else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)


model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)

num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path)
    current_dict = model.state_dict()
    printRed("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            #print(key)
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)
else:
    printRed("Not initialized by the pretrained full precision weights")

# initialize quantizer params
init_quant_model(model, train_loader, device)

if args.quan_method == "EWGS" or args.baseline:
    define_quantizer_scheduler = True
else:
    define_quantizer_scheduler = False
    
if args.quan_method == "EWGS" or args.baseline:
    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []
    for m in model.modules():
        if isinstance(m, FeatureQuantizer):
            quant_params.append(m.lF)
            quant_params.append(m.uF)
            quant_params.append(m.output_scale)
            print("FeatureQuantizer", m)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            print("nn", m)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                model_params.append(m.weight)
                model_params.append(m.bias)
    
    # quantization 파라미터만 학습
    for param in model_params:
        param.requires_grad = False
    for param in quant_params:
        param.requires_grad = True
        
    print("# total params:", sum(p.numel() for p in trainable_params))
    print("# model params:", sum(p.numel() for p in model_params))
    print("# quantizer params:", sum(p.numel() for p in quant_params))
    logging.info("# total params: {}".format(sum(p.numel() for p in trainable_params)))
    logging.info("# model params: {}".format(sum(p.numel() for p in model_params)))
    logging.info("# quantizer params: {}".format(sum(p.numel() for p in quant_params)))
    if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
        raise Exception('Mismatched number of trainable parmas')
else:
    raise NotImplementedError(f"Not implement {args.quan_method}!")


### initialize optimizer, scheduler, loss function
if define_quantizer_scheduler:
    # optimizer for quantizer params
    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    # scheduler for quantizer params
    if args.lr_scheduler_q == "step":
        if args.decay_schedule_q is not None:
            milestones_q = list(map(lambda x: int(x), args.decay_schedule_q.split('-')))
        else:
            milestones_q = [args.epochs+1]
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones_q, gamma=args.gamma)
    elif args.lr_scheduler_q == "cosine":
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)

# optimizer for model params 
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model.parameters(), lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model.parameters(), lr=args.lr_m, weight_decay=args.weight_decay)
    
# scheduler for model params
if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)

### train
total_iter = 0
best_acc = 0
acc_last5 = []
lambda_dict = {}
for ep in range(args.epochs):
    model.train()
    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    
    ### update grad scales
    if ep % args.update_every == 0 and ep != 0 and not args.baseline and args.use_hessian:
        update_grad_scales(model, train_loader, criterion, device, args)
        print("update grade scales")
    
    if define_quantizer_scheduler:
        writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], ep)
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_m.zero_grad()
        if define_quantizer_scheduler:
            optimizer_q.zero_grad()

        if args.quan_method == "EWGS":
            save_dict = {"iteration": total_iter, "writer": writer, "layer_num": None, "block_num": None, "conv_num": None, "type": None}
            # for lambda
            if total_iter >= 2:
                for i in range(total_iter-1):
                    lambda_dict[f"{i}"] = {}
            lambda_dict[f"{total_iter}"] = {}
        else:
            save_dict = None
            lambda_dict = None
            
        pred = model(images, save_dict, lambda_dict)
        loss_total = criterion(pred, labels)
        
        loss = loss_total
        loss.backward()
        
        optimizer_m.step()
        if define_quantizer_scheduler:
            optimizer_q.step()
            
        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1
    
    scheduler_m.step()
    if define_quantizer_scheduler:
        scheduler_q.step()

    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
        logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
        writer.add_scalar('test/acc', test_acc, ep)

        torch.save({
            'epoch':ep,
            'model':model.state_dict(),
            'test_acc': test_acc,
            'optimizer_m':optimizer_m.state_dict(),
            'scheduler_m':scheduler_m.state_dict(),
            'optimizer_q':optimizer_q.state_dict() if define_quantizer_scheduler else {},
            'scheduler_q':scheduler_q.state_dict() if define_quantizer_scheduler else {},
            'criterion':criterion.state_dict()
        }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'test_acc': test_acc,
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'optimizer_q':optimizer_q.state_dict() if define_quantizer_scheduler else {},
                'scheduler_q':scheduler_q.state_dict() if define_quantizer_scheduler else {},
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))

        # for record the average acccuracy of the last 5 epochs
        if ep >= args.epochs - 5:
            acc_last5.append(test_acc)
            
    for m in model.modules():
        if isinstance(m, FeatureQuantizer):
            if args.train_mode == 'teacher':
                logging.info("lF: {}".format(m.lF))
                logging.info("uF: {}".format(m.uF))
                logging.info("grad_scaleF: {}".format(m.bkwd_scaling_factorF.item()))
                logging.info("output_scale: {}".format(m.output_scale))
            logging.info('\n')
    

checkpoint_path_last = os.path.join(args.log_dir, 'checkpoint/last_checkpoint.pth')
checkpoint_path_best = os.path.join(args.log_dir, 'checkpoint/best_checkpoint.pth')
utils.test_accuracy(checkpoint_path_last, model, logging, device, test_loader)
utils.test_accuracy(checkpoint_path_best, model, logging, device, test_loader)


print(f"Total time: {(time.time()-start_time)/3600}h")
logging.info(f"Total time: {(time.time()-start_time)/3600}h")

print(f"Save to {args.log_dir}")
logging.info(f"Save to {args.log_dir}")
