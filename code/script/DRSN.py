import random
import time
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from raw_model.DRSN_CS.DRSN_CS import get_model
from plugins import eraser, utils
from dataset import dataset_hdf5_DRSN
from data.NEU_POWDER.POWDER_HDF5 import device_list as POWDER_device_list
from data.torchsig_HackRF.TORCHSIG_HDF5 import device_list as TORCHSIG_device_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args:argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    global era, cru
    cru = era.crumble[args.cru_name+args.time]
    print(args)
    args.inner_train = False
   
    '''
    DATA
    '''
    # data preparation
    if "bes" in args.class_names:
        raw_dataset = dataset_hdf5_DRSN.POWDER_Dataset_HDF5(hdf5_file=args.hdf5_file, 
                                                        standards=args.standard_train, days=args.day_train, recordings=args.recording_train, devices = args.device_train)
    if "device_0" in args.class_names:
        raw_dataset = dataset_hdf5_DRSN.TORCHSIG_Dataset_HDF5(hdf5_file=args.hdf5_file, 
                                                        modulations=args.modulation_train, devices = args.device_train)
    
    split_rate = args.split_rate if args.inner_train else 1.0
    train_size = int(split_rate * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    torch.manual_seed(args.seed_split)
    
    # train data 
    if "bes" in args.class_names:
        # POWDER TRAIN
        train_pack = [(random_split(raw_dataset, [train_size, test_size])[0], args.standard_train, args.day_train, args.recording_train), ]
    if "device_0" in args.class_names:
        # TORCHSIG TRAIN
        train_pack = [(random_split(raw_dataset, [train_size, test_size])[0], args.modulation_train), ]
    
    # test data
    if args.inner_train:
        if "bes" in args.class_names:
            test_pack = [(random_split(raw_dataset, [train_size, test_size])[1], args.standard_test, args.day_test, args.recording_test), ]
        if "device_0" in args.class_names:
            test_pack = [(random_split(raw_dataset, [train_size, test_size])[1], args.modulation_test), ]
    else: 
        test_pack = list()
        if "bes" in args.class_names:
            # POWDER TEST
            for standard in args.standard_test: 
                for day in args.day_test:
                    for recording in args.recording_test:
                        test_pack.append((dataset_hdf5_DRSN.POWDER_Dataset_HDF5(hdf5_file=args.hdf5_file,
                                                                                standards = [standard, ], days = [day, ],
                                                                                devices = args.device_test, recordings = [recording, ]), standard, day, recording, ))
        if "device_0" in args.class_names:
            # TORCHSIG TEST            
            for modulation in args.modulation_test:
                test_pack.append((dataset_hdf5_DRSN.TORCHSIG_Dataset_HDF5(hdf5_file=args.hdf5_file,
                                                                        modulations = [modulation, ], devices = args.device_test), modulation, ))
  
    # make dataloader
    for (data, *nargs) in train_pack:
        print("train args: {0}, number: {1}".format(nargs, len(data)))
    for (data, *nargs) in test_pack:
        print("test args: {0}, number: {1}".format(nargs, len(data)))
    train_loader = DataLoader(train_pack[0][0], batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader_pack = list()
    for (test_dataset, *nargs) in test_pack:
        test_loader_pack.append((DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers), nargs))
    train_source_iter = ForeverDataIterator(train_loader)

    '''
    MODEL
    '''
    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = get_model(args.arch, num_classes=len(args.class_names))# DRSN-CS 18layers or 34 layers
    classifier = backbone.to(device)

    # define optimizer and lr scheduler
    optimizer_warm = Adam(classifier.parameters(), args.lr)
    optimizer = Adam(classifier.parameters(), args.lr)
    lr_scheduler_warm = LinearLR(optimizer_warm, start_factor=1/args.warm_epoch, end_factor=1, total_iters=args.warm_epoch)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch%4+1)*0.15)
    
    # seed training
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    '''
    IF TEST, TEST ONLY
    '''
    if args.phase == 'test':
        # test
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = utils.validate(test_loader, classifier, args, device, tip="{1}".format(nargs))
            cru["{0}".format(nargs)]["test"] = acc1
            era.save(args.cru_name+args.time)
        return
        
    '''
    TRAIN
    '''
    # warming up
    for epoch in tqdm(range(args.warm_epoch), desc="WARMING UP  "):
        print("\nlr:", lr_scheduler_warm.get_last_lr()[0])
        # train for some epoch
        train(train_source_iter, None, classifier, None, optimizer_warm,
              lr_scheduler_warm, epoch, args, phase = "warm")
        # save latest checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # evaluate on validation set
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = round(utils.validate(test_loader, classifier, args, device, tip="{0}".format(nargs)), 4)
            cru["{0}".format(nargs)][epoch] = acc1
            era.save(args.cru_name+args.time)
    
    # start training
    start=time.time()
    acc_list = [[0, ] for i in range(len(test_pack))]
    for epoch in tqdm(range(args.epochs), desc="EPOCH LOOP  "):
        print("\nlr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, None, classifier, None, optimizer,
              lr_scheduler, epoch, args)
        # save latest checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # evaluate on validation set
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = round(utils.validate(test_loader, classifier, args, device, tip="{0}".format(nargs)), 4)
            acc_list[i].append(acc1)
            cru["{0}".format(nargs)][epoch+args.warm_epoch] = acc1
            era.save(args.cru_name+args.time)
            # remember best acc@1 and save best checkpoint
            if acc1 == max(acc_list[i]):
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        
    print("trainning time cost: {}".format(time.time()-start))
    for i, (_, *nargs) in enumerate(test_loader_pack):
        print("best_acc of {0}: {1}".format(nargs, max(acc_list[i])))
    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, phase = "train"):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch)) if phase == "train" else ProgressMeter(
        args.warm_iter,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch if phase == "train" else args.warm_iter):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device).float()
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y = model(x_s)

        # loss calculation
        cls_loss = F.cross_entropy(y, labels_s)
        loss = cls_loss
        cls_acc = accuracy(y, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    lr_scheduler.step()
    # global cru
    global era, cru
    cru = era.crumble[args.cru_name+args.time]
    cru["loss"][epoch] = loss.item()
    era.save(args.cru_name+args.time)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRSN')
    # csv logger
    parser.add_argument("--era_name", type=str, default="DRSN")    
    parser.add_argument("--cru_name", type=str, default="test")    
    parser.add_argument("--time", type=str, default=eraser.get_time())    
    
    # dataset parameters
    parser.add_argument('--hdf5_file', default="/home/zhangyezhuo/modulation_attack/data/NEU_POWDER/POWDER_DATASET.hdf5", 
                        help='POWDER hdf5 dataset path. ')   
    parser.add_argument('--class_names', default = "POWDER_device_list", choices=["POWDER_device_list", "TORCHSIG_device_list"], 
                        help='device name list. ')  
    
    parser.add_argument('-in', '--inner_train', type=bool, default=False, choices=[True, False],
                        help='whether train inside a setting. ')  
    parser.add_argument('--split_rate', type=float, default=0.8, 
                        help='training rate. ')  
    parser.add_argument('--seed_split', type=int, default=114514, 
                        help='seed for dataset spliting. ')  
    
    parser.add_argument('-stdtrain', '--standard_train', default=["4G", ], 
                        help='standard for training. ', nargs = "+")   
    parser.add_argument('-stdtest', '--standard_test', default=["4G", ], 
                        help='standard for testing. ', nargs = "+")   
    
    parser.add_argument('-modtrain', '--modulation_train', default=["ook", ], 
                        help='standard for training. ', nargs = "+")   
    parser.add_argument('-modtest', '--modulation_test', default=["ook", ], 
                        help='standard for testing. ', nargs = "+") 
    
    parser.add_argument('-daytrain', '--day_train', default=["Day_1", ], 
                        help='day for training. ', nargs = "+")   
    parser.add_argument('-daytest', '--day_test', default=["Day_1", ], 
                        help='day for testing. ', nargs = "+")   
    
    parser.add_argument('-devtrain', '--device_train', default=["bes", "browning", "honors", "meb"], 
                        help='device for training. ', nargs = "+")   
    parser.add_argument('-devtest', '--device_test', default=["bes", "browning", "honors", "meb"], 
                        help='device for testing. ', nargs = "+")   
    
    parser.add_argument('-rectrain', '--recording_train', default=["s1", ], 
                        help='recording for training. ', nargs = "+")   
    parser.add_argument('-rectest', '--recording_test', default=["s2", ], 
                        help='recording for testing. ', nargs = "+")   
    
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18', 'resnet34',])
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    
    parser.add_argument('--warm_epoch', default=30, type=int,
                        help='warm up epoch')
    parser.add_argument('--warm_iter', default=100, type=int,
                        help='Number of iterations per epoch for warming up')
    
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true', default=True, 
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('-l', "--log", type=str, default='drsn',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('-ph', "--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")    
    args = parser.parse_args()
    
    # eraser读写
    args.class_names = eval(args.class_names)  
    era=eraser.Eraser(args.log, name=args.era_name)
    if not os.path.exists(args.log+"{}.pkl".format(args.era_name)): era.save()
    era.load(args.log+"{}.pkl".format(args.era_name))
    era.add_crumble(args.cru_name+args.time)
    cru = era.crumble[args.cru_name+args.time]
    
    main(args)
