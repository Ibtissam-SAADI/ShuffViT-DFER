from __future__ import print_function

import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
from models import combmodel1
import utils
from KDEF import KDEF
from torch.autograd import Variable
from models.vgg import VGG
from models import SwinT
from models.mobilevit import mobilevit_s
from models.build import EfficientViT_M4
from models.efficientvit import EfficientViT
import timm
import itertools
import time
from vit_pytorch import SimpleViT
from torchvision import models
from models.EfficientFace import efficient_face
from lion_pytorch import Lion
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

# torch.manual_seed(42)

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='kmualign', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--mz', default='Adam', choices=['SGD', 'Adam', 'Lion'], type=str, help='optimizer')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# cut_size = 60


path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# Data
print('==> Preparing data..')
print(use_cuda)
transforms_vaild = torchvision.transforms.Compose([
                                     torchvision.transforms.ToPILImage(),
                                     torchvision.transforms.Resize((224,)),
                                     torchvision.transforms.ToTensor(),
                                     #torchvision.transforms.CenterCrop((96)),
                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                     #torchvision.transforms.Normalize((0.2274,), (0.2353,))
                                     ])

# For training data , we add some augmentation
transforms_train = torchvision.transforms.Compose([
                                      torchvision.transforms.ToPILImage(),
                                      torchvision.transforms.Resize((224,)),            
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.RandomRotation(40),
                                      torchvision.transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),
                                      torchvision.transforms.ToTensor(),
                                      #torchvision.transforms.Normalize((0.2274,), (0.2353,))
                                      torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                     ])
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_time = elapsed_time - elapsed_hours * 3600
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_hours, elapsed_mins, elapsed_secs
# Training

# Function to perform grid search
best_accuracy = 0.0
best_model = None
best_batch_size = None
best_lr = None
best_mizer = None
best_epoch = None

batch_sizes = [128, 48,16, 32]
learning_rates = [0.0001, 0.001, 0.01]
optimizers_list = ['SGD', 'Adam', 'Lion']
total_epoch = [160]


early_stopping_threshold = 10  # Number of epochs without improvement before stopping
early_stopping_counter = 0  # Counter to track epochs without improvement
best_Test_acc = 0  # Initialize the best test accuracy

# Perform grid search
best_accuracy = 0
best_params = {}

for bs, lr, mz, pch in itertools.product(batch_sizes, learning_rates, optimizers_list, total_epoch ):
    opt.bs = bs
    opt.lr = lr   
    opt.mz = mz
    opt.epoch = pch
               
             

    trainset = KDEF(split='Training', fold=opt.fold, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
    testset = KDEF(split='Testing', fold=opt.fold, transform=transforms_vaild)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs, shuffle=False, num_workers=0)

    net = None
    if opt.model == 'Ourmodel':
        
       num_classes = 7  # Adjust based on your task
       net  = combmodel1.CombinedModel(num_classes)
      

    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if mz == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    elif mz == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    elif mz == 'Lion':
        optimizer = Lion(net.parameters(), lr=opt.lr, weight_decay=1e-2)

    def train(epoch):
                  print('\nEpoch: %d' % epoch)
                  print('Batch_size:', bs)
                  print('learning_rate:', lr)
                  print('optimizer:', mz)
                  print('Total_epochs:', pch)

                  global Train_acc
                  net.to(device)
                  net.train()
                  train_loss = 0
                  correct = 0
                  total = 0
                  start_time = time.monotonic()


                  for batch_idx, (inputs, targets) in enumerate(trainloader):
                        #if use_cuda:
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        inputs, targets = Variable(inputs), Variable(targets)
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        #utils.clip_gradient(optimizer, 0.1)
                        optimizer.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.data).cpu().sum().item()

                        utils.progress_bar(batch_idx, len(trainloader), 'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

                        Train_acc = 100.*correct/total

    def test(epoch):
                        global Test_acc
                        global best_Test_acc
                        global best_Test_acc_epoch
                        net.to(device)
                        net.eval()
                        PrivateTest_loss = 0
                        correct = 0
                        total = 0
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            

                            #if use_cuda:
                            inputs, targets = inputs.to(device), targets.to(device)
                            inputs, targets = Variable(inputs), Variable(targets)
                            outputs = net(inputs)
                        
                            loss = criterion(outputs, targets)
                            PrivateTest_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += targets.size(0)
                            correct += predicted.eq(targets.data).cpu().sum().item()

                            utils.progress_bar(batch_idx, len(testloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
                                % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        # Save checkpoint.
                        Test_acc = 100.*correct/total

                        if Test_acc > best_Test_acc:
                            print('Saving..')
                            print("best_Test_acc: %0.3f" % Test_acc)
                            state = {'net': net.state_dict() if use_cuda else net,
                                'best_Test_acc': Test_acc,
                                'best_Test_acc_epoch': epoch,
                            }
                            if not os.path.isdir(opt.dataset + '_' + opt.model):
                                os.mkdir(opt.dataset + '_' + opt.model)
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            torch.save(state, os.path.join(path, 'Test_model.t7'))
                            best_Test_acc = Test_acc
                            best_Test_acc_epoch = epoch
    total_start_time = time.monotonic()
    #for total_epochs in total_epoch:
    for epoch in range(start_epoch, opt.epoch):
                        start_time = time.monotonic()
                        train(epoch)
                        test(epoch)
                        end_time = time.monotonic()
                        epoch_hours, epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s')

                        # if best_Test_acc > best_accuracy:
                        #  best_accuracy = best_Test_acc
                        #  best_params = {'batch_size': opt.bs, 'learning_rate': opt.lr, 'optimizer': opt.mz, 'Epochs': opt.epoch}
                        #  early_stopping_counter = 0
                        # else:
                        #  early_stopping_counter += 1
                        #  if early_stopping_counter >= early_stopping_threshold:
                        #         print(f'Early stopping at epoch {epoch + 1} due to lack of improvement.')
                        #         break

    total_end_time = time.monotonic()
    total_hours, total_mins, total_secs = epoch_time(total_start_time, total_end_time)
    total_time_estimate_hours = total_hours + (total_mins / 60) + (total_secs / 3600)
    print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s | Estimated Total Time: {total_time_estimate_hours:.2f} hours')

    if best_Test_acc > best_accuracy:
            best_accuracy = best_Test_acc
            best_params = {'batch_size': opt.bs, 'learning_rate': opt.lr, 'optmizer': opt.mz, 'Epochs': opt.epoch}

    print("Best parameters:")
    print(best_params)
    print("Best test accuracy: %0.3f" % best_accuracy)





