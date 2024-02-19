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
import utils
import matplotlib.pyplot as plt
from models import combmodel1
from models.efficientvit import EfficientViT
from KMU import KMU
from torch.autograd import Variable
from models.vgg import VGG
from models import SwinT
import timm
from lion_pytorch import Lion
import time
from vit_pytorch import SimpleViT
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

#torch.manual_seed(42)

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='efficientvitwcc', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_accuracy_values = []
test_accuracy_values = []
train_loss_values = []
test_loss_values = []

#cut_size = 60
total_epoch = 90

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# Data
print('==> Preparing data..')
print(use_cuda)
transforms_vaild = torchvision.transforms.Compose([
                                     torchvision.transforms.ToPILImage(),
                                     torchvision.transforms.Resize((224,)),
                                     #torchvision.transforms.CenterCrop(224),
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
                                      #torchvision.transforms.RandomCrop((200, 200)),
                                      torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                      torchvision.transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),
                                      torchvision.transforms.GaussianBlur(kernel_size=5),
                                      torchvision.transforms.ToTensor(),
                                      #torchvision.transforms.Normalize((0.2274,), (0.2353,))
                                      torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                     ])


trainset = KMU(split = 'Training', fold = opt.fold, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
testset = KMU(split = 'Testing', fold = opt.fold, transform=transforms_vaild)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


# Model
if opt.model == 'Ourmodel':
   num_classes = 6  # Adjust based on your task
   net  = combmodel1.CombinedModel(num_classes) 
   #net =  shufflenetw.CombinedModel(num_classes)
   #net =  shufflenetwcc.CombinedModel(num_classes)
   #net =  efficientvitw.CombinedModel(num_classes)
   #net =  efficientvitwcc.CombinedModel(num_classes)
   #net  = newmodel.CombinedModel(num_classes)
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'Test_model.t7'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')
'''
if use_cuda:
    print('use_cuda',use_cuda)
    device = "cuda:1" 
    net.to(device)
'''
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#optimizer = Lion(net.parameters(), lr=opt.lr, weight_decay=1e-2)
####
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_time = elapsed_time - elapsed_hours * 3600
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_hours, elapsed_mins, elapsed_secs

# Training

total_processing_time_train = 0
total_processing_time_test = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    global total_processing_time_train
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
        start_time = time.time() 
        outputs = net(inputs)
        end_time = time.time()  # Record the end time
        processing_time = end_time - start_time
        total_processing_time_train += processing_time
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
    #scheduler.step()  # Update learning rate
    Train_acc = 100.*correct/total
    train_accuracy_values.append(Train_acc)
    train_loss_values.append(train_loss / (batch_idx + 1))




def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    global total_processing_time_test
    net.to(device)
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    # Initialize variables to store processing times
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        

        #if use_cuda:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        start_time = time.time()  # Record the start time
   # Process the image with your model
        outputs = net(inputs)
        end_time = time.time()  # Record the end time
        processing_time = end_time - start_time
        
        total_processing_time_test += processing_time   
    
        loss = criterion(outputs, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        
        utils.progress_bar(batch_idx, len(testloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    Test_acc = 100.*correct/total
    test_accuracy_values.append(Test_acc)
    test_loss_values.append(PrivateTest_loss / (batch_idx + 1))
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
        
    num_training_samples = len(trainloader.dataset)
    num_testing_samples = len(testloader.dataset)
    print('Processing Time for test one image:', processing_time)
    average_processing_time_train = total_processing_time_train / num_training_samples
    average_processing_time_test = total_processing_time_test / num_testing_samples

# Print the results
    print(f'Average Processing Time for a Single Image (Training): {average_processing_time_train:.6f} seconds')
    print(f'Average Processing Time for a Single Image (Testing): {average_processing_time_test:.6f} seconds')        
total_start_time = time.monotonic()
for epoch in range(start_epoch, total_epoch):
    start_time = time.monotonic()
    train(epoch)
    test(epoch)
    end_time = time.monotonic()
    epoch_hours, epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
total_end_time = time.monotonic()

total_hours, total_mins, total_secs = epoch_time(total_start_time, total_end_time)
total_time_estimate_hours = total_hours + (total_mins / 60) + (total_secs / 3600)
print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s | Estimated Total Time: {total_time_estimate_hours:.2f} hours')
    

print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)



# After training loop is complete
# Plot training and test accuracy
'''''
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(start_epoch, total_epoch), train_accuracy_values, 'bo-', label='Training Accuracy')
plt.plot(range(start_epoch, total_epoch), test_accuracy_values, 'ro-', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and test loss
plt.subplot(1, 2, 2)
plt.plot(range(start_epoch, total_epoch), train_loss_values, 'bo-', label='Training Loss')
plt.plot(range(start_epoch, total_epoch), test_loss_values, 'ro-', label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
'''
##################################################################
