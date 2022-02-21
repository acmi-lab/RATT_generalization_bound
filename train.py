import enum
import os
import argparse

from models import *
from utils import progress_bar
import random
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn

from data_helper import *
from model_helper import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=.10, type=float, help='learning rate')
parser.add_argument('--clean-samples', default=40000, type=int, help='Amount of random data')
parser.add_argument('--noise-samples', default=10000, type=int, help='Amount of clean data')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--momentum', default= 0.9, type=float, help='Momentum')
parser.add_argument('--data-dir', default= "./data", type=str, help='Data directory')
parser.add_argument('--log-dir', default= "./log", type=str, help='Log directory')
parser.add_argument('--dataset', default= "CIFAR10", type=str, help='Dataset')
parser.add_argument('--arch', default= "Resnet", type=str, help='Architecture')
parser.add_argument('--batch-size', default=200, type=int, help='Batch size')
parser.add_argument('--num-classes', default=10, type=int, help='Num classes in the dataset')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
parser.add_argument('--check-steps', default=100, type=int, help='Number of steps to check accuracy')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = args.lr
momentum = args.momentum
wd = args.wd
data_dir = args.data_dir
dataset_name = args.dataset
clean_samples = args.clean_samples
noise_samples = args.noise_samples
arch = args.arch
batch_size = args.batch_size
num_classes = args.num_classes
epochs = args.epochs
check_steps = args.check_steps
log_dir = args.log_dir

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_file = os.path.join(log_dir, f"{dataset_name}_{arch}_lr_{lr}_momentum_{momentum}_wd_{wd}_clean_{clean_samples}_noise_{noise_samples}.txt")

log_file_handler = open(log_file, "w")

print('==> Loading data..')

transform = get_tranform(dataset_name)

train_data = get_train_data(data_dir, dataset_name, clean_samples, noise_samples, transform)
test_data = get_test_data(data_dir, dataset_name, transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
trainloader_check_err = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

print('==> Building model..')

net = get_model(arch, num_classes=num_classes)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = get_optimizer(net, dataset_name, lr=lr, momentum=momentum, weight_decay=wd, step_size= (clean_samples + noise_samples)*1.0*100/batch_size)

log_file_handler.write(f"Epoch, Step, Train Acc, Pred Acc, Test Acc\n")

print('==> Training..')

curr_step = 0
curr_epoch = 0

for curr_epoch in range(epochs): 

    for batch_idx, batch in enumerate(trainloader): 
        if curr_step % check_steps == 0: 
            
            train_acc, pred_acc = pred_accuracy(net, trainloader_check_err, device, num_classes = num_classes)
            true_acc = true_accuracy(net, testloader, device, num_classes=num_classes)

            log_file_handler.write(f"{curr_epoch}, {curr_step}, {train_acc}, {pred_acc}, {true_acc}\n")
            log_file_handler.flush()

        net.train()
        x,y = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        if scheduler is not None: 
            scheduler.step()

        curr_step += 1


log_file_handler.close()