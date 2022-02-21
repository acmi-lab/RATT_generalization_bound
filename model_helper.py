import numpy as np
import sys

import torch 
import torch.nn as nn
from transformer import AdamW
from keras.optimizers import adam


from models import *


def get_model(model_type, input_dim=None, num_classes=2 ): 
    if model_type.lower() == 'fcn':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, num_classes, bias=True)
        )
    elif model_type.lower() == 'linear':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, num_classes, bias=True),
        )
    elif model_type.lower() == 'resnet':
        net = ResNet18(num_classes=num_classes)

    elif model_type.lower() == 'lenet':
        net = LeNet(num_classes=num_classes)

    elif model_type.lower() == 'allconv': 
        net = AllConv(num_classes= num_classes)

    elif model_type.lower() == "distilbert":
        net = initialize_bert_based_model("distilbert-base-uncased", num_classes=num_classes)
    
    else:
        raise NotImplementedError("Model type must be one of FCN | Resnet | linear ... ")

    return net


def get_optimizer(net, dataset_name, lr = 0.1, momentum = 0.9, weight_decay = 0.0005, step_size = None) : 

    if dataset_name.lower().startswith("cifar") or dataset_name.lower().startswith("mnist"):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif dataset_name.lower().startswith("imdb_bert"):
        optimizer = AdamW(net.parameters(), lr=lr)
        scheduler = None

    elif dataset_name.lower().startswith("imdb_elmo"): 
        optimizer = adam(lr = lr, decay = weight_decay)
        scheduler = None
    else: 
        raise NotImplementedError("Dataset name must start with cifar | mnist | imdb_bert | imdb_elmo")

    return optimizer, scheduler


def pred_accuracy(net, dataloader, device, num_classes): 
    net.eval()
    true_correct = 0
    noisy_correct = 0

    true_total = 0
    noisy_total = 0

    with torch.no_grad(): 
        for batch_idx, batch in enumerate(dataloader): 

            inputs = batch[0].to(device)

            labels = batch[1].numpy()

            mask = batch[3].numpy()
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predicted  = predicted.cpu().numpy()
            
            true_total += labels.size(0) -  np.sum(mask)
            noisy_total += np.sum(mask)

            true_idx = np.where(mask == 0)[0]
            noisy_idx = np.where(mask == 1)[0]

            true_correct += np.sum(predicted[true_idx] == labels[true_idx])    
            noisy_correct += np.sum(predicted[noisy_idx] == labels[noisy_idx])

    train_acc = true_correct*100.0 / true_total
    noisy_acc = noisy_correct*100.0 / noisy_total
    pred_err = (100.0 - train_acc) + (num_classes - 1)*( 100.0 - num_classes/(num_classes - 1) * ( 100.0 - noisy_correct))
    pred_acc = 100.0 - pred_err
   
    return train_acc, pred_acc


def true_accuracy(net, dataloader, device, num_classes): 
    net.eval()
    true_correct = 0
    true_total = 0

    with torch.no_grad(): 
        for batch_idx, batch in enumerate(dataloader): 

            inputs = batch[0].to(device)

            labels = batch[1].numpy()

            true_total += labels.size(0)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predicted  = predicted.cpu().numpy()
            
            true_correct += np.sum(predicted == labels)    

    true_acc = true_correct*100.0 / true_total
   
    return true_acc