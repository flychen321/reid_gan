# -*- coding: utf-8 -*-
# Author: chen
from __future__ import print_function, division
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import matplotlib
from PIL import Image

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_dense
from random_erasing import RandomErasing
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
from scipy.io import savemat

soft_flag = True

######################################################################
# Options
parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--gpu_ids',default='3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_DesNet121', type=str, help='output model name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_soft_label', default=True, type=bool, help='use_soft_label')
parser.add_argument('--prob', default=80, type=float, help='hard label probability, in [0,1]')
parser.add_argument('--modelname', default='', type=str, help='save model name')
parser.add_argument('--max', default=80, type=float, help='max label probability, in [0,1]')
parser.add_argument('--min', default=60, type=float, help='min label probability, in [0,1]')

opt = parser.parse_args()
opt.use_dense = True
data_dir = opt.data_dir
name = opt.name
opt.prob = opt.prob / 100.0
print('prob = %.3f' % opt.prob)
assert opt.prob >= 0 and opt.prob <= 1
opt.max = opt.max / 100.0
print('max val = %.3f' % opt.max)
assert opt.max >= 0 and opt.max <= 1
opt.min = opt.min / 100.0
print('min val = %.3f' % opt.min)
assert opt.min >= 0 and opt.min <= 1

print('save model name = %s' % opt.modelname)
generated_image_size = 0
use_gpu = torch.cuda.is_available()

######################################################################
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(144, interpolation=3),
    transforms.RandomCrop((256, 128)),
    #   transforms.Resize(256,interpolation=3),
    #   transforms.RandomCrop(224,224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]

# print(transform_train_list)

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    # transforms.Resize(256,interpolation=3),
    # transforms.RandomCrop(224,224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)
    # this step is important, or error occurs "runtimeError: tensors are on different GPUs"


######################################################################
# Load model
# ----------single gpu training-----------------
def load_network(network, model_name=None):
    print('load pretraind model')
    if model_name == None:
        # save_path = os.path.join('./model', name, 'baseline_best_without_gan.pth')
        save_path = os.path.join('./model', name, 'net_best.pth')
    else:
        save_path = model_name
    network.load_state_dict(torch.load(save_path))
    return network


dataset_sizes = {}
dataset_train_dir = os.path.join(data_dir, 'train_new')
dataset_val_dir = os.path.join(data_dir, 'val_new')
dataset_sizes['train'] = sum(len(os.listdir(os.path.join(dataset_train_dir, i))) for i in os.listdir(dataset_train_dir))
dataset_sizes['val'] = sum(len(os.listdir(os.path.join(dataset_val_dir, i))) for i in os.listdir(dataset_val_dir))

dataloaders = {}
dataloaders['train'] = DataLoader(datasets.ImageFolder(dataset_train_dir, data_transforms['train']), batch_size=opt.batchsize,
                                  shuffle=True, num_workers=8)
dataloaders['val'] = DataLoader(datasets.ImageFolder(dataset_val_dir, data_transforms['val']), batch_size=opt.batchsize,
                                shuffle=True, num_workers=8)

print('dataset_sizes[train] = %s' % dataset_sizes['train'])
print('dataset_sizes[val] = %s' % dataset_sizes['val'])

class_num = {}
class_num['train'] = len(os.listdir(dataset_train_dir))
class_num['val'] = len(os.listdir(dataset_val_dir))
print('class_num  train = %d   test = %d' % (class_num['train'], class_num['val']))

if opt.use_dense:
    # print(len(class_names['train']))
    model = ft_net_dense(class_num['train'])  # 751 class for training data in market 1501 in total
    model_pred = ft_net_dense(class_num['train'])  # 751 class for training data in market 1501 in total
else:
    model = ft_net(class_num['train'])
    model_pred = ft_net(class_num['train'])

if use_gpu:
    model = model.cuda()
    model_pred = model_pred.cuda()

######################################################################
# Training the model
# ------------------

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


# cnt = 0
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global cnt
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)  # outputs.data  return the index of the biggest value in each row
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

                # for temp in range(flags.size()[0]):
                #     if flags.data[temp] == 1:
                #         preds[temp] = -1

                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = float(running_loss) / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if epoch >= 0:
                if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    save_network(model, 'best')
                    save_network(model, opt.modelname + '_best')
                if epoch >= 40 and (epoc+1)%10 == 0:
                    save_network(model, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val epoch: {:d}'.format(best_epoch))
    print('Best val Loss: {:.4f}  Acc: {:4f}'.format(best_loss, best_acc))

    save_network(model, 'last')
    save_network(model, opt.modelname + '_last')
    return model


criterion = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

refine = False
print('refine = %s' % refine)
if refine:
    ratio = 0.2
    step = 10
    epoc = 35
    load_network(model)
else:
    ratio = 1
    step = 30
    epoc = 100

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': ratio * 0.01},
    {'params': model.model.fc.parameters(), 'lr': ratio * 0.05},
    {'params': model.classifier.parameters(), 'lr': ratio * 0.05}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=0.1)

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoc)
