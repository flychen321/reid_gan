# -*- coding: utf-8 -*-
# Author: qiaoguan(https://github.com/qiaoguan/Person_reID_baseline_pytorch)
'''
this is the baseline,  if do not add gen_0000 folder(generateed images by DCGAN) under the training set,
so the LSRO equals to crossentropy loss, and the generated_image_size is 0. else the loss function will use the generated images, the loss function for
the generated images and original images are not the same.
'''
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
# soft_label = loadmat('/home/fly/gcn/gcn/data/data_temp/soft_label.mat')
# soft_label = loadmat('/home/fly/gcn/gcn/data/data_temp/weighted_label.mat')
soft_label = loadmat('/home/fly/gcn/gcn/data/data_temp/hard_label.mat')
soft_label = soft_label['soft_label']
print('soft_label.shape = ')
print(soft_label.shape)

######################################################################
# Options
parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--gpu_ids',default='3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_DesNet121', type=str, help='output model name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_soft_label', default=True, type=bool, help='use_soft_label')
parser.add_argument('--prob', default=0.8, type=float, help='hard label probability, in [0,1]')
parser.add_argument('--modelname', default='', type=str, help='save model name')

opt = parser.parse_args()
opt.use_dense = True
data_dir = opt.data_dir
name = opt.name
opt.prob = opt.prob/100.0
print('prob = %.3f' % opt.prob)
assert opt.prob >= 0 and opt.prob <= 1
print('save model name = %s' % opt.modelname)
generated_image_size = 0
'''
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
'''
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


# read dcgan data
class dcganDataset(Dataset):
    def __init__(self, root, transform=None, targte_transform=None):
        super(dcganDataset, self).__init__()
        self.image_dir = os.path.join(opt.data_dir, root)
        self.samples = []  # train_data   xxx_label_flag_yyy.jpg
        self.img_label = []
        self.img_flag = []
        self.transform = transform
        self.targte_transform = targte_transform
        #   self.class_num=len(os.listdir(self.image_dir))   # the number of the class
        self.train_val = root  # judge whether it is used for training for testing
        for folder in os.listdir(self.image_dir):
            fdir = self.image_dir + '/' + folder  # folder gen_0000 means the images are generated images, so their flags are 1
            files = os.listdir(fdir)
            for file in files:
                temp = folder + '_' + file
                if 'fake' in file:
                    prob = opt.prob
                    label = np.zeros((751,), dtype=np.float32)
                    label.fill((1 - prob) / 750)
                    label[int(folder[-4:])] = prob
                    self.img_label.append(label)  # need to modify
                    self.img_flag.append(1)
                else:
                    label = np.zeros((751,), dtype=np.float32)
                    label[int(folder[-4:])] = 1
                    self.img_label.append(label)
                    # self.img_label.append(int(folder[-4:]))
                    self.img_flag.append(0)
                self.samples.append(temp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        temp = self.samples[idx]  # folder_files
        foldername = temp[:4]
        filename = temp[5:]
        img = default_loader(self.image_dir + '/' + foldername + '/' + filename)
        if self.train_val == 'train_new':
            result = {'img': data_transforms['train'](img), 'label': self.img_label[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 1 for generated data
        else:
            result = {'img': data_transforms['val'](img), 'label': self.img_label[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 1 for generated data
        return result


def loss_entropy(input_soft, target_soft, reduce=True):
    input_soft = F.log_softmax(input_soft, dim=1)
    result = -target_soft * input_soft
    result = torch.sum(result, 1)
    if reduce:
        result = torch.mean(result)
    return result


loss_print_cnt = 0


class LSROloss(nn.Module):
    def __init__(self):  # change target to range(0,750)
        super(LSROloss, self).__init__()
        # input means the prediction score(torch Variable) 32*752,target means the corresponding label,

    def forward(self, input, target,
                flg):  # while flg means the flag=0 for true data and 1 for generated data)  batchsize*1
        global loss_print_cnt
        # print(type(input))
        if input.dim() > 2:  # N defines the number of images, C defines channels,  K class in total
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        # normalize input
        maxRow, _ = torch.max(input.data, 1)  # outputs.data  return the index of the biggest value in each row
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        loss = loss_entropy(input, target)

        # if loss_print_cnt % 500 == 0:
        #     print('flg = %s' % flg.view(1, -1).cpu().detach().numpy())
        #     print('floss = %s' % loss.cpu().detach().numpy())
        # loss_print_cnt += 1

        return loss


dataloaders = {}
dataloaders['train'] = DataLoader(dcganDataset('train_new', data_transforms['train']), batch_size=opt.batchsize,
                                  shuffle=True, num_workers=8)
dataloaders['val'] = DataLoader(dcganDataset('val_new', data_transforms['val']), batch_size=opt.batchsize,
                                shuffle=True, num_workers=8)

dataset_sizes = {}
dataset_train_dir = os.path.join(data_dir, 'train_new')
dataset_val_dir = os.path.join(data_dir, 'val_new')
dataset_sizes['train'] = sum(len(os.listdir(os.path.join(dataset_train_dir, i))) for i in os.listdir(dataset_train_dir))
dataset_sizes['val'] = sum(len(os.listdir(os.path.join(dataset_val_dir, i))) for i in os.listdir(dataset_val_dir))

print(dataset_sizes['train'])
print(dataset_sizes['val'])

# class_names={}
# class_names['train']=len(os.listdir(dataset_train_dir))
# class_names['val']=len(os.listdir(dataset_val_dir))
use_gpu = torch.cuda.is_available()

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

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                inputs = data['img']
                labels = data['label']
                flags = data['flag']

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    flags = Variable(flags.cuda())
                else:
                    inputs, labels, flags = Variable(inputs), Variable(labels), Variable(flags)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)  # outputs.data  return the index of the biggest value in each row
                loss = criterion(outputs, labels, flags)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

                # for temp in range(flags.size()[0]):
                #     if flags.data[temp] == 1:
                #         preds[temp] = -1

                running_corrects += torch.sum(preds == labels.max(1)[-1].data)

            epoch_loss = float(running_loss) / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                if epoch >= 0:
                    save_network(model, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    save_network(model, opt.modelname + '_best')
    return model


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
def load_network(network):
    print('load pretraind model')
    save_path = os.path.join('./model', name, 'base_net_best.pth')
    network.load_state_dict(torch.load(save_path))
    return network


# print('------------'+str(len(clas_names))+'--------------')
if opt.use_dense:
    # print(len(class_names['train']))
    model = ft_net_dense(751)  # 751 class for training data in market 1501 in total
else:
    model = ft_net(751)

if use_gpu:
    model = model.cuda()
criterion = LSROloss()

ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

refine = False
print('refine = %s' % refine)
if refine:
    ratio = 0.1
    step = 10
    epoc = 30
    load_network(model)
else:
    ratio = 1
    step = 40
    epoc = 130

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
