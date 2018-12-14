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
opt = parser.parse_args()
opt.use_dense = True

data_dir = opt.data_dir
name = opt.name

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
        self.img_label_soft = []
        self.img_flag = []
        self.transform = transform
        self.targte_transform = targte_transform
        #   self.class_num=len(os.listdir(self.image_dir))   # the number of the class
        self.train_val = root  # judge whether it is used for training for testing
        i = 0
        if root == 'train_new':
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder  # folder gen_0000 means the images are generated images, so their flags are 1
                if folder == 'gen_0000':
                    # print(os.listdir(fdir)[:50])
                    # print(np.sort(os.listdir(fdir))[:50])
                    # assert 0
                    files_sort = np.sort(os.listdir(fdir))
                    for files in files_sort:
                        temp = folder + '_' + files
                        self.img_label.append(int(folder[-4:]))
                        if soft_flag:
                            self.img_label_soft.append(soft_label[i])
                            i += 1
                        self.img_flag.append(1)
                        self.samples.append(temp)
                else:
                    for files in os.listdir(fdir):
                        temp = folder + '_' + files
                        self.img_label.append(int(folder))
                        if soft_flag:
                            self.img_label_soft.append(np.zeros((751,), dtype=np.float32))
                        self.img_flag.append(0)
                        self.samples.append(temp)
        else:  # val
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder
                for files in os.listdir(fdir):
                    temp = folder + '_' + files
                    self.img_label.append(int(folder))
                    if soft_flag:
                        self.img_label_soft.append(np.zeros((751,), dtype=np.float32))
                    self.img_flag.append(0)
                    self.samples.append(temp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        temp = self.samples[idx]  # folder_files
        # print(temp)
        if self.img_flag[idx] == 1:
            foldername = 'gen_0000'
            filename = temp[9:]
        else:
            foldername = temp[:4]
            filename = temp[5:]
        img = default_loader(self.image_dir + '/' + foldername + '/' + filename)
        if self.train_val == 'train_new':
            result = {'img': data_transforms['train'](img), 'label': self.img_label[idx],
                      'soft_label': self.img_label_soft[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 1 for generated data
        else:
            result = {'img': data_transforms['val'](img), 'label': self.img_label[idx],
                      'soft_label': self.img_label_soft[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 1 for generated data
        return result


def loss_entropy(input_soft, target_soft, reduce=False):
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

    def forward(self, input, target, target_soft,
                flg):  # while flg means the flag(=0 for true data and 1 for generated data)  batchsize*1
        global loss_print_cnt
        # print(type(input))
        if input.dim() > 2:  # N defines the number of images, C defines channels,  K class in total
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        # normalize input
        input_soft = input
        maxRow, _ = torch.max(input.data, 1)  # outputs.data  return the index of the biggest value in each row
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow

        target = target.view(-1, 1)  # batchsize*1
        flg = flg.view(-1, 1)
        loss_func = nn.MSELoss(reduce=False)
        if soft_flag:
            # input_soft = F.softmax(input_soft, dim=1)
            # flos = loss_func(input_soft, target_soft)
            # flos = torch.sum(flos, 1)
            # flos *= 10

            flos = loss_entropy(input_soft, target_soft, reduce=False)

        else:
            flos = F.log_softmax(input, dim=1)  # N*K?      batchsize*751
            flos = torch.sum(flos, 1) / flos.size(1)  # N*1  get average      gan loss
        logpt = F.log_softmax(input, dim=1)  # size: batchsize*751
        logpt = logpt.gather(1, target)  # here is a problem
        logpt = logpt.view(-1)  # N*1     original loss
        flg = flg.view(-1)
        flg = flg.type(torch.cuda.FloatTensor)
        loss = -1 * logpt * (1 - flg) + flos * flg

        if loss_print_cnt % 500 == 0:
            print('flg = %s' % flg.view(1, -1))
            print('floss = %s' % flos.view(1, -1))
            print('logpt = %s' % (-logpt).view(1, -1))
            print('floss_flag = %s' % (flos * flg).view(1, -1))
            print('logpt_flag = %s' % (-1 * logpt * (1 - flg)).view(1, -1))
            # print('loss_print_cnt =%-6d   loss.mean = %s' % (loss_print_cnt, (flos * flg).sum().data[0]/flg.sum().data[0]))
            # print('loss_print_cnt =%-6d  logpt.mean = %s' % (loss_print_cnt, (-1 * logpt * (1 - flg)).sum().data[0]/(32 - flg.sum().data[0])))
            if flg.sum().data[0] != 0 and flg.sum().data[0] != 32:
                print('cnt = %-6d     floss = %-18s      logpt = %-18s     loss.mean() = %-18s'
                      % (loss_print_cnt, (flos * flg).sum().data[0] / flg.sum().data[0],
                         (-1 * logpt * (1 - flg)).sum().data[0] / (32 - flg.sum().data[0]), loss.mean().data[0]))
        elif loss_print_cnt % 50 == 0:
            if flg.sum().data[0] != 0 and flg.sum().data[0] != 32:
                print('cnt = %-6d     floss = %-18s      logpt = %-18s     loss.mean() = %-18s'
                      % (loss_print_cnt, (flos * flg).sum().data[0] / flg.sum().data[0],
                         (-1 * logpt * (1 - flg)).sum().data[0] / (32 - flg.sum().data[0]), loss.mean().data[0]))
        loss_print_cnt += 1

        return loss.mean()


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
                soft_labels = data['soft_label']
                flags = data['flag']

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    soft_labels = Variable(soft_labels.cuda())
                    flags = Variable(flags.cuda())
                else:
                    inputs, labels, flags = Variable(inputs), Variable(labels), Variable(flags)
                    soft_labels = Variable(soft_labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)  # outputs.data  return the index of the biggest value in each row
                loss = criterion(outputs, labels, soft_labels, flags)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

                for temp in range(flags.size()[0]):
                    if flags.data[temp] == 1:
                        preds[temp] = -1

                running_corrects += torch.sum(preds == labels.data)
                # print('running_corrects: '+str(running_corrects))
                # print('cnt = %d' % cnt)
                # cnt = cnt + 1
                # print('running_loss = %s' % running_loss)
                # print('loss.data[0] = %s' % loss.data[0])

            epoch_loss = running_loss / dataset_sizes[phase]
            # print('epoch_loss = %s' % epoch_loss)
            # epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'train':
                # epoch_acc = running_corrects / (dataset_sizes[phase]-4992)    # 4992 generated image in total
                epoch_acc = float(running_corrects) / (
                            dataset_sizes[phase] - generated_image_size)  # 4992 generated image in total
            else:
                epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                if epoch >= 0:
                    save_network(model, epoch)
            #    draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    return model


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)
    # this step is important, or error occurs "runtimeError: tensors are on different GPUs" 


#   if torch.cuda.is_available:
#       network.cuda(gpu_ids[0])
# if torch.cuda.is_available:
#    network=nn.DataParallel(network,device_ids=[0,1,2]) # multi-GPU


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

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.01},
    {'params': model.model.fc.parameters(), 'lr': 0.05},
    {'params': model.classifier.parameters(), 'lr': 0.05}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

# model=nn.DataParallel(model,device_ids=[0,1,2]) # multi-GPU

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=130)
