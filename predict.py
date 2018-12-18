# -*- coding: utf-8 -*-
'''
if the model is trained by multi-GPU,  use the upper load_network() function, else use the load_network() below.
'''
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense
from PIL import Image
import shutil

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='./data/market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_DesNet121', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')

opt = parser.parse_args()
opt.use_dense = True
str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

process_list = ['val_new']
data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in process_list}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=4) for x in process_list}

use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ----------single gpu training-----------------
def load_network(network, model_name=None):
    print('load pretraind model')
    if model_name == None:
        save_path = os.path.join('./model', name, 'baseline_best_without_gan.pth')
    else:
        save_path = model_name
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# def extract_feature(model, dataloaders):
#     soft_labels = torch.FloatTensor()
#     count = 0
#     min = 1
#     for data in dataloaders:
#         img, label = data
#         n, c, h, w = img.size()
#         count += n
#         #  print(count)
#         input_img = Variable(img.cuda())
#         outputs = model(input_img)
#         pred_label = outputs.data.cpu()
#         pred_label = F.softmax(pred_label, dim=1)
#         hard_label = np.argmax(pred_label, 1)
#         soft_labels = torch.cat((soft_labels, pred_label), 0)
#         for i in range(len(hard_label)):
#             print('hard_label = %4d   max value = %.4f' % (
#             hard_label[i], np.max(soft_labels.detach().cpu().numpy(), 1)[i]))
#             if np.max(soft_labels.detach().cpu().numpy(), 1)[i] < min:
#                 min = np.max(soft_labels.detach().cpu().numpy(), 1)[i]
#         print('min = %s' % min)
#     return soft_labels

def cal_softlabels(model, train_new_path):
    min = 1
    cnt = 0
    cnt_error = 0
    probability = []
    source_path = os.path.join('data/market/pytorch', 'train_new_original')
    incorrect_path = 'data/market/pytorch/incorrect_related_iamges'
    if not os.path.exists(incorrect_path):
        os.mkdir(incorrect_path)
    for p in train_new_path:
        path, v = p
        if not 'fake' in path:
            file = os.path.split(path)[-1]
            real_label = int(os.path.split(path)[0][-4:])
            input_image = Image.open(path)
            input_image = data_transforms(input_image)
            input_image = torch.unsqueeze(input_image, 0)
            if use_gpu:
                input_image = input_image.cuda()
            outputs = model(input_image)
            pred_label = torch.squeeze(outputs)
            hard_label = torch.argmax(pred_label, 0)
            soft_label = F.softmax(pred_label, 0)
            soft_label = soft_label.detach().cpu().numpy()
            hard_label = hard_label.detach().cpu().numpy()
            # soft_labels = torch.cat((soft_labels, soft_label), 0)
            if hard_label != v:
                print('v = %4s     hard_label = %4d     max value = %.4f' % (v, hard_label, np.max(soft_label)))
                print('path = %s' % path)
                if not os.path.exists(os.path.join(incorrect_path, str(real_label).zfill(4))):
                    os.mkdir(os.path.join(incorrect_path, str(real_label).zfill(4)))
                shutil.copy(path, os.path.join(incorrect_path, str(real_label).zfill(4),
                                               os.path.splitext(file)[0] + '_incorrect_' + str(real_label).zfill(
                                                   4) + '_to_' + str(hard_label).zfill(4) + '.jpg'))
                sfiles = os.listdir(os.path.join(source_path, str(real_label).zfill(4)))
                for sfile in sfiles:
                    # print(os.path.join(source_path, str(real_label).zfill(4)), sfile)
                    # print(os.path.join(incorrect_path, str(real_label).zfill(4), sfile))
                    shutil.copy(os.path.join(os.path.join(source_path, str(real_label).zfill(4)), sfile),
                                             os.path.join(incorrect_path, str(real_label).zfill(4), sfile))
                # print(os.path.join(source_path, str(hard_label).zfill(4)))
                sfiles = os.listdir(os.path.join(source_path, str(hard_label).zfill(4)))
                for sfile in sfiles:
                    shutil.copy(os.path.join(os.path.join(source_path, str(hard_label).zfill(4)), sfile),
                                             os.path.join(incorrect_path, str(real_label).zfill(4), sfile))
                # os.remove(path)
                cnt_error += 1

            probability.append(soft_label[real_label])

            if np.max(soft_label) < min:
                min = np.max(soft_label)

            if cnt == 0:
                soft_labels = np.expand_dims(soft_label, 0)
            else:
                soft_labels = np.concatenate((soft_labels, np.expand_dims(soft_label, 0)), 0)
            cnt += 1
    print('cnt = %s   cnt_error=%s   acc = %.4f   min = %s' % (
        cnt, cnt_error, 1 - (cnt_error + 1e-5) / (cnt + 1e-5), min))
    probability = np.sort(probability)
    for i in range(len(probability)):
        print('i = %4d   prob = %.4f' % (i, probability[i]))

    if cnt == 0:
        return 0
    else:
        return soft_labels



def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


train_new_path = image_datasets[process_list[0]].imgs

train_new_cam, train_new_label = get_id(train_new_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)
model = load_network(model_structure, './model/model_train_new_all/net_10.pth')
# model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


def cal_one_softlabel(src_path, model=model):
    real_label = int(os.path.split(src_path)[0][-4:])
    input_image = Image.open(src_path)
    input_image = data_transforms(input_image)
    input_image = torch.unsqueeze(input_image, 0)
    if use_gpu:
        input_image = input_image.cuda()
    outputs = model(input_image)
    pred_label = torch.squeeze(outputs)
    hard_label = torch.argmax(pred_label, 0)
    soft_label = F.softmax(pred_label, 0)
    soft_label = soft_label.detach().cpu().numpy()
    hard_label = hard_label.detach().cpu().numpy()
    return soft_label, hard_label, real_label


def re_divide(src_train_path, src_val_path, dst_train_path='train_new_re', dst_val_path='val_new_re', model=model):
    dirs = os.listdir(src_train_path)
    if not os.path.exists(os.path.join(os.path.split(src_train_path)[0], dst_train_path)):
        os.mkdir(os.path.join(os.path.split(src_train_path)[0], dst_train_path))
    if not os.path.exists(os.path.join(os.path.split(src_val_path)[0], dst_val_path)):
        os.mkdir(os.path.join(os.path.split(src_val_path)[0], dst_val_path))
    for dir in dirs:
        files_train = os.listdir(os.path.join(src_train_path, dir))
        files_val = os.listdir(os.path.join(src_val_path, dir))
        files = []
        prob = []
        for file in files_train:
            files.append(os.path.join(src_train_path, dir, file))
        for file in files_val:
            files.append(os.path.join(src_val_path, dir, file))
        for file in files:
            label = cal_one_softlabel(file, model)
            prob.append(label[0][label[2]])
        _, index = torch.topk(torch.from_numpy(np.array(prob)), int(len(prob)/2))

        if not os.path.exists(os.path.join(os.path.split(src_train_path)[0], dst_train_path, dir)):
            os.mkdir(os.path.join(os.path.split(src_train_path)[0], dst_train_path, dir))
        if not os.path.exists(os.path.join(os.path.split(src_val_path)[0], dst_val_path, dir)):
            os.mkdir(os.path.join(os.path.split(src_val_path)[0], dst_val_path, dir))
        for i in range(len(files)):
            if i == index[-1]:
                shutil.copy(files[i], os.path.join(os.path.split(src_val_path)[0], dst_val_path, dir, os.path.split(files[i])[-1]))
            else:
                shutil.copy(files[i], os.path.join(os.path.split(src_val_path)[0], dst_train_path, dir, os.path.split(files[i])[-1]))

def genete_train_new_all(src_train_path, src_val_path, dst_train_path='train_new_all'):
    dirs = os.listdir(src_train_path)
    if not os.path.exists(os.path.join(os.path.split(src_train_path)[0], dst_train_path)):
        os.mkdir(os.path.join(os.path.split(src_train_path)[0], dst_train_path))
    for dir in dirs:
        files_train = os.listdir(os.path.join(src_train_path, dir))
        files_val = os.listdir(os.path.join(src_val_path, dir))
        if not os.path.exists(os.path.join(os.path.split(src_train_path)[0], dst_train_path, dir)):
            os.mkdir(os.path.join(os.path.split(src_train_path)[0], dst_train_path, dir))
        for file in files_train:
            shutil.copy(os.path.join(src_train_path, dir, file), os.path.join(os.path.split(src_train_path)[0], dst_train_path, dir,
                                               os.path.split(file)[-1]))
        for file in files_val:
            shutil.copy(os.path.join(src_val_path, dir, file), os.path.join(os.path.split(src_val_path)[0], dst_train_path, dir,
                                           os.path.split(file)[-1]))


src_train_path = 'data/market/pytorch/train_new_original'
src_val_path = 'data/market/pytorch/val_new_original'
re_divide(src_train_path, src_val_path)
# genete_train_new_all(src_train_path, src_val_path)

# train_new_softlabels = cal_softlabels(model, train_new_path)
#
# # Save to Matlab for check
# result = {'train_new_softlabels': train_new_softlabels, 'train_new_label': train_new_label,
#           'train_new_cam': train_new_cam}
# scipy.io.savemat('predict_softlabels.mat', result)
