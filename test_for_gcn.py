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
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense
import torch.nn.functional as F
from PIL import Image

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

process_list = ['train_all_new']

data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in process_list}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=4) for x in process_list}

use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ----------single gpu training-----------------
def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
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


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #  print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            # print(f.size())
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # L2 normalize
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    names = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
        names.append(filename)
    return names, camera_id, labels


train_path = image_datasets[process_list[0]].imgs

train_name, train_cam, train_label = get_id(train_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
train_feature = extract_feature(model, dataloaders[process_list[0]])


def get_dict(path):
    dict_label = {}
    dirs = os.listdir(path)
    dirs = np.sort(dirs)
    cnt = 0
    num = []
    for dir in dirs:
        if 'gen' not in dir:
            files = os.listdir(os.path.join(path, dir))
            # print('cnt = %s  len = %s' % (cnt, len(files)))
            num.append(len(files))
            cnt += 1
            for file in files:
                # dict_label[int(dir)] = int(file[:4])
                dict_label[file[:4]] = dir
                break
    print('index = %d' % np.argmax(num))
    print('max = %s' % (np.max(num)))
    print('ditc_label = %s' % dict_label)
    return dict_label


def cal_one_feature(src_path, model=model):
    img = Image.open(src_path)
    img = data_transforms(img)
    img = torch.unsqueeze(img, 0)
    n, c, h, w = img.size()
    if opt.use_dense:
        ff = torch.FloatTensor(n, 1024).zero_()
    else:
        ff = torch.FloatTensor(n, 2048).zero_()
    for i in range(2):
        if (i == 1):
            img = fliplr(img)
        input_image = Variable(img.cuda())
        outputs = model(input_image)
        f = outputs.data.cpu()
        ff = ff + f
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # L2 normalize
    ff = ff.div(fnorm.expand_as(ff))
    return ff


def get_gan_info(path):
    # dict_label = get_dict('data/market/pytorch/train_all_new')
    files = os.listdir(path)
    camera_id = []
    labels = []
    names = []
    features = torch.FloatTensor()

    for file in files:
        label = file[0:4]
        camera = file.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
        names.append(file)
        ff = cal_one_feature(os.path.join(path, file))
        features = torch.cat((features, ff), 0)

    return features, names, camera_id, labels


gen_path = 'data/market/pytorch/cam/bounding_box_train_camstyle_wo_guider'

gen_feature, gen_name, gen_cam, gen_label = get_gan_info(gen_path)

# Save to Matlab for check
result = {'train_f': train_feature.numpy(), 'train_name': train_name, 'train_label': train_label,
          'train_cam': train_cam, 'gen_f': gen_feature.numpy(), 'gen_name': gen_name,
          'gen_label': gen_label,
          'gen_cam': gen_cam}

scipy.io.savemat('pytorch_train_result.mat', result)
