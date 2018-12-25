# -*- coding: utf-8 -*-
# Author: chen
'''
this is the baseline,  if do not add gen_0000 folder(generateed images by DCGAN) under the training set,
so the LSRO equals to crossentropy loss, and the generated_image_size is 0. else the loss function will use the generated images, the loss function for
the generated images and original images are not the same.
'''
from __future__ import print_function, division
import torch
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
from PIL import Image
from model import ft_net, ft_net_dense
import torch.nn.functional as F
from scipy.io import loadmat
import os

soft_flag = True
dir_path = '/home/fly/github/reid_gcn/gcn/data/data_reid'
soft_labels = loadmat(os.path.join(dir_path, 'soft_label_dict_link24.mat'))

print('soft_labels len = ')
print(len(soft_labels))
for key in soft_labels.keys():
    if '.jpg' not in key:
        print(key)
######################################################################
# Options
use_gpu = torch.cuda.is_available()

######################################################################

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    # transforms.Resize(256,interpolation=3),
    # transforms.RandomCrop(224,224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'val': transforms.Compose(transform_val_list),
}


######################################################################
# Load model
# ----------single gpu training-----------------
def load_network(network, model_name=None):
    print('load pretraind model')
    if model_name == None:
        save_path = os.path.join('./model', 'ft_DesNet121', 'baseline_best_without_gan.pth')
    else:
        save_path = model_name
    network.load_state_dict(torch.load(save_path))
    return network


model = ft_net_dense(751)  # 751 class for training data in market 1501 in total

if use_gpu:
    model = model.cuda()

load_network(model)
model.eval()

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

dict_label = get_dict('data/market/pytorch/train_new')

def get_one_softlabel(path, model=model):
    input_image = Image.open(path)
    file = os.path.split(path)[-1]
    real_label = int(dict_label[file[:4]])
    input_image = data_transforms['val'](input_image)
    input_image = torch.unsqueeze(input_image, 0)
    if use_gpu:
        input_image = input_image.cuda()
    outputs = model(input_image)
    pred_label = torch.squeeze(outputs)
    hard_label = torch.argmax(pred_label, 0)
    soft_label = F.softmax(pred_label, 0)
    soft_label = soft_label.detach().cpu().numpy()

    # print(orig_value)
    # print(soft_label[real_label])
    # print(sum(soft_label))

    return soft_label, hard_label, real_label

camstyle_path = 'data/market/pytorch/cam/bounding_box_train_camstyle_wo_guider'

files = os.listdir(camstyle_path)
cnt_r_gcn = 0
cnt_r_cnn = 0
cnt = 0
for file in files:
    result = get_one_softlabel(os.path.join(camstyle_path, file))
    if result[1] == result[2]:
        cnt_r_cnn += 1
    if np.argmax(np.squeeze(soft_labels[file]), 0) == result[2]:
        cnt_r_gcn += 1
    cnt += 1

print('cnt = %d  cnn = %d  gcn = %d' % (cnt, cnt_r_cnn, cnt_r_gcn))


