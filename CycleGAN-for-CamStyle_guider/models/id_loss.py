# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
from torchvision import  transforms
import os
import id_model
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat


data_transforms = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_toimage = transforms.Compose([
    # transforms.Resize((288, 144), interpolation=3),
    transforms.ToPILImage(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()
# use_gpu = False

def load_network(network):
    # print(os.getcwd())
    save_path = os.path.join('models/reid_model', 'net_%s.pth' % 'best')
    network.load_state_dict(torch.load(save_path))
    return network


model_structure = id_model.ft_net_dense(751)

model = load_network(model_structure)
model = model.eval()
if use_gpu:
    model = model.cuda()

for parm in model.parameters():
    # print('before parm.requires_grad = %s' % parm.requires_grad)
    parm.requires_grad = False
    # print('after parm.requires_grad = %s' % parm.requires_grad)


dict_label = loadmat(os.path.join('models/reid_model', 'dict_label.mat'))
print(os.getcwd())
# print('ditc_label = %s' % dict_label_ori)
i = 0
for key in dict_label.keys():
    if isinstance(dict_label[key], np.ndarray):
        dict_label[key] = dict_label[key][0][0]
# print(dict_label)

def normlize_reverse(img):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    norm = np.ones(shape=(2, 3), dtype=np.float32)*0.5
    mean, std = np.split(norm, 2)
    for i in range(3):
        img[i] = img[i] * std[0][i]
        img[i] = img[i] + mean[0][i]
    return img


def convet_to_image(img):
    # img = data_transforms_toimage(img)
    # print(img.size())
    # img = img.cpu().numpy()
    img = img.cpu().detach().numpy()
    img = np.squeeze(img)
    img = normlize_reverse(img)
    img = np.transpose(img, [1, 2, 0])
    img = img * 255
    img = img.astype(np.uint8)
    # print('size = %s' % (img.size))
    img = Image.fromarray(img)
    return img


def show_image(img):
    img = convet_to_image(img)
    img.show()


def loss_entropy(predict, target):
    predict = F.log_softmax(predict, dim=1)
    # result = -predict[0][target]
    target = torch.zeros_like(predict).scatter_(dim=1, index=torch.LongTensor([[target]]).cuda(), value=1)
    result = -predict*target
    result = torch.sum(result, dim=1)
    result = torch.mean(result, dim=0)
    return result

def id_loss_test():
    print('id_loss_test ok')

cnt = 0
def id_loss(path=None, label=None, img=None, model=model):
    global cnt
    if path != None:
        input_image = Image.open(path)
        # print('path = %s' % path)
    # if img != None:
    else:
        # print('image = %s' % 'image')
        input_image = convet_to_image(img)
    cnt += 1
    # if cnt > 6:
    #     time.sleep(5)
    #     exit()
    # input_image.show()
    input_image = data_transforms(input_image)
    # print('image size')
    # print(input_image.shape)
    input_image = torch.unsqueeze(input_image, 0)
    if use_gpu:
        input_image = input_image.cuda()
    output = model(input_image)
    loss = loss_entropy(output, label)
    # print('loss = %s' % loss)
    return loss


# files = os.listdir(os.path.join(dirs_path, '0000'))
# for file in files:
#     id_loss(os.path.join(dirs_path, '0000', file), 0)
#     break
