import os
import shutil
import numpy as np
import cv2
import glob

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


def move_cam_image_to_train(src_path, dst_path):
    cnt = 0
    # files = os.listdir(src_path)
    files = glob.glob(os.path.join(src_path, '*.jpg'))
    dict_label = get_dict(dst_path)
    for file in files:
        file = os.path.split(file)[-1]
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        dir = dict_label[file[:4]]
        if not os.path.exists(os.path.join(dst_path, dir)):
            os.mkdir(os.path.join(dst_path, dir))
        # shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, dir, file))
        src = cv2.imread(os.path.join(src_path, file))
        dst_f = cv2.resize(src, (64, 128))
        cv2.imwrite(os.path.join(dst_path, dir, file), dst_f)
        cnt += 1
    print('cnt = %s' % cnt)


def move_cam_original_to_temp(src_path, dst_path):
    files = os.listdir(src_path)
    cnt = 0
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for file in files:
        # if '1to2' in file or '2to1' in file:
        if 'mask.' in file and ('c1' in file or 'c2' in file):
        # if 'c1' in file or 'c2' in file:
            shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))
            cnt += 1
    print('cnt = %s' % cnt)


def from_part_to_mask(path_src, path_dst):
    if not os.path.exists(path_dst):
        os.mkdir(path_dst)
    files = os.listdir(path_src)
    files = np.sort(files)
    cnt = 0
    for i in np.arange(0, len(files) - 1, 2):
        print('i = %s   cnt = %s   file_part = %s    file_real = %s' % (i, cnt, files[i], files[i + 1]))
        mask = cv2.imread(os.path.join(path_src, files[i]), 0)
        real = cv2.imread(os.path.join(path_src, files[i+1]), 1)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.merge([mask, mask, mask])
        # print(mask.shape)
        # cv2.imshow('mask', mask)
        k = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
        k = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask, kernel)
        mask = cv2.bitwise_and(real, mask)
        # mask = cv2.bitwise_or(real, cv2.bitwise_not(mask))
        # mask = cv2.resize(mask, (64, 128))
        # cv2.imshow('mask2', mask)
        # cv2.waitKey(0)
        # break

        cv2.imwrite(os.path.join(path_dst, files[i+1][:-11]+'_mask_255.jpg'), mask)

        cnt += 1

def every_cam_cnt(path):
    files = os.listdir(path)
    print('file num = %s' % len(files))
    cnt = np.zeros((6,)).astype(int)
    for file in files:
        for i in range(1, 7):
            if 'c%s'%i in file:
                # print('c%s'%i)
                cnt[i-1] += 1
                break

    print('cnt = %s' % cnt)
    print('total cnt = %s' % sum(cnt))



if __name__ == '__main__':

    train_new_original_path = 'data/market/pytorch/train_new_original'
    train_new_path = 'data/market/pytorch/train_new_original_0.2idloss_cam12_cam13'
    camstyle_path = '/home/dl/cf/cyclegan_guider/CycleGAN-for-CamStyle_guider/results/market/bounding_box_train_camstyle_cam1_cam3_0.2idloss'
    # get_dict(train_new_original_path)
    # src_base_path = '/home/dl/cf/reid_gan/data/market/pytorch/resize_rename'


    # dirs = os.listdir(src_base_path)
    # for dir in dirs:
    #     move_cam_image_to_train(os.path.join(src_base_path, dir), train_new_path)
    move_cam_image_to_train(camstyle_path, train_new_path)
