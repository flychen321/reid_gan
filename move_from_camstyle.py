import os
import shutil
import numpy as np
import cv2
import glob
import argparse
parser = argparse.ArgumentParser(description='Augment')
parser.add_argument('--data_dir', default='cycle_all', type=str, help='data_dir')
parser.add_argument('--mode', default=1, type=int, help='mode')
opt = parser.parse_args()
print('opt = %s' % opt)
data_dir = opt.data_dir
print('data_dir = %s' % data_dir)
print('opt.mode = %s' % opt.mode)

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
    # print('ditc_label = %s' % dict_label)
    return dict_label



def move_cam_image_to_train(original_path, src_path, dst_path, manner='id'):
    cnt = 0
    # files = os.listdir(src_path)
    print('src_path = %s' % src_path)
    print('dst_path = %s' % dst_path)
    files = glob.glob(os.path.join(src_path, '*', '*.jpg'))
    selected_file_num = int(opt.mode * 10000)
    print('total_file_num = %d' % len(files))
    print('selected_file_num = %d' % selected_file_num)
    files = np.random.choice(files, selected_file_num, replace=False)
    print('real selected_file_num = %d' % len(files))
    dict_label = get_dict(original_path)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(original_path, dst_path)
    if manner == 'id':
        dirs = os.listdir(src_path)
        np.random.shuffle(dirs)
        for dir in dirs:
            shutil.copytree(os.path.join(src_path, dir), os.path.join(dst_path, dir))
            cnt += len(os.listdir(os.path.join(src_path, dir)))
            if cnt > selected_file_num:
                break
    elif manner == 'dcgan':
        gen_dir = 'dgen'
        os.mkdir(os.path.join(dst_path, gen_dir))
        for file in files:
            shutil.copy(file, os.path.join(dst_path, gen_dir, os.path.split(file)[-1]))
            cnt += 1
    elif manner == 'cyclegan':
        for file in files:
            file = os.path.split(file)[-1]
            dir = dict_label[file[:4]]
            if not os.path.exists(os.path.join(dst_path, dir)):
                os.mkdir(os.path.join(dst_path, dir))
            shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, dir, file))
            # src = cv2.imread(os.path.join(src_path, file))
            # dst_f = cv2.resize(src, (64, 128))
            # cv2.imwrite(os.path.join(dst_path, dir, file), dst_f)
            cnt += 1
    else:
        print('manner %s is wrong !!!' % manner)
        exit()

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



def resize_rename(src_path, dst_path):
    dirs = os.listdir(src_path)
    for dir in dirs:
        cnt = 0
        if 'camstyle' in dir:
            if not os.path.exists(os.path.join(dst_path, dir)):
                os.mkdir(os.path.join(dst_path, dir))
            files = glob.glob(os.path.join(src_path, dir, '*.jpg'))
            for file in files:
                file = os.path.split(file)[-1]
                # print(os.path.join(src_path, dir, file))
                src = cv2.imread(os.path.join(src_path, dir, file))
                dst_f = cv2.resize(src, (64, 128), interpolation=cv2.INTER_CUBIC)
                # print(os.path.join(dst_path, dir, file[:-4]+dir[-10:]+'.jpg'))
                cv2.imwrite(os.path.join(dst_path, dir, file[:-4]+dir[-10:]+'.jpg'), dst_f)
                cnt += 1
            print('dir = %s  cnt = %d' % (dir, cnt))


if __name__ == '__main__':

    train_new_original_path = 'data/market/pytorch/train_new_original'
    train_new_dst_path = 'data/market/pytorch/train_new'
    aug_src_path = 'data/market/pytorch/id_all'
    # get_dict(train_new_original_path)
    # src_base_path = '/home/dl/cf/reid_gan/data/market/pytorch/resize_rename'


    # dirs = os.listdir(src_base_path)
    # for dir in dirs:
    #     move_cam_image_to_train(os.path.join(src_base_path, dir), train_new_path)
    move_cam_image_to_train(train_new_original_path, aug_src_path, train_new_dst_path)
    # orignal_camstyle_path = 'CycleGAN-for-CamStyle_guider/results/market'
    # new_camstyle_path = 'CycleGAN-for-CamStyle_guider/results/market/resize_rename'
    # resize_rename(orignal_camstyle_path, new_camstyle_path)
