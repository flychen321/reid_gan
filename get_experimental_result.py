import os
import pandas as pd
from pandas import DataFrame
from glob import glob


def get_one_result(path):
    # data = pd.read_excel("result.xlsx", sheet_name='Sheet1')
    # data = {'name':['fly','yang'], 'age':[25,14]}
    files = glob('/home/dl/cf/reid_gan/log/*/*')
    print(len(files))
    name = []
    epoc = []
    val_loss = []
    val_acc = []
    rank_1 = []
    rank_5 = []
    rank_10 = []
    map_ = []
    rerank_1 = []
    rerank_5 = []
    rerank_10 = []
    remap = []
    ''' 
        train Loss: 0.0018 Acc: 0.9946
        val Loss: 0.0138 Acc: 0.9174
        Training complete in 140m 59s
        Best val epoch: 84
        Best val Loss: 0.0139  Acc: 0.924101
        -------test-----------
        top1:0.918349 top5:0.971793 top10:0.985451 mAP:0.789647
        calculate initial distance
        Reranking complete in 1m 4s
        top1:0.931413 top5:0.965855 top10:0.977138 mAP:0.904607
    '''
    for file in files:
        print(file)
        f = open(file, 'r')
        r = f.readlines()
        # print(file.split('/')[-2]+'/'+file.split('/')[-1])
        if len(r) < 10:
            print('len(r) = %s' % len(r))
            continue
        if 'Best val epoch' not in r[-7]:
            print('Best val epoch not in r[-7]')
            continue
        if 'Best val Loss' not in r[-6]:
            print('Best val Loss not in r[-6]')
            continue
        if 'top1:' not in r[-4]:
            print('top1: not in r[-4]')
            continue
        if 'top1:' not in r[-1]:
            print('top1: not in r[-4]')
            continue

        name.append(file.split('/')[-2] + '/' + file.split('/')[-1])
        if 'Best val epoch' in r[-7]:
            # print(r[-7].split(':')[-1].strip())
            epoc.append(r[-7].split(':')[-1].strip())
        if 'Best val Loss' in r[-6]:
            # print(r[-6].split(':')[1].strip().split('A')[0].strip())
            # print(r[-6].split(':')[2].strip())
            val_loss.append(r[-6].split(':')[1].strip().split('A')[0].strip())
            val_acc.append(r[-6].split(':')[2].strip())
        if 'top1:' in r[-4]:
            # print(r[-4].split(':')[1].split('t')[0].strip())
            # print(r[-4].split(':')[2].split('t')[0].strip())
            # print(r[-4].split(':')[3].split('m')[0].strip())
            # print(r[-4].split(':')[4].strip())
            rank_1.append(r[-4].split(':')[1].split('t')[0].strip())
            rank_5.append(r[-4].split(':')[2].split('t')[0].strip())
            rank_10.append(r[-4].split(':')[3].split('m')[0].strip())
            map_.append(r[-4].split(':')[4].strip())
        if 'top1:' in r[-1]:
            # print(r[-1].split(':')[1].split('t')[0].strip())
            # print(r[-1].split(':')[2].split('t')[0].strip())
            # print(r[-1].split(':')[3].split('m')[0].strip())
            # print(r[-1].split(':')[4].strip())
            rerank_1.append(r[-1].split(':')[1].split('t')[0].strip())
            rerank_5.append(r[-1].split(':')[2].split('t')[0].strip())
            rerank_10.append(r[-1].split(':')[3].split('m')[0].strip())
            remap.append(r[-1].split(':')[4].strip())

    data = {'name': name, 'epoc': epoc, 'val_loss': val_loss, 'val_acc': val_acc, 'rank_1': rank_1, 'rank_5': rank_5,
            'rank_10': rank_10, 'map': map_, 'rerank_1': rerank_1, 'rerank_5': rerank_5, 'rerank_10': rerank_10,
            'remap': remap}
    print(data)
    frame = DataFrame(data)
    print(frame)
    frame.to_excel('/home/dl/cf/reid_gan/log/result.xlsx')


if __name__ == '__main__':
    path = '/home/dl/cf/reid_gan/log/log_0'
    get_one_result(path)
