import os
import numpy as np

value = [[0, 100], [70, 90], [60, 80], [60, 90], [50, 100], [80, 90], [70, 80], [80, 80], [90, 90], ]
for i in np.arange(3, 6):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)
    # cmd = 'python train_baseline.py --use_dense  --modelname ' + 'prob_' + str(i) + ' --prob ' + str(
    #     i) + ' --min ' + str(value[i][0]) + ' --max ' + str(value[i][1]) + ' >> ' + log_name

    os.system('python move_from_camstyle.py --mode ' + str(i) + ' >>  ' + log_name)
    cmd = 'python train_without_gan.py --use_dense  --modelname ' + 'prob_' + str(i) + ' --prob ' + str(
        i) + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system(cmd)
    os.system('python test.py  --use_dense --which_epoch best  >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)
    os.system('python test.py  --use_dense --which_epoch last  >>  ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

