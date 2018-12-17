import os
import numpy as np

value = [[60, 80], [70, 90], [60, 90]]
for i in np.arange(len(value)):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)
    cmd = 'python train_baseline.py --use_dense  --modelname ' + 'prob_' + str(i) + ' --prob ' + str(i) + ' --min ' + str(value[i][0]) + ' --max ' + str(value[i][1]) + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system(cmd)
    os.system('python test.py  --use_dense' + ' >> ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)
