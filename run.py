import os
import numpy as np

for i in np.arange(100, 0, -5):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)
    cmd = 'python train_baseline.py --use_dense  --modelname ' + 'prob_' + str(i) + ' --prob ' + str(i) + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system('python train_baseline.py --use_dense  --modelname ' + 'prob_' + str(i) + ' --prob ' + str(i) + ' >> ' + log_name)
    os.system('python test.py  --use_dense' + ' >> ' + log_name)
    os.system('python evaluate.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)
