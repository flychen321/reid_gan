#!/bin/bash
#python train_baseline_original.py --use_dense
#for((i=1;i<=6;i++))
#do
#    for((j=i;j<=6;j++))
#    do
#    #    python test.py  --use_dense  --ratio  $a
#    #    python evaluate.py
#    #    python evaluate_rerank.py
##        echo $i,$j > analysis_log_$i'and'$j
##        python analysis_cam.py > analysis_log_$a
#        pwd
#        sleep 10
#
#    done
#done


python train_baseline.py --use_dense
python test.py  --use_dense
python evaluate.py
python evaluate_rerank.py