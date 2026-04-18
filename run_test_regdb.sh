CUDA_VISIBLE_DEVICES=4,5 \
 python test_regdb.py -b 128 -a agw -d  regdb_rgb \
 --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
 --data-dir "/data/wml/dataset/RegDB/"
