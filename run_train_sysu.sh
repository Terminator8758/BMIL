CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py --use_global_clustering -mb CMhybrid --sample_ratio 0.5 --epochs 50 -b 128 -a agw -d sysu_all \
--iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
--data-dir "/data/wml/dataset/SYSU-MM01/"

