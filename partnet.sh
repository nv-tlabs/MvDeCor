python pretrain_partnet.py \
       --exp_name pretrain_partnet \
       --data_dir partnet/renderings/ \
       --batch_size 16 \
       --test_batch_size 2 \
       --epochs 300 \
       --patience 5 \
       --lr 0.001 \
       --all_imgs True \
       --neg_samples 4000 \
       --include_depth True \
       --include_normal True \
       --num_classes 50 \
       --emb_dims 64 \
       --train_workers 10 \
       --test_workers 2 \
       --sampling 0 \
       --augment False \
       --num_points 5000 \
       --multi_gpus \
       --up_layers True \
       --max_num_images 86 \
       --train_file_name train_val_all.json


