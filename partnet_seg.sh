category=Faucet
run_id=0
fewshot=10
lamb=0.001

python train_seg_partnet.py \
--exp_name partnet_seg2d_pretrained_$category_$lamb \
--run_id $run_id \
--data_dir partnet/renderings/ \ # self sup rendering folder
--selfsup_data_dir partnet/renderings/ \ # self sup rendering folder
--test_data_dir partnet/partnet_dataset/ \ # segmentation dataset folder
--batch_size 8 \
 --test_batch_size 2 \
 --up_layers True \
 --epochs 100 \
 --patience 5 \
 --lr 0.001 \
 --all_imgs True \
 --neg_samples 4000 \
 --include_depth True \
 --include_normal True \
 --num_classes 50 \
 --emb_dims 64 \
 --train_workers 1 \
 --test_workers 1 \
 --sampling 0 \
 --augment False \
 --num_points 5000 \
 --multi_gpus \
 --gamma 0.99 \
 --train_file_name train_val_$category.json \
 --train_label_file_name train_$category.json \
 --category $category \
 --up_seg_layers 1 \
 --few_shots $fewshot \
 --voting entropy \
 --few_images -1 \
 --lamb $lamb \
 --pretrained_model "pretrained_model_name.pth"

