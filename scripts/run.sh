CUDA_VISIBLE_DEVICES=0 python tools/LinearProbing.py \
 --dataset imagenet \
 --data_folder ./data/imagenet \
 --save_path ./output \
 --tb_path ./output \
 --model_path ./ckpts/cmc_models/resnet50v2.pth \
 --model resnet50v2 --learning_rate 30 --layer 6 \
 --gpu 0

CUDA_VISIBLE_DEVICES=0 python tools/train_CMC.py \
 --batch_size 256 --num_workers 8 \
 --data_folder ./data/imagenet \
 --model_path ./output \
 --tb_path ./output 

python tools/train_moco_ins.py \
 --batch_size 64 --num_workers 8 --nce_k 16384 --softmax --moco \
 --model_path ./output \
 --tb_path ./output \
 --gpu 0


python tools/train_moco_ins_homo.py \
 --batch_size 64 --num_workers 0 --nce_k 16384 --softmax --moco \
 --model_path ./output \
 --tb_path ./output \
 --gpu 0