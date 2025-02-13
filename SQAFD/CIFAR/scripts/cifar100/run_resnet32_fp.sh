#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: ResNet-32
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP
# Bit-width: W1A1, W2A2, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE



if [ $METHOD_TYPE == "fp_crd_cosine/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE


elif [ $METHOD_TYPE == "Qfeature_1bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-2 \
                    --lr_q 5e-4 \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --train_mode 'teacher' \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10\
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --epochs 72


elif [ $METHOD_TYPE == "Qfeature_2bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-2 \
                    --lr_q 5e-4 \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --train_mode 'teacher' \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --epochs 72

elif [ $METHOD_TYPE == "Qfeature_4bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-2 \
                    --lr_q 5e-4 \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --train_mode 'teacher' \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --epochs 72
fi



# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"