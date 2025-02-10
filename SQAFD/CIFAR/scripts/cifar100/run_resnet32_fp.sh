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



if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'resnet32_fp' \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --epochs 1200

elif [ $METHOD_TYPE == "Qfeature_1bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet32_fp' \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10\
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/Teacher_Quant/T1bit_EWGS/last_checkpoint.pth' \
                    --epochs 120


elif [ $METHOD_TYPE == "Qfeature_2bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet32_fp' \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/Teacher_Quant/T2bit_EWGS/last_checkpoint.pth' \
                    --epochs 120

elif [ $METHOD_TYPE == "Qfeature_4bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet32_fp' \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/Teacher_Quant/T4bit_EWGS/last_checkpoint.pth' \
                    --epochs 120