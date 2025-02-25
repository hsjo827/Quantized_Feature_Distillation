#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP, STE+QFD, EWGS+QFD
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

### Train QFD Student model
### 1-bit teacher
if [ $METHOD_TYPE == "QFD/T1_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T1_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T1_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T1_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T1_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T1_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

### 2-bit teacher
elif [ $METHOD_TYPE == "QFD/T2_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T2_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T2_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T2_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T2_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '2' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T2_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

### 3-bit teacher
elif [ $METHOD_TYPE == "QFD/T3_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T3_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T3_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T3_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T3_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T3_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

### 4-bit teacher
elif [ $METHOD_TYPE == "QFD/T4_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T4_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T4_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T4_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \

elif [ $METHOD_TYPE == "QFD/T4_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '3' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Teacher_quant/T4_STE_Adam/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        
fi




# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"