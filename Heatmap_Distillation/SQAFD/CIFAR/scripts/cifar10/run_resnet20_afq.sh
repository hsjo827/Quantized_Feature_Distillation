#!/bin/bash

####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: AFQ+EWGS
# Bit-width: T2, T3, T4, W2A2, W3A3, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE

### AFQ + use student quant params
if [ $METHOD_TYPE == "AFQ_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --use_adapter_t True \
                    --use_adapter_s True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "AFQ_2bit/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --use_adapter_t True \
                    --use_adapter_s True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200
                    
elif [ $METHOD_TYPE == "AFQ_4bit/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --use_adapter_t True \
                    --use_adapter_s True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"
