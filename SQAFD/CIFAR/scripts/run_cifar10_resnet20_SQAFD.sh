#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: SQAFD+STE, SQAFD+EWGS
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


### SQAFD+STE
if [ $METHOD_TYPE == "SQAKD_T2_W2A2_STE/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline True \
                    --use_hessian False\
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.5 \
                    --epochs 1200

elif [ $METHOD_TYPE == "SQAKD_T3_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 8 \
                    --act_levels 8 \
                    --feature_levels 8 \
                    --baseline True \
                    --use_hessian False\
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.5 \
                    --epochs 1200

elif [ $METHOD_TYPE == "SQAKD_T4_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline True \
                    --use_hessian False\
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.5 \
                    --epochs 1200

### SQAFD+EWGS
elif [ $METHOD_TYPE == "SQAKD_T2_W2A2_EWGS/" ]
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
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.5 \
                    --epochs 1200

elif [ $METHOD_TYPE == "SQAKD_T3_W3A3_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 8 \
                    --act_levels 8 \
                    --feature_levels 8 \
                    --baseline False \
                    --use_hessian True\
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.5 \
                    --epochs 1200

elif [ $METHOD_TYPE == "SQAKD_T4_W4A4_EWGS/" ] 
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
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/last_checkpoint.pth' \
                    --distill 'kd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp_sqakd/checkpoint/last_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --kd_gamma 0.8 \
                    --epochs 1200
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"