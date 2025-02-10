#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: ResNet-32
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: AFQ+EWGS
# Bit-width: T1, T2, T4, W1A1, W2A2, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE


if [ $METHOD_TYPE == "AFQ_T1_W1A1_use_student_quant_params/"]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200

elif [ $METHOD_TYPE == "AFQ_T2_W2A2_use_student_quant_params/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200

elif [ $METHOD_TYPE == "AFQ_T4_W4A4_use_student_quant_params/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200

### AFQ + not use_student_quant_params 
elif [ $METHOD_TYPE == "AFQ_T1_W1A1/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params False \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/Teacher_Quant/T1bit_EWGS/last_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200

elif [ $METHOD_TYPE == "AFQ_T2_W2A2/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params False \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/Teacher_Quant/T2bit_EWGS/last_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200
elif [ $METHOD_TYPE == "AFQ_T4_W4A4/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet32_quant' \
                    --dataset 'cifar100'\
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --TFeatureOder 'AFQ' \
                    --use_student_quant_params False \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_sqakd/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR100_ResNet32/Teacher_Quant/T4bit_EWGS/last_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 1200
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"