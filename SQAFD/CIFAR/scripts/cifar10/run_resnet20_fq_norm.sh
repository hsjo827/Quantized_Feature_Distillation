#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FQ+Normalize+EWGS
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


### FQ + use studentquant params
if [ $METHOD_TYPE == "FQ_T1_W1A1_use_student_quant_params/" ]
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
                    --use_student_quant_params True \
                    --use_adapter_t False \
                    --use_adapter_s False \
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

elif [ $METHOD_TYPE == "FQ_T2_W2A2_use_student_quant_params/" ]
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
                    --use_adapter_t False \
                    --use_adapter_s False \
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

elif [ $METHOD_TYPE == "FQ_T4_W4A4_use_student_quant_params/" ]
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
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_alpha 500 \
                    --epochs 1200


### FQ + not use student quant params
elif [ $METHOD_TYPE == "FQ_T1_W1A1/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
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

elif [ $METHOD_TYPE == "FQ_T2_W2A2/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
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

elif [ $METHOD_TYPE == "FQ_T4_W4A4/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
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

### FQ + Normalize + use studentquant params
elif [ $METHOD_TYPE == "FQ_T1_W1A1_normalize_use_student_quant_params/" ]
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
                    --use_student_quant_params True \
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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

elif [ $METHOD_TYPE == "FQ_T2_W2A2_normalize_use_student_quant_params/" ]
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
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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

elif [ $METHOD_TYPE == "FQ_T4_W4A4_normalize_use_student_quant_params/" ]
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
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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


### FQ + Normalize + not use student quant params
elif [ $METHOD_TYPE == "FQ_T1_W1A1_normalize/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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

elif [ $METHOD_TYPE == "FQ_T2_W2A2_normalize/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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


elif [ $METHOD_TYPE == "FQ_T4_W4A4_normalize/" ]
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
                    --use_student_quant_params False \
                    --use_adapter_t False \
                    --use_adapter_s False \
                    --use_map_norm True \
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
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"