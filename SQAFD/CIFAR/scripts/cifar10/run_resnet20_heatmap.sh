#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FQA+EWGS
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


### Heatmap Distillation
if [ $METHOD_TYPE == "Heatmap_T01_S01_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_01' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_S01_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_01' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_S01_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_01' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200


elif [ $METHOD_TYPE == "Heatmap_T01_S0p_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_S0p_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_S0p_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_Spm_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_Spm_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T01_Spm_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_01' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T0p_S0p_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T0p_S0p_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T0p_S0p_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_0p' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T0p_Spm_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_T0p_Spm_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200                        

elif [ $METHOD_TYPE == "Heatmap_T0p_Spm_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_0p' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_Tpm_Spm_1bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 2 \
                    --act_levels 2 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_pm' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200      

elif [ $METHOD_TYPE == "Heatmap_Tpm_Spm_2bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_pm' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
                    --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet20_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500 \
                    --epochs 1200

elif [ $METHOD_TYPE == "Heatmap_Tpm_Spm_4bit/" ]
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                    --arch 'resnet20_quant' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam'\
                    --weight_levels 16 \
                    --act_levels 16 \
                    --baseline False \
                    --use_hessian True\
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --transform_type_t 'binary_pm' \
                    --transform_type_s 'binary_pm' \
                    --use_heatmap_distillation True \
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