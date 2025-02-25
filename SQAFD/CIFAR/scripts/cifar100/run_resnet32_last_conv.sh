    #!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: ResNet-32
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


### FQ + use student quant params + last conv                    
if [ $METHOD_TYPE == "FQ_last_conv_1bit/" ]
then
    python3 train_quant_distill.py --gpu_id '1' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 2 \
                    --act_levels 2 \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter False \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720

elif [ $METHOD_TYPE == "FQ_last_conv_2bit/" ]
then
    python3 train_quant_distill.py --gpu_id '1' \
                    --dataset 'cifar100'\
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter False \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720

elif [ $METHOD_TYPE == "FQ_last_conv_4bit/" ]
then
    python3 train_quant_distill.py --gpu_id '1' \
                    --dataset 'cifar100'\
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter False \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720


### FQA + use student quant params + last conv                    
elif [ $METHOD_TYPE == "FQA(s)_last_conv_1bit/" ]
then
    python3 train_quant_distill.py --gpu_id '2' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 2 \
                    --act_levels 2 \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter True \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720

elif [ $METHOD_TYPE == "FQA(s)_last_conv_2bit/" ]
then
    python3 train_quant_distill.py --gpu_id '2' \
                    --dataset 'cifar100'\
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 4 \
                    --act_levels 4 \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter True \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720

elif [ $METHOD_TYPE == "FQA(s)_last_conv_4bit/" ]
then
    python3 train_quant_distill.py --gpu_id '2' \
                    --dataset 'cifar100'\
                    --arch 'resnet32_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --lr_m 5e-4 \
                    --lr_a 5e-4 \
                    --lr_q 5e-6 \
                    --lr_c 5e-7 \
                    --weight_levels 16 \
                    --act_levels 16 \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True \
                    --use_student_quant_params True \
                    --use_adapter True \
                    --train_last_conv True \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --distill 'fd' \
                    --fd_type 'L2' \
                    --teacher_path './results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                    --teacher_arch 'resnet32_fp' \
                    --train_mode 'student' \
                    --kd_gamma 1 \
                    --kd_alpha 500\
                    --epochs 720

fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"