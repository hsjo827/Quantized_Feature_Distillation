#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP(EWGS), T1_STE, T2_STE, T3_STE, T4_STE
# Bit-width: T1, T2, T3, T4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE


### Full-precision model
if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --epochs 400

### Train Quantized Teacher model
### STE
elif [ $METHOD_TYPE == "Teacher_quant/T1_STE_SGD/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 2 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'SGD' \
                    --optimizer_q 'SGD' \


elif [ $METHOD_TYPE == "Teacher_quant/T1_STE_Adam/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 2 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \


elif [ $METHOD_TYPE == "Teacher_quant/T2_STE_SGD/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 4 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'SGD' \
                    --optimizer_q 'SGD' \


elif [ $METHOD_TYPE == "Teacher_quant/T2_STE_Adam/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 4 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \


elif [ $METHOD_TYPE == "Teacher_quant/T3_STE_SGD/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 8 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'SGD' \
                    --optimizer_q 'SGD' \


elif [ $METHOD_TYPE == "Teacher_quant/T3_STE_Adam/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 8 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \


elif [ $METHOD_TYPE == "Teacher_quant/T4_STE_SGD/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 16 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'SGD' \
                    --optimizer_q 'SGD' \

elif [ $METHOD_TYPE == "Teacher_quant/T4_STE_SGD/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 16 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \


elif [ $METHOD_TYPE == "Teacher_quant/T8_STE_SGD/" ]
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 256 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'SGD' \
                    --optimizer_q 'SGD' \


elif [ $METHOD_TYPE == "Teacher_quant/T8_STE_Adam/" ]
then
    python3 train_fp_to_feature_quant.py --gpu_id '2' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --feature_levels 256 \
                    --quan_method 'STE' \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \

                        
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"