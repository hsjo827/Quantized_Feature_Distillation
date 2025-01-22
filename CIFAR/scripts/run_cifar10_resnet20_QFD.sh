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



if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --epochs 1200

elif [ $METHOD_TYPE == "Qfeature_1bits_STE/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 2 \
                    --baseline True \
                    --use_hessian False \
                    --update_every 10 \
                    --quan_method STE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_2bits_STE/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 4 \
                    --baseline True \
                    --use_hessian False \
                    --update_every 10 \
                    --quan_method STE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_3bits_STE/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 8 \
                    --baseline True \
                    --use_hessian False \
                    --update_every 10 \
                    --quan_method STE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_4bits_STE/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 16 \
                    --baseline True \
                    --use_hessian False \
                    --update_every 10 \
                    --quan_method STE \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_1bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 2 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --quan_method EWGS \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40


elif [ $METHOD_TYPE == "Qfeature_2bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --quan_method EWGS \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_3bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 8 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --quan_method EWGS \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "Qfeature_4bits_EWGS/" ] 
then
    python3 train_fp_to_feature_quant.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --QFeatureFlag True \
                    --feature_levels 16 \
                    --baseline False \
                    --use_hessian True \
                    --update_every 10 \
                    --quan_method EWGS \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                    --epochs 40

elif [ $METHOD_TYPE == "FeatureKD_T1_W1A1_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --feature_levels 2 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_1bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T2_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T3_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T4_W2A2_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T2_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T3_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T4_W3A3_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T2_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T3_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T4_W4A4_STE/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline True \
                        --use_hessian False \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_STE/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T1_W1A1_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --feature_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_1bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T2_W2A2_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T3_W2A2_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200  

elif [ $METHOD_TYPE == "FeatureKD_T4_W2A2_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --feature_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200    

elif [ $METHOD_TYPE == "FeatureKD_T2_W3A3_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T3_W3A3_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T4_W3A3_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --feature_levels 8 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200

elif [ $METHOD_TYPE == "FeatureKD_T2_W4A4_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_2bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200 

elif [ $METHOD_TYPE == "FeatureKD_T3_W4A4_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_3bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200 

elif [ $METHOD_TYPE == "FeatureKD_T4_W4A4_EWGS/" ] 
then
    python3 train_quant_with_featureKD.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --feature_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/Qfeature_4bits_EWGS/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.5 \
                        --kd_alpha 0.0 \
                       --kd_beta 0.0 \
                        --epochs 200 
                        
fi




# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"