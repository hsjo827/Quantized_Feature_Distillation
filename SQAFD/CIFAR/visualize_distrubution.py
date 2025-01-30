import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from torch.utils.data import DataLoader
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from models.custom_models_resnet import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze feature maps distribution')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--teacher_path', type=str, required=True, help='path to teacher checkpoint')
    parser.add_argument('--student_path', type=str, required=True, help='path to student checkpoint')
    parser.add_argument('--teacher_arch', type=str, default='resnet20_fp', help='teacher architecture')
    parser.add_argument('--student_arch', type=str, default='resnet20_quant', help='student architecture')
    parser.add_argument('--save_dir', type=str, default='./feature_analysis', help='directory to save results')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for testing')
    
    # Add required arguments for model initialization
    parser.add_argument('--QWeightFlag', type=str2bool, default=True)
    parser.add_argument('--QActFlag', type=str2bool, default=True)
    parser.add_argument('--QFeatureFlag', type=str2bool, default=True)
    parser.add_argument('--weight_levels', type=int, default=2)
    parser.add_argument('--act_levels', type=int, default=2)
    parser.add_argument('--feature_levels', type=int, default=2)
    parser.add_argument('--baseline', type=str2bool, default=False)
    parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0)
    parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0)
    parser.add_argument('--bkwd_scaling_factorF', type=float, default=0.0)
    
    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_model(model_path, arch, args, device):
    """Load model from checkpoint"""
    model_class = globals().get(arch)
    model = model_class(args)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model

def extract_features(model, data_loader, device):
    """Extract features from all test data"""
    all_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc='Extracting features'):
            images = images.to(device)
            features = model.feature_extractor(images)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)

def plot_distributions(teacher_features, student_features, save_dir, args):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Teacher histogram
    if args.teacher_arch.endswith('quant'):
        n_bins_teacher = args.act_levels
    else:
        n_bins_teacher = 100
        
    # Student histogram
    if args.student_arch.endswith('quant'):
        n_bins_student = args.act_levels
    else:
        n_bins_student = 100

    plt.hist(teacher_features.flatten(), bins=n_bins_teacher, alpha=0.5, 
            label=f'Teacher ({args.teacher_arch})', density=True, color='blue')
    plt.hist(student_features.flatten(), bins=n_bins_student, alpha=0.5, 
            label=f'Student ({args.student_arch})', density=True, color='red')
    
    plt.title(f'Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'feature_distribution.png'))
    plt.close()

    # Print unique values
    print(f"\nUnique values in distribution:")
    print(f"Teacher unique values: {len(np.unique(teacher_features))}")
    print(f"Student unique values: {len(np.unique(student_features))}")

def main():
    args = parse_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로더 설정
    if args.dataset == 'cifar10':
        args.num_classes = 10
        _, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")
    else:
        args.num_classes = 100
        _, test_dataset = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # 모델 로드
    print("Loading teacher model...")
    teacher = load_model(args.teacher_path, args.teacher_arch, args, device)
    
    print("Loading student model...")
    student = load_model(args.student_path, args.student_arch, args, device)
    
    # Feature map 추출
    print("Extracting teacher features...")
    teacher_features = extract_features(teacher, test_loader, device)
    
    print("Extracting student features...")
    student_features = extract_features(student, test_loader, device)
    
    # 분포 시각화
    print("Plotting distributions...")
    plot_distributions(teacher_features, student_features, args.save_dir, args)

if __name__ == '__main__':
    main()