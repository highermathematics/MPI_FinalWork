#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from Dowload_Data import download_all, download
from Access_And_Read import load_kaggle_house_data, explore_data, analyze_target_variable
from PreData import preprocess_data, check_data_quality, save_preprocessed_data
from Train import (
    get_net, get_advanced_net, log_rmse, train, 
    k_fold, train_and_predict, save_predictions
)

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Kaggle房价预测项目                          ║
    ║                  House Price Prediction                      ║
    ║                                                              ║
    ║  一个完整的机器学习项目，包含数据处理、模型训练和预测功能        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_environment():
    """设置项目环境"""
    # 创建必要的目录
    directories = ['../data', '../models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ 项目环境设置完成")

def download_data_pipeline():
    """数据下载流水线"""
    print("\n=== 步骤1: 数据下载 ===")
    try:
        # 检查数据是否已存在
        train_file = os.path.join('..', 'data', 'kaggle_house_pred_train.csv')
        test_file = os.path.join('..', 'data', 'kaggle_house_pred_test.csv')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("✓ 数据文件已存在，跳过下载")
            return True
        
        print("正在下载数据集...")
        download_all()
        print("✓ 数据下载完成")
        return True
    except Exception as e:
        print(f"✗ 数据下载失败: {e}")
        return False

def load_and_explore_data():
    """数据加载和探索流水线"""
    print("\n=== 步骤2: 数据加载和探索 ===")
    try:
        # 加载数据
        train_data, test_data = load_kaggle_house_data()
        print("✓ 数据加载完成")
        
        # 探索数据
        explore_data(train_data, test_data)
        
        # 分析目标变量
        analyze_target_variable(train_data)
        
        return train_data, test_data
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None, None

def preprocess_data_pipeline(train_data, test_data):
    """数据预处理流水线"""
    print("\n=== 步骤3: 数据预处理 ===")
    try:
        # 预处理数据
        result = preprocess_data(train_data, test_data, verbose=True)
        train_features, test_features, train_labels, all_features = result
        
        # 检查数据质量
        check_data_quality(train_features, test_features, train_labels)
        
        # 保存预处理后的数据
        save_preprocessed_data(train_features, test_features, train_labels, all_features)
        
        print("✓ 数据预处理完成")
        return train_features, test_features, train_labels, all_features
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return None, None, None, None

def train_model_pipeline(train_features, test_features, train_labels, test_data, 
                        model_type='linear', use_kfold=True):
    """模型训练流水线"""
    print(f"\n=== 步骤4: 模型训练 ({model_type}) ===")
    try:
        # 训练参数
        num_epochs = 100
        lr = 5
        weight_decay = 0
        batch_size = 64
        k = 5  # K折交叉验证的折数
        
        if use_kfold:
            print(f"使用{k}折交叉验证训练模型...")
            train_l, valid_l = k_fold(
                k, train_features, train_labels, 
                num_epochs, lr, weight_decay, batch_size
            )
            print(f"✓ K折交叉验证完成")
            print(f"平均训练log rmse: {train_l:.6f}")
            print(f"平均验证log rmse: {valid_l:.6f}")
        
        # 训练最终模型并生成预测
        print("训练最终模型并生成预测...")
        preds, net = train_and_predict(
            train_features, test_features, train_labels, test_data,
            num_epochs, lr, weight_decay, batch_size, model_type
        )
        
        # 保存预测结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'results/submission_{model_type}_{timestamp}.csv'
        save_predictions(test_data, preds, filename)
        
        print(f"✓ 模型训练完成，预测结果已保存到: {filename}")
        return preds, net
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return None, None

def run_complete_pipeline(model_type='linear', use_kfold=True, skip_download=False):
    """运行完整的机器学习流水线"""
    start_time = time.time()
    
    print_banner()
    setup_environment()
    
    # 步骤1: 数据下载
    if not skip_download:
        if not download_data_pipeline():
            return False
    
    # 步骤2: 数据加载和探索
    train_data, test_data = load_and_explore_data()
    if train_data is None:
        return False
    
    # 步骤3: 数据预处理
    train_features, test_features, train_labels, all_features = preprocess_data_pipeline(
        train_data, test_data
    )
    if train_features is None:
        return False
    
    # 步骤4: 模型训练
    preds, net = train_model_pipeline(
        train_features, test_features, train_labels, test_data,
        model_type, use_kfold
    )
    if preds is None:
        return False
    
    # 完成
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🎉 项目执行完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"预测结果已生成，可以提交到Kaggle进行评估")
    print(f"{'='*60}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Kaggle房价预测项目')
    parser.add_argument('--model', type=str, default='linear', 
                       choices=['linear', 'advanced'],
                       help='模型类型 (default: linear)')
    parser.add_argument('--no-kfold', action='store_true',
                       help='跳过K折交叉验证')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过数据下载（假设数据已存在）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（跳过下载和K折验证）')
    
    args = parser.parse_args()
    
    # 快速模式设置
    if args.quick:
        args.skip_download = True
        args.no_kfold = True
        print("🚀 快速模式已启用")
    
    try:
        success = run_complete_pipeline(
            model_type=args.model,
            use_kfold=not args.no_kfold,
            skip_download=args.skip_download
        )
        
        if success:
            print("\n✅ 项目执行成功！")
            return 0
        else:
            print("\n❌ 项目执行失败！")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        return 1
    except Exception as e:
        print(f"\n💥 未预期的错误: {e}")
        return 1

if __name__ == "__main__":
    # 如果没有命令行参数，显示帮助信息
    if len(sys.argv) == 1:
        print("\n使用示例:")
        print("python main.py                    # 使用默认设置运行")
        print("python main.py --model advanced   # 使用高级模型")
        print("python main.py --quick            # 快速模式")
        print("python main.py --no-kfold         # 跳过K折验证")
        print("python main.py --skip-download    # 跳过数据下载")
        print("python main.py --help             # 显示帮助信息")
        print()
    
    exit_code = main()
    sys.exit(exit_code)
