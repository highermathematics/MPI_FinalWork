import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入之前定义的数据加载函数
from Access_And_Read import load_kaggle_house_data

def preprocess_data(train_data, test_data, verbose=True):
    """
    完整的数据预处理流程
    
    参数:
    train_data: 训练数据
    test_data: 测试数据
    verbose: 是否打印详细信息
    
    返回:
    train_features: 处理后的训练特征张量
    test_features: 处理后的测试特征张量
    train_labels: 训练标签张量
    all_features: 处理后的所有特征DataFrame
    """
    if verbose:
        print("=== 开始数据预处理 ===")
    
    # 1. 合并训练和测试数据的特征（除了ID和标签）
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    
    if verbose:
        print(f"合并后特征数据形状: {all_features.shape}")
        print(f"原始特征数量: {all_features.shape[1]}")
    
    # 2. 处理数值特征
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    
    if verbose:
        print(f"数值特征数量: {len(numeric_features)}")
        print("数值特征示例:", list(numeric_features[:10]))
    
    # 标准化数值特征: x ← (x - μ) / σ
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    
    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    
    if verbose:
        print("数值特征标准化完成")
    
    # 3. 处理类别特征
    categorical_features = all_features.dtypes[all_features.dtypes == 'object'].index
    
    if verbose:
        print(f"类别特征数量: {len(categorical_features)}")
        print("类别特征示例:", list(categorical_features[:10]))
    
    # 独热编码处理类别特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    
    if verbose:
        print(f"独热编码后特征数量: {all_features.shape[1]}")
        print(f"特征数量从 {len(numeric_features) + len(categorical_features)} 增加到 {all_features.shape[1]}")
    
    # 4. 确保所有数据都是数值类型
    # 检查并转换任何剩余的非数值列
    for col in all_features.columns:
        if all_features[col].dtype == 'object':
            if verbose:
                print(f"警告: 发现object类型列 {col}，尝试转换为数值类型")
            # 尝试转换为数值类型
            all_features[col] = pd.to_numeric(all_features[col], errors='coerce')
            # 填充转换失败的NaN值
            all_features[col] = all_features[col].fillna(0)
    
    # 确保所有列都是float类型
    all_features = all_features.astype('float32')
    
    if verbose:
        print("数据类型检查和转换完成")
        print(f"最终数据类型: {all_features.dtypes.unique()}")
    
    # 5. 转换为张量格式
    n_train = train_data.shape[0]
    
    try:
        # 转换为PyTorch张量
        train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
        train_labels = torch.tensor(
            train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
        
        if verbose:
            print(f"\n=== 数据预处理完成 ===")
            print(f"训练特征形状: {train_features.shape}")
            print(f"测试特征形状: {test_features.shape}")
            print(f"训练标签形状: {train_labels.shape}")
        
        return train_features, test_features, train_labels, all_features
        
    except Exception as e:
        if verbose:
            print(f"张量转换失败: {e}")
            print("DataFrame信息:")
            print(f"形状: {all_features.shape}")
            print(f"数据类型: {all_features.dtypes.value_counts()}")
            print(f"是否包含NaN: {all_features.isnull().any().any()}")
            print(f"是否包含inf: {np.isinf(all_features.values).any()}")
        raise e

def analyze_preprocessing_effects(all_features, numeric_features_original, verbose=True):
    """
    分析预处理效果
    """
    if verbose:
        print("\n=== 预处理效果分析 ===")
    
    # 检查数值特征的标准化效果
    numeric_cols = [col for col in all_features.columns if not col.endswith('_nan') and 
                   any(orig in col for orig in numeric_features_original[:5])]
    
    if len(numeric_cols) > 0:
        sample_features = all_features[numeric_cols[:5]]
        
        if verbose:
            print("\n标准化后的数值特征统计:")
            print(sample_features.describe())
        
        # 绘制标准化效果
        plt.figure(figsize=(15, 4))
        for i, col in enumerate(numeric_cols[:5]):
            plt.subplot(1, 5, i+1)
            plt.hist(sample_features[col].dropna(), bins=30, alpha=0.7)
            plt.title(f'{col}\n均值: {sample_features[col].mean():.3f}\n标准差: {sample_features[col].std():.3f}')
            plt.xlabel('标准化后的值')
            plt.ylabel('频次')
        
        plt.tight_layout()
        plt.suptitle('标准化后的数值特征分布', y=1.02)
        plt.show()
    
    # 分析独热编码效果
    dummy_cols = [col for col in all_features.columns if '_' in col and not col.endswith('_nan')]
    
    if verbose:
        print(f"\n独热编码生成的特征数量: {len(dummy_cols)}")
        print("独热编码特征示例:", dummy_cols[:10])
    
    return sample_features if len(numeric_cols) > 0 else None

def check_data_quality(train_features, test_features, train_labels, verbose=True):
    """
    检查数据质量
    """
    if verbose:
        print("\n=== 数据质量检查 ===")
    
    # 检查是否有无穷大或NaN值
    train_inf = torch.isinf(train_features).sum().item()
    train_nan = torch.isnan(train_features).sum().item()
    test_inf = torch.isinf(test_features).sum().item()
    test_nan = torch.isnan(test_features).sum().item()
    label_inf = torch.isinf(train_labels).sum().item()
    label_nan = torch.isnan(train_labels).sum().item()
    
    if verbose:
        print(f"训练特征中的无穷大值: {train_inf}")
        print(f"训练特征中的NaN值: {train_nan}")
        print(f"测试特征中的无穷大值: {test_inf}")
        print(f"测试特征中的NaN值: {test_nan}")
        print(f"训练标签中的无穷大值: {label_inf}")
        print(f"训练标签中的NaN值: {label_nan}")
    
    # 检查特征值范围
    if verbose:
        print(f"\n训练特征值范围: [{train_features.min():.3f}, {train_features.max():.3f}]")
        print(f"测试特征值范围: [{test_features.min():.3f}, {test_features.max():.3f}]")
        print(f"训练标签值范围: [{train_labels.min():.3f}, {train_labels.max():.3f}]")
    
    return {
        'train_inf': train_inf,
        'train_nan': train_nan,
        'test_inf': test_inf,
        'test_nan': test_nan,
        'label_inf': label_inf,
        'label_nan': label_nan
    }

def save_preprocessed_data(train_features, test_features, train_labels, all_features, save_path='../data/'):
    """
    保存预处理后的数据
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 保存张量数据
    torch.save(train_features, os.path.join(save_path, 'train_features.pt'))
    torch.save(test_features, os.path.join(save_path, 'test_features.pt'))
    torch.save(train_labels, os.path.join(save_path, 'train_labels.pt'))
    
    # 保存特征DataFrame
    all_features.to_csv(os.path.join(save_path, 'all_features_processed.csv'), index=False)
    
    print(f"预处理后的数据已保存到: {save_path}")

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    try:
        # 1. 加载原始数据
        print("正在加载数据...")
        train_data, test_data = load_kaggle_house_data()
        
        # 保存原始数值特征列表用于分析
        all_features_temp = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
        numeric_features_original = all_features_temp.dtypes[all_features_temp.dtypes != 'object'].index
        
        # 2. 执行数据预处理
        train_features, test_features, train_labels, all_features = preprocess_data(
            train_data, test_data, verbose=True
        )
        
        # 3. 分析预处理效果
        analyze_preprocessing_effects(all_features, numeric_features_original, verbose=True)
        
        # 4. 检查数据质量
        quality_report = check_data_quality(train_features, test_features, train_labels, verbose=True)
        
        # 5. 保存预处理后的数据
        save_preprocessed_data(train_features, test_features, train_labels, all_features)
        
        print("\n=== 数据预处理流程完成 ===")
        print("数据已准备好用于模型训练！")
        
        return {
            'train_features': train_features,
            'test_features': test_features,
            'train_labels': train_labels,
            'all_features': all_features,
            'quality_report': quality_report
        }
        
    except Exception as e:
        print(f"数据预处理过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    # 执行数据预处理
    result = main()
    
    if result:
        print("\n预处理结果:")
        print(f"- 训练特征: {result['train_features'].shape}")
        print(f"- 测试特征: {result['test_features'].shape}")
        print(f"- 训练标签: {result['train_labels'].shape}")
        print(f"- 处理后特征总数: {result['all_features'].shape[1]}")