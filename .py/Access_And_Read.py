# 如果没有安装pandas，请取消下一行的注释
# !pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入之前定义的下载函数
from Dowload_Data import download, DATA_HUB, DATA_URL

# 添加Kaggle房价数据集到DATA_HUB
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

def load_kaggle_house_data():
    """加载Kaggle房价数据集"""
    print("正在加载Kaggle房价数据集...")
    
    # 使用pandas分别加载包含训练数据和测试数据的两个CSV文件
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    
    return train_data, test_data

def explore_data(train_data, test_data):
    """探索数据集的基本信息"""
    print("\n=== 数据集基本信息 ===")
    
    # 训练数据集包括1460个样本，每个样本80个特征和1个标签
    # 测试数据集包含1459个样本，每个样本80个特征
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    print("\n=== 查看前四个和最后两个特征，以及相应标签（房价） ===")
    # 显示前4行的特定列：前4个特征和最后3个特征（包括标签）
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    
    print("\n=== 训练数据列名 ===")
    print("前10个特征:", list(train_data.columns[:10]))
    print("最后5个特征:", list(train_data.columns[-5:]))
    
    print("\n=== 数据类型分布 ===")
    print(train_data.dtypes.value_counts())
    
    print("\n=== 缺失值统计 ===")
    missing_train = train_data.isnull().sum()
    missing_test = test_data.isnull().sum()
    print(f"训练集缺失值总数: {missing_train.sum()}")
    print(f"测试集缺失值总数: {missing_test.sum()}")
    
    # 显示缺失值最多的前10个特征
    print("\n训练集缺失值最多的特征:")
    print(missing_train[missing_train > 0].sort_values(ascending=False).head(10))
    
    return missing_train, missing_test

def preprocess_features(train_data, test_data):
    """预处理特征数据"""
    print("\n=== 开始特征预处理 ===")
    
    # 我们可以看到，在每个样本中，第一个特征是ID，
    # 这有助于模型识别每个训练样本。虽然这很方便，但它不携带任何用于预测的信息。
    # 因此，在将数据提供给模型之前，我们将其从数据集中删除。
    
    # 合并训练和测试数据的特征（除了ID和标签）
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    
    print(f"合并后的特征数据形状: {all_features.shape}")
    print(f"特征数量: {all_features.shape[1]}")
    
    # 获取数值特征和类别特征
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    categorical_features = all_features.dtypes[all_features.dtypes == 'object'].index
    
    print(f"\n数值特征数量: {len(numeric_features)}")
    print(f"类别特征数量: {len(categorical_features)}")
    
    print("\n前10个数值特征:", list(numeric_features[:10]))
    print("前10个类别特征:", list(categorical_features[:10]))
    
    return all_features, numeric_features, categorical_features

def analyze_target_variable(train_data):
    """分析目标变量（房价）"""
    print("\n=== 目标变量分析 ===")
    
    # 获取房价数据
    prices = train_data['SalePrice']
    
    print(f"房价统计信息:")
    print(f"平均价格: ${prices.mean():.2f}")
    print(f"中位数价格: ${prices.median():.2f}")
    print(f"最低价格: ${prices.min():.2f}")
    print(f"最高价格: ${prices.max():.2f}")
    print(f"标准差: ${prices.std():.2f}")
    
    # 绘制房价分布图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(prices, bins=50, alpha=0.7, color='skyblue')
    plt.title('房价分布')
    plt.xlabel('房价 ($)')
    plt.ylabel('频次')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log(prices), bins=50, alpha=0.7, color='lightgreen')
    plt.title('房价对数分布')
    plt.xlabel('log(房价)')
    plt.ylabel('频次')
    
    plt.tight_layout()
    plt.show()
    
    return prices

def main():
    """主函数：执行完整的数据加载和探索流程"""
    try:
        # 1. 加载数据
        train_data, test_data = load_kaggle_house_data()
        
        # 2. 探索数据
        missing_train, missing_test = explore_data(train_data, test_data)
        
        # 3. 预处理特征
        all_features, numeric_features, categorical_features = preprocess_features(train_data, test_data)
        
        # 4. 分析目标变量
        prices = analyze_target_variable(train_data)
        
        print("\n=== 数据加载和探索完成 ===")
        print("接下来可以进行数据预处理和模型训练。")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'all_features': all_features,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'prices': prices
        }
        
    except Exception as e:
        print(f"数据加载过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    # 执行数据加载和探索
    data_dict = main()
    
    if data_dict:
        print("\n数据已成功加载到以下变量中:")
        for key, value in data_dict.items():
            if hasattr(value, 'shape'):
                print(f"- {key}: {type(value).__name__} {value.shape}")
            else:
                print(f"- {key}: {type(value).__name__}")