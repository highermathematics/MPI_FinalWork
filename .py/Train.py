# 在文件开头添加d2l相关的导入和函数
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time
import warnings
warnings.filterwarnings('ignore')

# 添加d2l相关函数
class d2l:
    @staticmethod
    def concat(tensors, dim=0):
        """连接张量"""
        return torch.cat(tensors, dim=dim)
    
    @staticmethod
    def plot(x, y, xlabel=None, ylabel=None, legend=None, xlim=None, 
             yscale='linear', figsize=(6, 4)):
        """绘制图表"""
        plt.figure(figsize=figsize)
        if isinstance(y[0], list):
            for i, yi in enumerate(y):
                plt.plot(x, yi, label=legend[i] if legend else f'Series {i}')
        else:
            plt.plot(x, y)
        
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if legend:
            plt.legend()
        if xlim:
            plt.xlim(xlim)
        if yscale == 'log':
            plt.yscale('log')
        
        plt.grid(True, alpha=0.3)
        plt.show()

# 导入预处理模块
from PreData import preprocess_data
from Access_And_Read import load_kaggle_house_data

# 定义损失函数
loss = nn.MSELoss()

def get_net(in_features):
    """
    创建线性模型网络
    
    首先，我们训练一个带有损失平方的线性模型。
    显然线性模型很难让我们在竞赛中获胜，但线性模型提供了一种健全性检查，
    以查看数据中是否存在有意义的信息。
    如果我们在这里不能做得比随机猜测更好，那么我们很可能存在数据处理错误。
    如果一切顺利，线性模型将作为基线（baseline）模型，
    让我们直观地知道最好的模型有超出简单的模型多少。
    
    参数:
    in_features: 输入特征数量
    
    返回:
    net: 线性模型
    """
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def get_advanced_net(in_features, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
    """
    创建更复杂的深度神经网络
    
    参数:
    in_features: 输入特征数量
    hidden_sizes: 隐藏层大小列表
    dropout_rate: Dropout比率
    
    返回:
    net: 深度神经网络
    """
    layers = []
    
    # 输入层到第一个隐藏层
    layers.append(nn.Linear(in_features, hidden_sizes[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    
    # 隐藏层
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    
    # 输出层
    layers.append(nn.Linear(hidden_sizes[-1], 1))
    
    net = nn.Sequential(*layers)
    return net

def log_rmse(net, features, labels):
    """
    计算对数均方根误差
    
    房价就像股票价格一样，我们关心的是相对数量，而不是绝对数量。
    因此，我们更关心相对误差 (y - ŷ)/y，而不是绝对误差 y - ŷ。
    
    解决这个问题的一种方法是用价格预测的对数来衡量差异。
    事实上，这也是比赛中官方用来评价提交质量的误差指标。
    
    参数:
    net: 训练好的模型
    features: 特征数据
    labels: 标签数据
    
    返回:
    log_rmse: 对数均方根误差
    """
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """
    训练模型 - 使用Adam优化器
    
    与前面的部分不同，我们的训练函数将借助Adam优化器
    （我们将在后面章节更详细地描述它）。
    Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
    
    参数:
    net: 神经网络模型
    train_features: 训练特征
    train_labels: 训练标签
    test_features: 测试特征（验证集）
    test_labels: 测试标签（验证集），可以为None
    num_epochs: 训练轮数
    learning_rate: 学习率
    weight_decay: 权重衰减
    batch_size: 批次大小
    
    返回:
    train_ls: 训练损失历史
    test_ls: 验证损失历史
    """
    train_ls, test_ls = [], []
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        
        # 计算并记录对数RMSE
        net.eval()
        with torch.no_grad():
            train_ls.append(log_rmse(net, train_features, train_labels))
            # 只有当test_labels不为None时才计算测试损失
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        
        # 打印训练进度
        if (epoch + 1) % 20 == 0:
            if test_labels is not None:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train log RMSE: {train_ls[-1]:.6f}, Test log RMSE: {test_ls[-1]:.6f}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train log RMSE: {train_ls[-1]:.6f}')
    
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    """
    获取K折交叉验证的第i折数据
    
    参数:
    k: 折数
    i: 当前折的索引
    X: 特征数据
    y: 标签数据
    
    返回:
    X_train, y_train: 训练数据
    X_valid, y_valid: 验证数据
    """
    assert k > 1
    fold_size = X.shape[0] // k
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, 
           batch_size):
    """
    K折交叉验证 - 按照原始代码风格实现
    
    当我们在K折交叉验证中训练K次后，返回训练和验证误差的平均值。
    
    参数:
    k: 折数
    X_train: 训练特征
    y_train: 训练标签
    num_epochs: 训练轮数
    learning_rate: 学习率
    weight_decay: 权重衰减
    batch_size: 批次大小
    
    返回:
    train_l_sum / k: 平均训练log rmse
    valid_l_sum / k: 平均验证log rmse
    """
    train_l_sum, valid_l_sum = 0, 0
    
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    
    return train_l_sum / k, valid_l_sum / k

def k_fold_cross_validation(k, X_train, y_train, num_epochs, learning_rate, 
                          weight_decay, batch_size, model_type='linear'):
    """
    K折交叉验证
    
    参数:
    k: 折数
    X_train: 训练特征
    y_train: 训练标签
    num_epochs: 训练轮数
    learning_rate: 学习率
    weight_decay: 权重衰减
    batch_size: 批次大小
    model_type: 模型类型 ('linear' 或 'advanced')
    
    返回:
    train_l_sum: 平均训练损失
    valid_l_sum: 平均验证损失
    """
    train_l_sum, valid_l_sum = 0, 0
    
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        X_train_fold, y_train_fold, X_valid_fold, y_valid_fold = data
        
        # 创建模型
        if model_type == 'linear':
            net = get_net(X_train.shape[1])
        else:
            net = get_advanced_net(X_train.shape[1])
        
        # 训练模型
        train_ls, valid_ls = train(
            net, X_train_fold, y_train_fold, X_valid_fold, y_valid_fold,
            num_epochs, learning_rate, weight_decay, batch_size
        )
        
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        
        print(f'Fold {i + 1}, Train log RMSE: {float(train_ls[-1]):.6f}, '
              f'Valid log RMSE: {float(valid_ls[-1]):.6f}')
    
    return train_l_sum / k, valid_l_sum / k

def train_and_predict(train_features, test_features, train_labels, test_data,
                     num_epochs, lr, weight_decay, batch_size, model_type='linear'):
    """
    训练模型并生成预测结果
    
    参数:
    train_features: 训练特征
    test_features: 测试特征
    train_labels: 训练标签
    test_data: 原始测试数据（用于获取ID）
    num_epochs: 训练轮数
    lr: 学习率
    weight_decay: 权重衰减
    batch_size: 批次大小
    model_type: 模型类型
    
    返回:
    preds: 预测结果
    net: 训练好的模型
    """
    # 创建模型
    if model_type == 'linear':
        net = get_net(train_features.shape[1])
    else:
        net = get_advanced_net(train_features.shape[1])
    
    # 训练模型 - 修复：测试标签应该为None，因为我们没有测试集的真实标签
    train_ls, _ = train(
        net, train_features, train_labels, test_features, None,
        num_epochs, lr, weight_decay, batch_size
    )
    
    # 生成预测
    net.eval()
    with torch.no_grad():
        preds = net(test_features).detach().numpy()
    
    return preds, net

def plot_training_history(train_ls, test_ls):
    """
    绘制训练历史
    
    参数:
    train_ls: 训练损失历史
    test_ls: 验证损失历史
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_ls) + 1), train_ls, label='训练log RMSE', color='blue')
    if test_ls:
        plt.plot(range(1, len(test_ls) + 1), test_ls, label='验证log RMSE', color='red')
    plt.xlabel('轮数')
    plt.ylabel('log RMSE')
    plt.title('训练和验证log RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_predictions(test_data, preds, filename='submission.csv'):
    """
    保存预测结果到CSV文件
    
    参数:
    test_data: 原始测试数据
    preds: 预测结果
    filename: 输出文件名
    """
    import pandas as pd
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_data.Id,
        'SalePrice': preds.reshape(-1)
    })
    
    submission.to_csv(filename, index=False)
    print(f"预测结果已保存到: {filename}")

def main():
    """
    主训练流程
    """
    print("=== Kaggle房价预测模型训练 ===")
    
    # 1. 加载和预处理数据
    print("\n1. 加载数据...")
    train_data, test_data = load_kaggle_house_data()
    
    print("\n2. 预处理数据...")
    train_features, test_features, train_labels, all_features = preprocess_data(
        train_data, test_data, verbose=True
    )
    
    # 2. 设置超参数
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    
    print(f"\n3. 开始训练基线线性模型...")
    print(f"超参数设置: k={k}, epochs={num_epochs}, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}")
    
    # 3. K折交叉验证 - 线性模型
    train_l, valid_l = k_fold_cross_validation(
        k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, 'linear'
    )
    
    print(f'\n线性模型 {k}-fold 验证: 平均训练log RMSE: {float(train_l):.6f}, '
          f'平均验证log RMSE: {float(valid_l):.6f}')
    
    # 4. 训练最终模型并生成预测
    print("\n4. 训练最终线性模型...")
    preds_linear, net_linear = train_and_predict(
        train_features, test_features, train_labels, test_data,
        num_epochs, lr, weight_decay, batch_size, 'linear'
    )
    
    # 5. 保存线性模型预测结果
    save_predictions(test_data, preds_linear, 'linear_submission.csv')
    
    # 6. 训练更复杂的模型
    print("\n5. 训练深度神经网络模型...")
    lr_advanced, weight_decay_advanced = 0.001, 0.01
    
    train_l_adv, valid_l_adv = k_fold_cross_validation(
        k, train_features, train_labels, num_epochs, lr_advanced, 
        weight_decay_advanced, batch_size, 'advanced'
    )
    
    print(f'\n深度模型 {k}-fold 验证: 平均训练log RMSE: {float(train_l_adv):.6f}, '
          f'平均验证log RMSE: {float(valid_l_adv):.6f}')
    
    # 7. 训练最终深度模型并生成预测
    preds_advanced, net_advanced = train_and_predict(
        train_features, test_features, train_labels, test_data,
        num_epochs, lr_advanced, weight_decay_advanced, batch_size, 'advanced'
    )
    
    # 8. 保存深度模型预测结果
    save_predictions(test_data, preds_advanced, 'advanced_submission.csv')
    
    print("\n=== 训练完成 ===")
    print(f"线性模型验证log RMSE: {float(valid_l):.6f}")
    print(f"深度模型验证log RMSE: {float(valid_l_adv):.6f}")
    print(f"性能提升: {((float(valid_l) - float(valid_l_adv)) / float(valid_l) * 100):.2f}%")
    
    return {
        'linear_model': net_linear,
        'advanced_model': net_advanced,
        'linear_rmse': float(valid_l),
        'advanced_rmse': float(valid_l_adv),
        'linear_preds': preds_linear,
        'advanced_preds': preds_advanced
    }

if __name__ == "__main__":
    # 执行训练
    results = main()
    
    if results:
        print("\n训练结果已保存，可以提交到Kaggle进行评估。")