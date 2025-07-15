import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np

# 设置中文字体 - 只保留Windows系统确保存在的字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 验证字体是否可用的辅助函数
def get_available_font(font_list):
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_list:
        if font in available_fonts:
            return font
    return None

# 检查并设置可用字体
available_font = get_available_font(plt.rcParams["font.family"])
if available_font:
    plt.rcParams["font.family"] = [available_font]
    print(f"已设置可用字体: {available_font}")
else:
    print("警告: 未找到指定的中文字体，可能导致中文显示异常")
    # 列出所有可用字体供调试
    print("系统可用字体列表前20个: ", [f.name for f in fm.fontManager.ttflist[:20]])

# 移除函数外的冗余代码，确保所有绘图逻辑在函数内执行

def visualize_training_log(log_path="results/training_log.csv"):
    """
    可视化训练日志中的RMSE变化曲线
    :param log_path: 训练日志CSV文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(log_path):
        print(f"错误: 日志文件 {log_path} 不存在，请先运行训练程序生成日志")
        return

    # 读取日志数据
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"读取日志文件失败: {e}")
        return

    # 确保必要的列存在
    required_columns = ['epoch', 'rmse', 'model_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 日志文件缺少必要的列: {missing_columns}")
        return

    # 调试输出：显示所有模型类型
    print(f"日志中包含的模型类型: {df['model_type'].unique()}")

    # 创建图表
    plt.figure(figsize=(12, 7))

    # 模型名称映射（确保与日志中的model_type完全匹配）
    MODEL_NAME_MAPPING = {
        'linear': '线性回归模型',         # 对应蓝色直线
        'advanced': '深度神经网络模型',  # 对应红色点划线
    }

    # 定义不同模型的样式
    MODEL_STYLES = {
        '线性回归模型': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},  # 直线
        '深度神经网络模型': {'color': 'red', 'linestyle': '-.', 'linewidth': 2} # 点划线
    }

    # 绘制每条曲线
    for model_type in df['model_type'].unique():
        model_name = MODEL_NAME_MAPPING.get(model_type, model_type)
        model_data = df[df['model_type'] == model_type]
        style = MODEL_STYLES.get(model_name, {'color': 'black', 'linestyle': '-'})
        
        # 调试输出：显示当前绘制的模型
        print(f"绘制模型: {model_name}, 数据点数量: {len(model_data)}")
        
        plt.plot(model_data['epoch'], model_data['rmse'],
                 label=f'{model_name} (最小RMSE: {model_data["rmse"].min():.4f})',
                 **style)

    # 添加图例
    plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)

    # 添加标题和坐标轴标签
    plt.title('不同模型训练过程中RMSE变化曲线对比', fontsize=14)
    plt.xlabel('迭代次数 (Epoch)', fontsize=12)
    plt.ylabel('均方根误差 (RMSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图表
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"training_rmse_curve_{timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存至: {output_path}")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    visualize_training_log()
# 添加日志文件读取代码（确保在绘图前执行）
LOG_PATH = 'results/training_log.csv'
if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(f"错误: 日志文件 {LOG_PATH} 不存在")
log_df = pd.read_csv(LOG_PATH)  # 这行定义了log_df变量

# 检查必要的列是否存在
required_columns = ['epoch', 'rmse', 'model_type']
missing_columns = [col for col in required_columns if col not in log_df.columns]
if missing_columns:
    raise ValueError(f"错误: 日志文件缺少必要的列: {missing_columns}")

# 设置中文字体（适配 Windows 系统）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 绘制训练损失曲线
plt.figure(figsize=(12, 7))

# 关键修复：显式模型名称映射（根据您的实际model_type值调整）
MODEL_NAME_MAPPING = {
    'linear': '线性回归模型',         # 如果model_type是'linear'则显示为中文
    'decision_tree': '决策树模型',   # 替换为您数据中实际的model_type值
    'neural_network': '神经网络模型' # 确保key与training_log.csv中的model_type完全一致
}

# 定义不同模型的样式（确保视觉区分）
MODEL_STYLES = {
    '线性回归模型': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
    '神经网络模型': {'color': 'red', 'linestyle': '-.', 'linewidth': 2}
}

# 绘制每条曲线
for model_type in log_df['model_type'].unique():
    # 获取中文模型名称（如果没有映射则使用原始名称）
    model_name = MODEL_NAME_MAPPING.get(model_type, model_type)
    model_data = log_df[log_df['model_type'] == model_type]
    
    # 获取样式配置
    style = MODEL_STYLES.get(model_name, {'color': 'black', 'linestyle': '-'})
    
    # 绘制曲线并显式指定标签
    plt.plot(model_data['epoch'], model_data['rmse'],
             label=f'{model_name} (最小RMSE: {model_data["rmse"].min():.4f})',
             **style)

# 强制显示图例（确保不会被遗漏）
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)

plt.xlabel('迭代次数 (Epoch)', fontsize=12)
plt.ylabel('均方根误差 (RMSE)', fontsize=12)
plt.title('不同模型训练过程中RMSE变化曲线对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 1. 强制设置Windows系统 guaranteed 存在的中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 2. 验证字体是否加载成功（调试用，可保留）
available_fonts = [f.name for f in fm.fontManager.ttflist]
used_font = next((f for f in plt.rcParams["font.family"] if f in available_fonts), "默认字体")
print(f"使用字体: {used_font}")  # 终端会显示实际使用的字体