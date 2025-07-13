#include "house_price_prediction.h"

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#endif

// 保存预测结果到CSV文件
int save_predictions(const char* filename, double* predictions, int* ids, int count) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "无法创建输出文件: %s\n", filename);
        return 0;
    }

    // 写入表头
    fprintf(file, "Id,SalePrice\n");

    // 写入预测结果
    for (int i = 0; i < count; i++) {
        fprintf(file, "%d,%.6f\n", ids[i], predictions[i]);
    }

    fclose(file);
    return 1;
}

// 打印程序使用帮助
void print_help() {
    printf("Kaggle房价预测项目 - C语言实现\n");
    printf("使用方法: house_price_prediction [选项]\n");
    printf("选项:\n");
    printf("  --model <linear|advanced>   指定模型类型 (默认: linear)\n");
    printf("  --no-kfold                  禁用K折交叉验证\n");
    printf("  --skip-download             跳过数据下载 (假设数据已存在)\n");
    printf("  --quick                     快速模式 (跳过下载和K折验证)\n");
    printf("  --help                      显示此帮助信息\n");
}

// 主函数
int main(int argc, char* argv[]) {
    // 默认参数
    const char* model_type = "linear";
    int use_kfold = 1;
    int quick_mode = 0;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_type = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--no-kfold") == 0) {
            use_kfold = 0;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = 1;
            use_kfold = 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        } else {
            fprintf(stderr, "未知选项: %s\n", argv[i]);
            print_help();
            return 1;
        }
    }

    // 快速模式提示
    if (quick_mode) {
        printf("🚀 快速模式已启用\n");
    }

    // 步骤1: 加载训练数据
    printf("\n===步骤1:加载数据 ===\n");
    
    // 检查文件是否存在
    const char* train_path = "./data/kaggle_house_pred_train.csv";
    const char* test_path = "./data/kaggle_house_pred_test.csv";
    printf("尝试打开训练数据文件: %s\n", train_path);
    printf("尝试打开测试数据文件: %s\n", test_path);
    FILE* test_file = fopen(test_path, "r");
    if (!test_file) {
        perror("训练数据文件打开失败"); // 将显示如"No such file or directory"等具体原因
        return 1;
    }
    fclose(test_file);
    
    // 加载数据
    Dataset* train_dataset = load_csv(train_path, 1);
    Dataset* test_dataset = load_csv(test_path, 0);

    if (!train_dataset) {
        fprintf(stderr, "数据加载失败: 训练数据集解析错误\n");
        free_dataset(test_dataset);
        return 1;
    }
    if (!test_dataset) {
        fprintf(stderr, "数据加载失败: 测试数据集解析错误\n");
        free_dataset(train_dataset);
        return 1;
    }

    printf("✓ 训练数据加载完成: %d 样本, %d 特征\n", train_dataset->count, train_dataset->feature_count);
    printf("✓ 测试数据加载完成: %d 样本, %d 特征\n", test_dataset->count, test_dataset->feature_count);

    // 步骤2: 数据预处理
    printf("\n=== 步骤2: 数据预处理 ===\n");
    printf("处理缺失值...\n");
    handle_missing_values(train_dataset);
    handle_missing_values(test_dataset);

    printf("标准化特征...\n");
    normalize_features(train_dataset);
    normalize_features(test_dataset);
    printf("✓ 数据预处理完成\n");

    // 步骤3: 模型训练
    printf("\n=== 步骤3: 模型训练 ===\n");
    LinearRegressionModel* model = create_linear_regression_model(train_dataset->feature_count);
    if (!model) {
        fprintf(stderr, "模型创建失败\n");
        free_dataset(train_dataset);
        free_dataset(test_dataset);
        return 1;
    }

    // 训练参数
    double learning_rate = 0.1;
    int epochs = 1000;
    double weight_decay = 1e-4;
    int k_fold = 5;

    // K折交叉验证
    if (use_kfold) {
        printf("使用%d折交叉验证...\n", k_fold);
        double avg_rmse = k_fold_cross_validation(train_dataset, k_fold, learning_rate, epochs, weight_decay);
        printf("✓ K折交叉验证完成，平均RMSE: %.6f\n", avg_rmse);
    }

    // 训练最终模型
    printf("训练最终模型...\n");
    train_linear_regression(model, train_dataset, learning_rate, epochs, weight_decay);

    // 评估训练集性能
    double train_rmse = calculate_rmse(model, train_dataset);
    printf("✓ 模型训练完成，训练集RMSE: %.6f\n", train_rmse);

    // 步骤4: 生成预测
    printf("\n=== 步骤4: 生成预测 ===\n");
    int prediction_count = test_dataset->count;
    double* predictions = (double*)malloc(sizeof(double) * prediction_count);
    int* ids = (int*)malloc(sizeof(int) * prediction_count);

    for (int i = 0; i < prediction_count; i++) {
        predictions[i] = predict(model, test_dataset->data[i].features);
        ids[i] = test_dataset->data[i].id;
    }

    // 生成带时间戳的输出文件名
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    // 确保results目录存在
#ifdef _WIN32
    int dir_err = _mkdir("results");
#else
    int dir_err = mkdir("results", 0755);
#endif
    if (dir_err != 0) {
        if (errno != EEXIST) {
            perror("创建results目录失败");
            free(predictions);
            free(ids);
            free_linear_regression_model(model);
            free_dataset(train_dataset);
            free_dataset(test_dataset);
            return 1;
        }
    }

    char output_filename[100];
    snprintf(output_filename, sizeof(output_filename), "results/submission_%s_%s.csv", model_type, timestamp);

    // 保存预测结果
    if (save_predictions(output_filename, predictions, ids, prediction_count)) {
        printf("✓ 预测结果已保存到: %s\n", output_filename);
    } else {
        fprintf(stderr, "✗ 保存预测结果失败\n");
    }

    // 清理资源
    free(predictions);
    free(ids);
    free_linear_regression_model(model);
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    printf("\n🎉 项目执行完成！\n");
    return 0;
}
perror("训练数据文件打开失败"); // 将显示如"No such file or directory"等具体原因
