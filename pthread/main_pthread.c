#include "house_price_prediction_pthread.h"

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

    fprintf(file, "Id,SalePrice\n");
    for (int i = 0; i < count; i++) {
        fprintf(file, "%d,%.6f\n", ids[i], predictions[i]);
    }

    fclose(file);
    return 1;
}

void print_help() {
    printf("Kaggle房价预测项目 - pthread并行版本\n");
    printf("使用方法: house_price_prediction_pthread [选项]\n");
    printf("选项:\n");
    printf("  --model <linear|advanced>   指定模型类型 (默认: linear)\n");
    printf("  --threads <num>             指定线程数 (默认: 4)\n");
    printf("  --no-kfold                  禁用K折交叉验证\n");
    printf("  --quick                     快速模式 (跳过K折验证)\n");
    printf("  --help                      显示此帮助信息\n");
}

int main(int argc, char* argv[]) {
    // 默认参数
    const char* model_type = "linear";
    int use_kfold = 1;
    int quick_mode = 0;
    int num_threads = 4;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_type = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            if (num_threads <= 0) num_threads = 4;
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

    printf("🚀 使用 %d 个线程进行并行计算\n", num_threads);
    if (quick_mode) {
        printf("🚀 快速模式已启用\n");
    }

    // 步骤1: 加载数据
    printf("\n=== 步骤1: 加载数据 ===\n");
    const char* train_path = "./data/kaggle_house_pred_train.csv";
    const char* test_path = "./data/kaggle_house_pred_test.csv";
    
    Dataset* train_dataset = load_csv(train_path, 1);
    Dataset* test_dataset = load_csv(test_path, 0);

    if (!train_dataset || !test_dataset) {
        fprintf(stderr, "数据加载失败\n");
        free_dataset(train_dataset);
        free_dataset(test_dataset);
        return 1;
    }

    printf("✓ 训练数据加载完成: %d 样本, %d 特征\n", train_dataset->count, train_dataset->feature_count);
    printf("✓ 测试数据加载完成: %d 样本, %d 特征\n", test_dataset->count, test_dataset->feature_count);

    // 步骤2: 数据预处理（并行）
    printf("\n=== 步骤2: 数据预处理（并行） ===\n");
    printf("处理缺失值...\n");
    handle_missing_values_pthread(train_dataset, num_threads);
    handle_missing_values_pthread(test_dataset, num_threads);

    printf("标准化特征...\n");
    normalize_features_pthread(train_dataset, num_threads);
    normalize_features_pthread(test_dataset, num_threads);
    printf("✓ 数据预处理完成\n");

    // 步骤3: 模型训练（并行）
    printf("\n=== 步骤3: 模型训练（并行） ===\n");
    LinearRegressionModel* model = create_linear_regression_model(train_dataset->feature_count);
    if (!model) {
        fprintf(stderr, "模型创建失败\n");
        free_dataset(train_dataset);
        free_dataset(test_dataset);
        return 1;
    }

    double learning_rate = 0.1;
    int epochs = 1000;
    double weight_decay = 1e-4;
    int k_fold = 5;

    // K折交叉验证（并行）
    if (use_kfold) {
        printf("使用%d折交叉验证（并行）...\n", k_fold);
        double avg_rmse = k_fold_cross_validation_pthread(train_dataset, k_fold, learning_rate, epochs, weight_decay, num_threads);
        printf("✓ K折交叉验证完成，平均RMSE: %.6f\n", avg_rmse);
    }

    // 训练最终模型（并行）
    printf("训练最终模型（并行）...\n");
    train_linear_regression_pthread(model, train_dataset, learning_rate, epochs, weight_decay, num_threads);

    // 评估训练集性能（并行）
    double train_rmse = calculate_rmse_pthread(model, train_dataset, num_threads);
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

    // 生成输出文件
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

#ifdef _WIN32
    int dir_err = _mkdir("results");
#else
    int dir_err = mkdir("results", 0755);
#endif
    if (dir_err != 0 && errno != EEXIST) {
        perror("创建results目录失败");
    }

    char output_filename[100];
    snprintf(output_filename, sizeof(output_filename), "results/submission_%s_pthread_%s.csv", model_type, timestamp);

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

    printf("\n🎉 pthread并行版本执行完成！\n");
    return 0;
}