#include "house_price_prediction.h"

// 计算特征的均值和标准差
static void compute_mean_and_std(Dataset* dataset, double* mean, double* std) {
    int feature_count = dataset->feature_count;
    int sample_count = dataset->count;

    // 初始化均值数组
    for (int i = 0; i < feature_count; i++) {
        mean[i] = 0.0;
        int valid_count = 0;

        // 计算均值
        for (int j = 0; j < sample_count; j++) {
            double val = dataset->data[j].features[i];
            if (!isnan(val)) {
                mean[i] += val;
                valid_count++;
            }
        }

        if (valid_count > 0) {
            mean[i] /= valid_count;
        } else {
            mean[i] = 0.0; // 如果所有值都是NaN，设为0
        }

        // 计算标准差
        std[i] = 0.0;
        valid_count = 0;
        for (int j = 0; j < sample_count; j++) {
            double val = dataset->data[j].features[i];
            if (!isnan(val)) {
                std[i] += pow(val - mean[i], 2);
                valid_count++;
            }
        }

        if (valid_count > 1) {
            std[i] = sqrt(std[i] / (valid_count - 1)); // 使用样本标准差
        } else {
            std[i] = 1.0; // 如果标准差为0，设为1避免除零错误
        }
    }
}

// 处理缺失值（使用均值填充）
void handle_missing_values(Dataset* dataset) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    int sample_count = dataset->count;
    double* mean = (double*)malloc(sizeof(double) * feature_count);

    // 计算每个特征的均值
    for (int i = 0; i < feature_count; i++) {
        mean[i] = 0.0;
        int valid_count = 0;

        for (int j = 0; j < sample_count; j++) {
            double val = dataset->data[j].features[i];
            if (!isnan(val)) {
                mean[i] += val;
                valid_count++;
            }
        }

        if (valid_count > 0) {
            mean[i] /= valid_count;
        } else {
            mean[i] = 0.0; // 如果所有值都是NaN，设为0
        }

        // 用均值填充缺失值
        for (int j = 0; j < sample_count; j++) {
            if (isnan(dataset->data[j].features[i])) {
                dataset->data[j].features[i] = mean[i];
            }
        }
    }

    free(mean);
}

// 特征归一化（标准化）
void normalize_features(Dataset* dataset) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    double* mean = (double*)malloc(sizeof(double) * feature_count);
    double* std = (double*)malloc(sizeof(double) * feature_count);

    compute_mean_and_std(dataset, mean, std);

    // 标准化每个特征
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < dataset->count; j++) {
            if (std[i] > 1e-6) { // 避免除零
                dataset->data[j].features[i] = (dataset->data[j].features[i] - mean[i]) / std[i];
            } else {
                dataset->data[j].features[i] = 0.0;
            }
        }
    }

    free(mean);
    free(std);
}

// 复制特征数组
double* copy_features(double* features, int feature_count) {
    double* new_features = (double*)malloc(sizeof(double) * feature_count);
    memcpy(new_features, features, sizeof(double) * feature_count);
    return new_features;
}

// 打乱数据集
void shuffle_dataset(Dataset* dataset) {
    if (!dataset || dataset->count <= 1) return;

    srand(time(NULL));
    for (int i = dataset->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // 交换数据
        HouseData temp = dataset->data[i];
        dataset->data[i] = dataset->data[j];
        dataset->data[j] = temp;
    }
}

// 划分训练集和测试集
Dataset* train_test_split(Dataset* dataset, double test_size) {
    if (!dataset || test_size <= 0 || test_size >= 1) return NULL;

    shuffle_dataset(dataset);
    int test_count = (int)(dataset->count * test_size);
    int train_count = dataset->count - test_count;

    // 创建测试集
    Dataset* test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->count = test_count;
    test_dataset->feature_count = dataset->feature_count;
    test_dataset->data = (HouseData*)malloc(sizeof(HouseData) * test_count);

    // 复制测试集数据
    for (int i = 0; i < test_count; i++) {
        test_dataset->data[i].id = dataset->data[train_count + i].id;
        test_dataset->data[i].label = dataset->data[train_count + i].label;
        test_dataset->data[i].features = copy_features(
            dataset->data[train_count + i].features, dataset->feature_count
        );
    }

    // 调整训练集大小
    dataset->count = train_count;

    return test_dataset;
}