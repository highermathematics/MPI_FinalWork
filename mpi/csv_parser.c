#include "house_price_prediction_mpi.h"

// 计算CSV文件中的特征数量
int count_features(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return -1;
    }

    char buffer[1024];
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        return -1;
    }

    int count = 1;
    for (int i = 0; buffer[i] != '\0'; i++) {
        if (buffer[i] == ',') {
            count++;
        }
    }

    fclose(file);
    return count - 1; // 减去ID列
}

// 从CSV文件加载数据集
Dataset* load_csv(const char* filename, int has_label) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    // 计算特征数量
    int feature_count = count_features(filename);
    if (feature_count <= 0) {
        fclose(file);
        return NULL;
    }

    // 计算数据行数
    char buffer[1024];
    int row_count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        row_count++;
    }
    row_count--; // 减去表头
    rewind(file);

    // 分配数据集内存
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->data = (HouseData*)malloc(sizeof(HouseData) * row_count);
    dataset->count = row_count;
    dataset->feature_count = has_label ? feature_count - 1 : feature_count;

    // 跳过表头
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        free_dataset(dataset);
        return NULL;
    }

    // 解析数据行
    int row = 0;
    while (fgets(buffer, sizeof(buffer), file) && row < row_count) {
        char* token = strtok(buffer, ",");
        if (!token) continue;

        // 解析ID
        dataset->data[row].id = atoi(token);
        dataset->data[row].features = (double*)malloc(sizeof(double) * dataset->feature_count);

        // 解析特征
        for (int i = 0; i < dataset->feature_count; i++) {
            token = strtok(NULL, ",");
            if (!token || strcmp(token, "") == 0) {
                dataset->data[row].features[i] = NAN; // 标记为缺失值
            } else {
                dataset->data[row].features[i] = atof(token);
            }
        }

        // 解析标签（如果有）
        if (has_label) {
            token = strtok(NULL, ",\n");
            if (token && strcmp(token, "") != 0) {
                dataset->data[row].label = atof(token);
            } else {
                dataset->data[row].label = 0.0;
            }
        } else {
            dataset->data[row].label = 0.0;
        }

        row++;
    }

    fclose(file);
    dataset->count = row; // 更新实际读取的行数
    return dataset;
}

// 释放数据集内存
void free_dataset(Dataset* dataset) {
    if (dataset) {
        if (dataset->data) {
            for (int i = 0; i < dataset->count; i++) {
                free(dataset->data[i].features);
            }
            free(dataset->data);
        }
        free(dataset);
    }
}