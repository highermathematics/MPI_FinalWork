#include "house_price_prediction_pthread.h"

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
    return count - 1;
}

// 从CSV文件加载数据集
Dataset* load_csv(const char* filename, int has_label) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    int feature_count = count_features(filename);
    if (feature_count <= 0) {
        fclose(file);
        return NULL;
    }

    char buffer[1024];
    int row_count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        row_count++;
    }
    row_count--;
    rewind(file);

    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->data = (HouseData*)malloc(sizeof(HouseData) * row_count);
    dataset->count = row_count;
    dataset->feature_count = has_label ? feature_count - 1 : feature_count;

    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        fprintf(stderr, "Error reading header from file: %s\n", filename);
        return NULL;
    }

    int row = 0;
    while (fgets(buffer, sizeof(buffer), file) && row < row_count) {
        char* token = strtok(buffer, ",");
        if (!token) continue;

        dataset->data[row].id = atoi(token);
        dataset->data[row].features = (double*)malloc(sizeof(double) * dataset->feature_count);

        for (int i = 0; i < dataset->feature_count; i++) {
            token = strtok(NULL, ",");
            if (!token) {
                dataset->data[row].features[i] = 0.0;
            } else if (strcmp(token, "") == 0) {
                dataset->data[row].features[i] = NAN;
            } else {
                dataset->data[row].features[i] = atof(token);
            }
        }

        if (has_label) {
            token = strtok(NULL, ",");
            dataset->data[row].label = token ? atof(token) : 0.0;
        } else {
            dataset->data[row].label = 0.0;
        }

        row++;
    }

    fclose(file);
    return dataset;
}

// 释放数据集内存
void free_dataset(Dataset* dataset) {
    if (!dataset) return;

    for (int i = 0; i < dataset->count; i++) {
        free(dataset->data[i].features);
    }
    free(dataset->data);
    free(dataset);
}