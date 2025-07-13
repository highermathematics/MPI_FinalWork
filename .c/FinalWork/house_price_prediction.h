#ifndef HOUSE_PRICE_PREDICTION_H
#define HOUSE_PRICE_PREDICTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 定义数据结构
typedef struct {
    int id;
    double* features;
    double label;
} HouseData;

typedef struct {
    HouseData* data;
    int count;
    int feature_count;
} Dataset;

typedef struct {
    double* weights;
    double bias;
    int feature_count;
} LinearRegressionModel;

// CSV文件处理函数
Dataset* load_csv(const char* filename, int has_label);
void free_dataset(Dataset* dataset);
int count_features(const char* filename);

// 数据预处理函数
void normalize_features(Dataset* dataset);
void handle_missing_values(Dataset* dataset);
Dataset* train_test_split(Dataset* dataset, double test_size);

// 线性回归模型函数
LinearRegressionModel* create_linear_regression_model(int feature_count);
void train_linear_regression(LinearRegressionModel* model, Dataset* train_data, double learning_rate, int epochs, double weight_decay);
double predict(LinearRegressionModel* model, double* features);
double calculate_rmse(LinearRegressionModel* model, Dataset* dataset);
void free_linear_regression_model(LinearRegressionModel* model);

// K折交叉验证函数
double k_fold_cross_validation(Dataset* dataset, int k, double learning_rate, int epochs, double weight_decay);

// 文件操作函数
int save_predictions(const char* filename, double* predictions, int* ids, int count);

// 工具函数
double* copy_features(double* features, int feature_count);
void shuffle_dataset(Dataset* dataset);

#endif // HOUSE_PRICE_PREDICTION_H