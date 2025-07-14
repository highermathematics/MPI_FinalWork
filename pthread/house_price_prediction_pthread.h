#ifndef HOUSE_PRICE_PREDICTION_PTHREAD_H
#define HOUSE_PRICE_PREDICTION_PTHREAD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

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

// 线程参数结构
typedef struct {
    int thread_id;
    int num_threads;
    Dataset* dataset;
    LinearRegressionModel* model;
    double learning_rate;
    double weight_decay;
    double* gradients;
    double* bias_gradient;
    pthread_mutex_t* mutex;
    int start_idx;
    int end_idx;
    double* local_error;
} ThreadData;

typedef struct {
    Dataset* dataset;
    double* mean;
    double* std;
    int start_idx;
    int end_idx;
    pthread_mutex_t* mutex;
} PreprocessThreadData;

// CSV文件处理函数
Dataset* load_csv(const char* filename, int has_label);
void free_dataset(Dataset* dataset);
int count_features(const char* filename);

// 数据预处理函数（并行版本）
void normalize_features_pthread(Dataset* dataset, int num_threads);
void handle_missing_values_pthread(Dataset* dataset, int num_threads);
Dataset* train_test_split(Dataset* dataset, double test_size);

// 线性回归模型函数（并行版本）
LinearRegressionModel* create_linear_regression_model(int feature_count);
void train_linear_regression_pthread(LinearRegressionModel* model, Dataset* train_data, 
                                   double learning_rate, int epochs, double weight_decay, int num_threads);
double predict(LinearRegressionModel* model, double* features);
double calculate_rmse_pthread(LinearRegressionModel* model, Dataset* dataset, int num_threads);
void free_linear_regression_model(LinearRegressionModel* model);

// K折交叉验证函数（并行版本）
double k_fold_cross_validation_pthread(Dataset* dataset, int k, double learning_rate, 
                                      int epochs, double weight_decay, int num_threads);

// 文件操作函数
int save_predictions(const char* filename, double* predictions, int* ids, int count);

// 工具函数
double* copy_features(double* features, int feature_count);
void shuffle_dataset(Dataset* dataset);

// 线程函数声明
void* training_thread(void* arg);
void* preprocessing_thread(void* arg);
void* missing_values_thread(void* arg);  // 新增
void* rmse_calculation_thread(void* arg);

#endif // HOUSE_PRICE_PREDICTION_PTHREAD_H