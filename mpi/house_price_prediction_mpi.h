#ifndef HOUSE_PRICE_PREDICTION_MPI_H
#define HOUSE_PRICE_PREDICTION_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

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

// MPI信息结构
typedef struct {
    int rank;
    int size;
    MPI_Comm comm;
} MPIInfo;

// CSV文件处理函数
Dataset* load_csv(const char* filename, int has_label);
void free_dataset(Dataset* dataset);
int count_features(const char* filename);

// 数据预处理函数（MPI版本）
void normalize_features_mpi(Dataset* dataset, MPIInfo* mpi_info);
void handle_missing_values_mpi(Dataset* dataset, MPIInfo* mpi_info);
Dataset* distribute_dataset_mpi(Dataset* full_dataset, MPIInfo* mpi_info);
Dataset* gather_predictions_mpi(Dataset* local_dataset, MPIInfo* mpi_info);

// 线性回归模型函数（MPI版本）
LinearRegressionModel* create_linear_regression_model(int feature_count);
void train_linear_regression_mpi(LinearRegressionModel* model, Dataset* train_data, 
                               double learning_rate, int epochs, double weight_decay, MPIInfo* mpi_info);
double predict(LinearRegressionModel* model, double* features);
double calculate_rmse_mpi(LinearRegressionModel* model, Dataset* dataset, MPIInfo* mpi_info);
void free_linear_regression_model(LinearRegressionModel* model);

// K折交叉验证函数（MPI版本）
double k_fold_cross_validation_mpi(Dataset* dataset, int k, double learning_rate, 
                                 int epochs, double weight_decay, MPIInfo* mpi_info);

// 文件操作函数
int save_predictions(const char* filename, double* predictions, int* ids, int count);

// 工具函数
double* copy_features(double* features, int feature_count);
void shuffle_dataset(Dataset* dataset);
void broadcast_model_mpi(LinearRegressionModel* model, MPIInfo* mpi_info);

#endif // HOUSE_PRICE_PREDICTION_MPI_H