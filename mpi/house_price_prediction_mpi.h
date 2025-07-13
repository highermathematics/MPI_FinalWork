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

// MPI相关结构
typedef struct {
    int rank;
    int size;
    int local_start;
    int local_count;
} MPIInfo;

// CSV文件处理函数
Dataset* load_csv(const char* filename, int has_label);
void free_dataset(Dataset* dataset);
int count_features(const char* filename);

// 数据预处理函数
// 数据预处理函数
void normalize_features_mpi(Dataset* dataset, MPIInfo* mpi_info);
void handle_missing_values_mpi(Dataset* dataset, MPIInfo* mpi_info);
Dataset* distribute_data(Dataset* dataset, MPIInfo* mpi_info);

// 线性回归模型函数
LinearRegressionModel* create_linear_regression_model(int feature_count);
void train_linear_regression_mpi(LinearRegressionModel* model, Dataset* local_data, 
                                double learning_rate, int epochs, double weight_decay, MPIInfo* mpi_info);
double predict(LinearRegressionModel* model, double* features);
double calculate_rmse_mpi(LinearRegressionModel* model, Dataset* local_data, MPIInfo* mpi_info);
void free_linear_regression_model(LinearRegressionModel* model);

// MPI辅助函数
void init_mpi_info(MPIInfo* mpi_info, int total_samples);
void broadcast_model(LinearRegressionModel* model, int root, MPIInfo* mpi_info);
void gather_predictions(double* local_predictions, int local_count, 
                       double* global_predictions, int* counts, int* displs, MPIInfo* mpi_info);

// 文件操作函数
int save_predictions(const char* filename, double* predictions, int* ids, int count);

#endif