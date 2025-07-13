#include "house_price_prediction_mpi.h"

// MPI并行标准化特征
void normalize_features_mpi(Dataset* dataset, MPIInfo* mpi_info) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    double* local_mean = (double*)calloc(feature_count, sizeof(double));
    double* local_var = (double*)calloc(feature_count, sizeof(double));
    double* global_mean = (double*)calloc(feature_count, sizeof(double));
    double* global_var = (double*)calloc(feature_count, sizeof(double));
    
    int local_count = dataset->count;
    int total_count = 0;
    
    // 计算局部均值
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < local_count; j++) {
            if (!isnan(dataset->data[j].features[i])) {
                local_mean[i] += dataset->data[j].features[i];
            }
        }
    }
    
    // 聚合全局均值
    MPI_Allreduce(local_mean, global_mean, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    for (int i = 0; i < feature_count; i++) {
        global_mean[i] /= total_count;
    }
    
    // 计算局部方差
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < local_count; j++) {
            if (!isnan(dataset->data[j].features[i])) {
                double diff = dataset->data[j].features[i] - global_mean[i];
                local_var[i] += diff * diff;
            }
        }
    }
    
    // 聚合全局方差
    MPI_Allreduce(local_var, global_var, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // 标准化局部数据
    for (int i = 0; i < feature_count; i++) {
        double std_dev = sqrt(global_var[i] / (total_count - 1));
        if (std_dev > 1e-8) { // 避免除零
            for (int j = 0; j < local_count; j++) {
                if (!isnan(dataset->data[j].features[i])) {
                    dataset->data[j].features[i] = (dataset->data[j].features[i] - global_mean[i]) / std_dev;
                }
            }
        }
    }
    
    free(local_mean);
    free(local_var);
    free(global_mean);
    free(global_var);
}

// MPI并行处理缺失值
void handle_missing_values_mpi(Dataset* dataset, MPIInfo* mpi_info) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    int local_count = dataset->count;
    
    double* local_sum = (double*)calloc(feature_count, sizeof(double));
    int* local_valid_count = (int*)calloc(feature_count, sizeof(int));
    double* global_mean = (double*)calloc(feature_count, sizeof(double));
    int* global_valid_count = (int*)calloc(feature_count, sizeof(int));
    
    // 计算局部统计
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < local_count; j++) {
            double val = dataset->data[j].features[i];
            if (!isnan(val)) {
                local_sum[i] += val;
                local_valid_count[i]++;
            }
        }
    }
    
    // 聚合全局统计
    MPI_Allreduce(local_sum, global_mean, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_valid_count, global_valid_count, feature_count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 计算全局均值
    for (int i = 0; i < feature_count; i++) {
        if (global_valid_count[i] > 0) {
            global_mean[i] /= global_valid_count[i];
        } else {
            global_mean[i] = 0.0;
        }
    }
    
    // 用全局均值填充局部缺失值
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < local_count; j++) {
            if (isnan(dataset->data[j].features[i])) {
                dataset->data[j].features[i] = global_mean[i];
            }
        }
    }
    
    free(local_sum);
    free(local_valid_count);
    free(global_mean);
    free(global_valid_count);
}