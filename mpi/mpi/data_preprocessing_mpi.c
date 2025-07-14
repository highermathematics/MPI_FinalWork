#include "house_price_prediction_mpi.h"

// 分发数据集到各个MPI进程
Dataset* distribute_dataset_mpi(Dataset* full_dataset, MPIInfo* mpi_info) {
    int feature_count = 0;
    int total_count = 0;
    
    // 广播数据集基本信息
    if (mpi_info->rank == 0) {
        if (full_dataset) {
            feature_count = full_dataset->feature_count;
            total_count = full_dataset->count;
        }
    }
    
    MPI_Bcast(&feature_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (feature_count <= 0 || total_count <= 0) {
        return NULL;
    }
    
    // 计算每个进程的数据量
    int local_count = total_count / mpi_info->size;
    int remainder = total_count % mpi_info->size;
    
    // 最后几个进程处理余数
    if (mpi_info->rank < remainder) {
        local_count++;
    }
    
    // 创建本地数据集
    Dataset* local_dataset = (Dataset*)malloc(sizeof(Dataset));
    local_dataset->count = local_count;
    local_dataset->feature_count = feature_count;
    local_dataset->data = (HouseData*)malloc(sizeof(HouseData) * local_count);
    
    // 为本地数据分配特征数组
    for (int i = 0; i < local_count; i++) {
        local_dataset->data[i].features = (double*)malloc(sizeof(double) * feature_count);
    }
    
    // 准备发送缓冲区（只在主进程中）
    double* send_features = NULL;
    double* send_labels = NULL;
    int* send_ids = NULL;
    int* sendcounts_features = NULL;
    int* sendcounts_data = NULL;
    int* displs_features = NULL;
    int* displs_data = NULL;
    
    if (mpi_info->rank == 0) {
        send_features = (double*)malloc(sizeof(double) * total_count * feature_count);
        send_labels = (double*)malloc(sizeof(double) * total_count);
        send_ids = (int*)malloc(sizeof(int) * total_count);
        sendcounts_features = (int*)malloc(sizeof(int) * mpi_info->size);
        sendcounts_data = (int*)malloc(sizeof(int) * mpi_info->size);
        displs_features = (int*)malloc(sizeof(int) * mpi_info->size);
        displs_data = (int*)malloc(sizeof(int) * mpi_info->size);
        
        // 准备发送数据
        for (int i = 0; i < total_count; i++) {
            for (int j = 0; j < feature_count; j++) {
                send_features[i * feature_count + j] = full_dataset->data[i].features[j];
            }
            send_labels[i] = full_dataset->data[i].label;
            send_ids[i] = full_dataset->data[i].id;
        }
        
        // 计算发送计数和位移
        int offset_features = 0;
        int offset_data = 0;
        for (int i = 0; i < mpi_info->size; i++) {
            int count = total_count / mpi_info->size;
            if (i < remainder) count++;
            
            sendcounts_features[i] = count * feature_count;
            sendcounts_data[i] = count;
            displs_features[i] = offset_features;
            displs_data[i] = offset_data;
            
            offset_features += sendcounts_features[i];
            offset_data += sendcounts_data[i];
        }
    }
    
    // 准备接收缓冲区
    double* recv_features = (double*)malloc(sizeof(double) * local_count * feature_count);
    double* recv_labels = (double*)malloc(sizeof(double) * local_count);
    int* recv_ids = (int*)malloc(sizeof(int) * local_count);
    
    // 分发特征数据
    MPI_Scatterv(send_features, sendcounts_features, displs_features, MPI_DOUBLE, 
                 recv_features, local_count * feature_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 分发标签和ID（使用MPI_Scatterv而不是MPI_Scatter）
    MPI_Scatterv(send_labels, sendcounts_data, displs_data, MPI_DOUBLE, 
                 recv_labels, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(send_ids, sendcounts_data, displs_data, MPI_INT, 
                 recv_ids, local_count, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 复制到本地数据结构
    for (int i = 0; i < local_count; i++) {
        for (int j = 0; j < feature_count; j++) {
            local_dataset->data[i].features[j] = recv_features[i * feature_count + j];
        }
        local_dataset->data[i].label = recv_labels[i];
        local_dataset->data[i].id = recv_ids[i];
    }
    
    // 清理临时缓冲区
    if (mpi_info->rank == 0) {
        free(send_features);
        free(send_labels);
        free(send_ids);
        free(sendcounts_features);
        free(sendcounts_data);
        free(displs_features);
        free(displs_data);
    }
    
    free(recv_features);
    free(recv_labels);
    free(recv_ids);
    
    return local_dataset;
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

// MPI并行标准化特征
void normalize_features_mpi(Dataset* dataset, MPIInfo* mpi_info) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    int local_count = dataset->count;
    
    double* local_sum = (double*)calloc(feature_count, sizeof(double));
    double* local_sum_sq = (double*)calloc(feature_count, sizeof(double));
    double* global_mean = (double*)calloc(feature_count, sizeof(double));
    double* global_var = (double*)calloc(feature_count, sizeof(double));
    
    int total_count = 0;
    
    // 计算局部统计
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < local_count; j++) {
            double val = dataset->data[j].features[i];
            if (!isnan(val)) {
                local_sum[i] += val;
                local_sum_sq[i] += val * val;
            }
        }
    }
    
    // 聚合全局统计
    MPI_Allreduce(local_sum, global_mean, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_sum_sq, global_var, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 计算全局均值和方差
    for (int i = 0; i < feature_count; i++) {
        global_mean[i] /= total_count;
        global_var[i] = (global_var[i] / total_count) - (global_mean[i] * global_mean[i]);
        
        // 标准化局部数据
        double std_dev = sqrt(global_var[i]);
        if (std_dev > 1e-8) {
            for (int j = 0; j < local_count; j++) {
                if (!isnan(dataset->data[j].features[i])) {
                    dataset->data[j].features[i] = (dataset->data[j].features[i] - global_mean[i]) / std_dev;
                }
            }
        } else {
            for (int j = 0; j < local_count; j++) {
                dataset->data[j].features[i] = 0.0;
            }
        }
    }
    
    free(local_sum);
    free(local_sum_sq);
    free(global_mean);
    free(global_var);
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
        HouseData temp = dataset->data[i];
        dataset->data[i] = dataset->data[j];
        dataset->data[j] = temp;
    }
}