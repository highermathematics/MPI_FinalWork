#include "house_price_prediction_mpi.h"

// 创建线性回归模型
// 在create_linear_regression_model函数中
LinearRegressionModel* create_linear_regression_model(int feature_count) {
    LinearRegressionModel* model = (LinearRegressionModel*)malloc(sizeof(LinearRegressionModel));
    if (!model) return NULL;

    model->feature_count = feature_count;
    model->weights = (double*)calloc(feature_count, sizeof(double));
    model->bias = 0.0;

    if (!model->weights) {
        free(model);
        return NULL;
    }

    // 小随机初始化权重
    for (int i = 0; i < feature_count; i++) {
        model->weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.01;  // [-0.005, 0.005]
    }

    return model;
}

// MPI并行训练线性回归模型（极保守版本）
void train_linear_regression_mpi(LinearRegressionModel* model, Dataset* local_data, 
                                double learning_rate, int epochs, double weight_decay, MPIInfo* mpi_info) {
    if (!model || !local_data || local_data->count == 0) return;

    int feature_count = model->feature_count;
    int local_sample_count = local_data->count;
    
    // 使用最小的批处理大小
    const int BATCH_SIZE = 1;  // 每次只处理1个特征
    
    // 局部梯度
    double* local_dw = (double*)calloc(feature_count, sizeof(double));
    double local_db = 0.0;
    
    // 全局梯度（单个特征）
    double batch_local_dw = 0.0;
    double batch_global_dw = 0.0;
    double global_db = 0.0;
    
    // 总样本数
    int total_samples = 0;
    int mpi_result = MPI_Allreduce(&local_sample_count, &total_samples, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_result != MPI_SUCCESS) {
        fprintf(stderr, "进程 %d: 获取总样本数失败\n", mpi_info->rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (mpi_info->rank == 0) {
        printf("开始训练，总样本数: %d, 特征数: %d, 批处理大小: %d\n", total_samples, feature_count, BATCH_SIZE);
        printf("进入训练循环...\n");
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        if (mpi_info->rank == 0 && epoch % 10 == 0) {
            printf("开始 Epoch %d\n", epoch);
        }
        
        // 重置局部梯度
        memset(local_dw, 0, sizeof(double) * feature_count);
        local_db = 0.0;

        // 计算局部梯度
        for (int i = 0; i < local_sample_count; i++) {
            double y_pred = model->bias;
            for (int j = 0; j < feature_count; j++) {
                y_pred += model->weights[j] * local_data->data[i].features[j];
            }
            double error = y_pred - local_data->data[i].label;

            // 累积局部梯度
            local_db += error;
            for (int j = 0; j < feature_count; j++) {
                local_dw[j] += error * local_data->data[i].features[j];
            }
        }
        
        if (mpi_info->rank == 0 && epoch % 10 == 0) {
            printf("局部梯度计算完成\n");
        }

        // 逐个特征处理梯度的MPI_Allreduce
        for (int j = 0; j < feature_count; j++) {
            batch_local_dw = local_dw[j];
            
            // 对单个特征进行MPI_Allreduce
            mpi_result = MPI_Allreduce(&batch_local_dw, &batch_global_dw, 1, 
                                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if (mpi_result != MPI_SUCCESS) {
                fprintf(stderr, "进程 %d: MPI_Allreduce (特征 %d) 失败，错误码: %d\n", 
                       mpi_info->rank, j, mpi_result);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // 存储全局梯度
            local_dw[j] = batch_global_dw;
        }
        
        if (mpi_info->rank == 0 && epoch % 10 == 0) {
            printf("特征梯度聚合完成\n");
        }
        
        // 处理偏置梯度
        mpi_result = MPI_Allreduce(&local_db, &global_db, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (mpi_result != MPI_SUCCESS) {
            fprintf(stderr, "进程 %d: MPI_Allreduce (bias) 失败，错误码: %d\n", mpi_info->rank, mpi_result);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (mpi_info->rank == 0 && epoch % 10 == 0) {
            printf("偏置梯度聚合完成\n");
        }

        // 平均梯度并更新参数
        global_db /= total_samples;
        model->bias -= learning_rate * global_db;
        
        for (int j = 0; j < feature_count; j++) {
            local_dw[j] /= total_samples;
            model->weights[j] -= learning_rate * (local_dw[j] + weight_decay * model->weights[j]);
        }
        
        if (mpi_info->rank == 0 && epoch % 10 == 0) {
            printf("参数更新完成\n");
        }

        // 每100个epoch计算训练进度（所有进程都参与）
        if (epoch % 100 == 0) {
            if (mpi_info->rank == 0) {
                printf("计算 Epoch %d 的 RMSE...\n", epoch);
            }
            double rmse = calculate_rmse_mpi(model, local_data, mpi_info);
            if (mpi_info->rank == 0) {
                if (rmse > 0) {
                    printf("Epoch %d, RMSE: %.6f\n", epoch, rmse);
                } else {
                    printf("Epoch %d, RMSE计算失败，跳过\n", epoch);
                }
            }
        }
    }

    free(local_dw);
}

// 预测函数
double predict(LinearRegressionModel* model, double* features) {
    if (!model || !features) return 0.0;

    double result = model->bias;
    for (int i = 0; i < model->feature_count; i++) {
        result += model->weights[i] * features[i];
    }
    return result;
}

// MPI并行计算RMSE（修复版本）
// MPI并行计算RMSE（安全版本）
double calculate_rmse_mpi(LinearRegressionModel* model, Dataset* local_data, MPIInfo* mpi_info) {
    if (!model || !local_data) return 0.0;

    double local_sum_squared_error = 0.0;
    double local_count_double = (double)local_data->count;  // 转换为double类型

    // 计算局部平方误差和
    for (int i = 0; i < local_data->count; i++) {
        double prediction = predict(model, local_data->data[i].features);
        double error = prediction - local_data->data[i].label;
        local_sum_squared_error += error * error;
    }

    // 使用统一的double类型进行聚合
    double global_sum_squared_error = 0.0;
    double total_count_double = 0.0;
    
    // 聚合误差和
    int mpi_result = MPI_Allreduce(&local_sum_squared_error, &global_sum_squared_error, 
                                  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_result != MPI_SUCCESS) {
        if (mpi_info->rank == 0) {
            fprintf(stderr, "RMSE计算中误差和聚合失败，错误码: %d\n", mpi_result);
        }
        return 0.0;
    }
    
    // 聚合样本数（使用double类型）
    mpi_result = MPI_Allreduce(&local_count_double, &total_count_double, 
                              1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (mpi_result != MPI_SUCCESS) {
        if (mpi_info->rank == 0) {
            fprintf(stderr, "RMSE计算中样本数聚合失败，错误码: %d\n", mpi_result);
        }
        return 0.0;
    }

    if (total_count_double <= 0.0) return 0.0;
    return sqrt(global_sum_squared_error / total_count_double);
}

// 释放模型内存
void free_linear_regression_model(LinearRegressionModel* model) {
    if (model) {
        free(model->weights);
        free(model);
    }
}

// 初始化MPI信息
void init_mpi_info(MPIInfo* mpi_info, int total_samples) {
    int samples_per_process = total_samples / mpi_info->size;
    int remainder = total_samples % mpi_info->size;
    
    mpi_info->local_start = mpi_info->rank * samples_per_process + (mpi_info->rank < remainder ? mpi_info->rank : remainder);
    mpi_info->local_count = samples_per_process + (mpi_info->rank < remainder ? 1 : 0);
}

// 广播模型参数
void broadcast_model(LinearRegressionModel* model, int root, MPIInfo* mpi_info) {
    (void)mpi_info;  // 消除未使用参数警告
    MPI_Bcast(model->weights, model->feature_count, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(&model->bias, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

// 分发数据到各进程
Dataset* distribute_data(Dataset* dataset, MPIInfo* mpi_info) {
    Dataset* local_dataset = (Dataset*)malloc(sizeof(Dataset));
    local_dataset->count = mpi_info->local_count;
    local_dataset->feature_count = dataset ? dataset->feature_count : 0;
    
    // 广播特征数量
    MPI_Bcast(&local_dataset->feature_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (local_dataset->count > 0) {
        local_dataset->data = (HouseData*)malloc(sizeof(HouseData) * local_dataset->count);
        
        // 为每个样本分配特征数组
        for (int i = 0; i < local_dataset->count; i++) {
            local_dataset->data[i].features = (double*)malloc(sizeof(double) * local_dataset->feature_count);
        }
        
        if (mpi_info->rank == 0) {
            // 主进程分发数据
            for (int proc = 0; proc < mpi_info->size; proc++) {
                MPIInfo temp_info = *mpi_info;
                temp_info.rank = proc;
                init_mpi_info(&temp_info, dataset->count);
                
                if (proc == 0) {
                    // 复制数据到本地
                    for (int i = 0; i < temp_info.local_count; i++) {
                        local_dataset->data[i].id = dataset->data[temp_info.local_start + i].id;
                        local_dataset->data[i].label = dataset->data[temp_info.local_start + i].label;
                        memcpy(local_dataset->data[i].features, 
                               dataset->data[temp_info.local_start + i].features,
                               sizeof(double) * local_dataset->feature_count);
                    }
                } else {
                    // 发送数据到其他进程
                    for (int i = 0; i < temp_info.local_count; i++) {
                        int global_idx = temp_info.local_start + i;
                        MPI_Send(&dataset->data[global_idx].id, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
                        MPI_Send(&dataset->data[global_idx].label, 1, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                        MPI_Send(dataset->data[global_idx].features, local_dataset->feature_count, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            // 其他进程接收数据
            for (int i = 0; i < local_dataset->count; i++) {
                MPI_Recv(&local_dataset->data[i].id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&local_dataset->data[i].label, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_dataset->data[i].features, local_dataset->feature_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        local_dataset->data = NULL;
    }
    
    return local_dataset;
}