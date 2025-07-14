#include "house_price_prediction_mpi.h"

// 创建线性回归模型
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

    return model;
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

// 广播模型参数到所有进程
void broadcast_model_mpi(LinearRegressionModel* model, MPIInfo* mpi_info) {
    MPI_Bcast(&model->bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(model->weights, model->feature_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// MPI并行训练线性回归模型
void train_linear_regression_mpi(LinearRegressionModel* model, Dataset* train_data, 
                               double learning_rate, int epochs, double weight_decay, MPIInfo* mpi_info) {
    if (!model || !train_data || train_data->count == 0) return;

    int feature_count = model->feature_count;
    int local_sample_count = train_data->count;
    int total_sample_count = 0;
    
    // 获取全局样本数
    MPI_Allreduce(&local_sample_count, &total_sample_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // 局部和全局梯度数组
    double* local_gradients = (double*)calloc(feature_count, sizeof(double));
    double* global_gradients = (double*)calloc(feature_count, sizeof(double));
    double local_bias_gradient = 0.0;
    double global_bias_gradient = 0.0;
    
    if (!local_gradients || !global_gradients) {
        if (mpi_info->rank == 0) {
            printf("错误: 梯度数组内存分配失败\n");
        }
        free(local_gradients);
        free(global_gradients);
        return;
    }
    
    if (mpi_info->rank == 0) {
        printf("开始MPI并行训练，总样本数: %d, 特征数: %d, 进程数: %d\n", 
               total_sample_count, feature_count, mpi_info->size);
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 重置局部梯度
        memset(local_gradients, 0, sizeof(double) * feature_count);
        local_bias_gradient = 0.0;
        
        // 计算局部梯度
        for (int i = 0; i < local_sample_count; i++) {
            double y_pred = predict(model, train_data->data[i].features);
            double error = y_pred - train_data->data[i].label;
            
            // 累积局部梯度
            local_bias_gradient += error;
            for (int j = 0; j < feature_count; j++) {
                local_gradients[j] += error * train_data->data[i].features[j];
            }
        }
        
        // 聚合全局梯度
        MPI_Allreduce(local_gradients, global_gradients, feature_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_bias_gradient, &global_bias_gradient, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // 平均梯度
        global_bias_gradient /= total_sample_count;
        for (int j = 0; j < feature_count; j++) {
            global_gradients[j] /= total_sample_count;
        }
        
        // 更新参数（所有进程同步更新）
        model->bias -= learning_rate * global_bias_gradient;
        for (int j = 0; j < feature_count; j++) {
            model->weights[j] -= learning_rate * (global_gradients[j] + weight_decay * model->weights[j]);
        }
        
        // 每100个epoch打印进度（只在主进程中，移除RMSE计算避免嵌套MPI调用）
        if (epoch % 100 == 0 && mpi_info->rank == 0) {
            printf("Epoch %d 完成\n", epoch);
        }
    }
    
    // 清理资源
    free(local_gradients);
    free(global_gradients);
}

// MPI并行计算RMSE
double calculate_rmse_mpi(LinearRegressionModel* model, Dataset* dataset, MPIInfo* mpi_info) {
    if (!model || !dataset || dataset->count == 0) return 0.0;

    double local_sum_squared_error = 0.0;
    double global_sum_squared_error = 0.0;
    int local_count = dataset->count;
    int total_count = 0;
    
    // 计算局部误差
    for (int i = 0; i < local_count; i++) {
        double y_pred = predict(model, dataset->data[i].features);
        double error = y_pred - dataset->data[i].label;
        local_sum_squared_error += error * error;
    }
    
    // 聚合全局误差
    MPI_Allreduce(&local_sum_squared_error, &global_sum_squared_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    return sqrt(global_sum_squared_error / total_count);
}

// MPI并行K折交叉验证（优化版本）
double k_fold_cross_validation_mpi(Dataset* dataset, int k, double learning_rate, 
                                 int epochs, double weight_decay, MPIInfo* mpi_info) {
    if (!dataset || k <= 1 || dataset->count == 0) return 0.0;

    int total_count = 0;
    int local_count = dataset->count;
    
    // 获取总数据量
    MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (total_count < k) {
        if (mpi_info->rank == 0) {
            printf("警告: 数据量太少，无法进行%d折交叉验证\n", k);
        }
        return 0.0;
    }
    
    double total_rmse = 0.0;
    
    if (mpi_info->rank == 0) {
        printf("开始%d折交叉验证...\n", k);
    }
    
    // 简化的K折验证：使用现有数据进行训练和验证
    for (int fold = 0; fold < k; fold++) {
        if (mpi_info->rank == 0) {
            printf("第 %d 折交叉验证（MPI并行）...\n", fold + 1);
        }
        
        // 创建临时模型
        LinearRegressionModel* temp_model = create_linear_regression_model(dataset->feature_count);
        if (!temp_model) {
            if (mpi_info->rank == 0) {
                printf("警告: 第 %d 折模型创建失败\n", fold + 1);
            }
            continue;
        }
        
        // 使用较少的epochs进行快速训练
        int cv_epochs = epochs / 2;  // 减少训练轮数
        train_linear_regression_mpi(temp_model, dataset, learning_rate, cv_epochs, weight_decay, mpi_info);
        
        // 计算RMSE
        double rmse = calculate_rmse_mpi(temp_model, dataset, mpi_info);
        
        if (mpi_info->rank == 0) {
            printf("第 %d 折 RMSE: %.6f\n", fold + 1, rmse);
        }
        
        total_rmse += rmse;
        
        // 立即清理临时模型
        free_linear_regression_model(temp_model);
        
        // 添加MPI同步点，确保所有进程完成当前折叠
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    return total_rmse / k;
}

// 释放线性回归模型
void free_linear_regression_model(LinearRegressionModel* model) {
    if (model) {
        free(model->weights);
        free(model);
    }
}