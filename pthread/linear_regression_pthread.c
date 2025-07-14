#include "house_price_prediction_pthread.h"

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

// 训练线程函数
void* training_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int feature_count = data->model->feature_count;
    
    // 局部梯度累积
    double* local_dw = (double*)calloc(feature_count, sizeof(double));
    double local_db = 0.0;
    
    // 计算该线程负责的数据范围的梯度
    for (int i = data->start_idx; i < data->end_idx; i++) {
        double y_pred = data->model->bias;
        for (int j = 0; j < feature_count; j++) {
            y_pred += data->model->weights[j] * data->dataset->data[i].features[j];
        }
        double error = y_pred - data->dataset->data[i].label;
        
        // 累积梯度
        local_db += error;
        for (int j = 0; j < feature_count; j++) {
            local_dw[j] += error * data->dataset->data[i].features[j];
        }
    }
    
    // 聚合阶段：将局部梯度加到全局梯度中
    pthread_mutex_lock(data->mutex);
    *(data->bias_gradient) += local_db;
    for (int j = 0; j < feature_count; j++) {
        data->gradients[j] += local_dw[j];
    }
    pthread_mutex_unlock(data->mutex);
    
    free(local_dw);
    return NULL;
}

// 并行训练线性回归模型
void train_linear_regression_pthread(LinearRegressionModel* model, Dataset* train_data, 
                                   double learning_rate, int epochs, double weight_decay, int num_threads) {
    if (!model || !train_data || train_data->count == 0) return;

    int feature_count = model->feature_count;
    int sample_count = train_data->count;
    
    // 分配线程数据
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    ThreadData* thread_data = (ThreadData*)malloc(sizeof(ThreadData) * num_threads);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    // 全局梯度数组
    double* gradients = (double*)calloc(feature_count, sizeof(double));
    double bias_gradient = 0.0;
    
    // 划分数据到各个线程
    int chunk_size = sample_count / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].dataset = train_data;
        thread_data[i].model = model;
        thread_data[i].learning_rate = learning_rate;
        thread_data[i].weight_decay = weight_decay;
        thread_data[i].gradients = gradients;
        thread_data[i].bias_gradient = &bias_gradient;
        thread_data[i].mutex = &mutex;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? sample_count : (i + 1) * chunk_size;
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 重置梯度
        memset(gradients, 0, sizeof(double) * feature_count);
        bias_gradient = 0.0;
        
        // 创建线程计算梯度
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, training_thread, &thread_data[i]);
        }
        
        // 等待所有线程完成
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // 平均梯度并更新参数
        bias_gradient /= sample_count;
        model->bias -= learning_rate * bias_gradient;
        
        for (int j = 0; j < feature_count; j++) {
            gradients[j] /= sample_count;
            model->weights[j] -= learning_rate * (gradients[j] + weight_decay * model->weights[j]);
        }
        
        // 每100个epoch打印进度
        if (epoch % 100 == 0) {
            double rmse = calculate_rmse_pthread(model, train_data, num_threads);
            printf("Epoch %d, RMSE: %.6f\n", epoch, rmse);
        }
    }
    
    // 清理资源
    free(threads);
    free(thread_data);
    free(gradients);
    pthread_mutex_destroy(&mutex);
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

// RMSE计算线程函数
void* rmse_calculation_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    double local_sum = 0.0;
    
    for (int i = data->start_idx; i < data->end_idx; i++) {
        double y_pred = predict(data->model, data->dataset->data[i].features);
        double error = y_pred - data->dataset->data[i].label;
        local_sum += error * error;
    }
    
    // 将局部结果加到全局结果中
    pthread_mutex_lock(data->mutex);
    *(data->local_error) += local_sum;
    pthread_mutex_unlock(data->mutex);
    
    return NULL;
}

// 并行计算RMSE
double calculate_rmse_pthread(LinearRegressionModel* model, Dataset* dataset, int num_threads) {
    if (!model || !dataset || dataset->count == 0) return 0.0;

    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    ThreadData* thread_data = (ThreadData*)malloc(sizeof(ThreadData) * num_threads);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    double total_error = 0.0;
    
    int chunk_size = dataset->count / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].dataset = dataset;
        thread_data[i].model = model;
        thread_data[i].mutex = &mutex;
        thread_data[i].local_error = &total_error;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? dataset->count : (i + 1) * chunk_size;
    }
    
    // 创建线程
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, rmse_calculation_thread, &thread_data[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 清理资源
    free(threads);
    free(thread_data);
    pthread_mutex_destroy(&mutex);
    
    return sqrt(total_error / dataset->count);
}

// 并行K折交叉验证
double k_fold_cross_validation_pthread(Dataset* dataset, int k, double learning_rate, 
                                      int epochs, double weight_decay, int num_threads) {
    if (!dataset || k <= 1 || dataset->count < k) return 0.0;

    int fold_size = dataset->count / k;
    double total_rmse = 0.0;

    Dataset* temp_data = (Dataset*)malloc(sizeof(Dataset));
    temp_data->feature_count = dataset->feature_count;
    temp_data->data = (HouseData*)malloc(sizeof(HouseData) * dataset->count);
    memcpy(temp_data->data, dataset->data, sizeof(HouseData) * dataset->count);
    temp_data->count = dataset->count;

    shuffle_dataset(temp_data);

    for (int i = 0; i < k; i++) {
        printf("第 %d 折交叉验证（并行）...\n", i + 1);

        int start = i * fold_size;
        int end = (i == k - 1) ? dataset->count : start + fold_size;
        int val_count = end - start;

        Dataset val_set;
        val_set.count = val_count;
        val_set.feature_count = dataset->feature_count;
        val_set.data = &temp_data->data[start];

        Dataset train_set;
        train_set.count = dataset->count - val_count;
        train_set.feature_count = dataset->feature_count;
        train_set.data = (HouseData*)malloc(sizeof(HouseData) * train_set.count);

        int idx = 0;
        for (int j = 0; j < dataset->count; j++) {
            if (j < start || j >= end) {
                train_set.data[idx++] = temp_data->data[j];
            }
        }

        LinearRegressionModel* model = create_linear_regression_model(dataset->feature_count);
        train_linear_regression_pthread(model, &train_set, learning_rate, epochs, weight_decay, num_threads);

        double rmse = calculate_rmse_pthread(model, &val_set, num_threads);
        printf("第 %d 折 RMSE: %.6f\n", i + 1, rmse);
        total_rmse += rmse;

        free_linear_regression_model(model);
        free(train_set.data);
    }

    free(temp_data->data);
    free(temp_data);

    return total_rmse / k;
}

// 释放线性回归模型内存
void free_linear_regression_model(LinearRegressionModel* model) {
    if (model) {
        free(model->weights);
        free(model);
    }
}