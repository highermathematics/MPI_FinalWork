#include "house_price_prediction.h"

// 创建线性回归模型
LinearRegressionModel* create_linear_regression_model(int feature_count) {
    LinearRegressionModel* model = (LinearRegressionModel*)malloc(sizeof(LinearRegressionModel));
    if (!model) return NULL;

    model->feature_count = feature_count;
    model->weights = (double*)calloc(feature_count, sizeof(double)); // 初始化为0
    model->bias = 0.0;

    if (!model->weights) {
        free(model);
        return NULL;
    }

    return model;
}

// 训练线性回归模型（批量梯度下降）
void train_linear_regression(LinearRegressionModel* model, Dataset* train_data, double learning_rate, int epochs, double weight_decay) {
    if (!model || !train_data || train_data->count == 0) return;

    int feature_count = model->feature_count;
    int sample_count = train_data->count;
    double* dw = (double*)calloc(feature_count, sizeof(double));
    double db = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // 重置梯度
        memset(dw, 0, sizeof(double) * feature_count);
        db = 0.0;

        // 计算梯度
        for (int i = 0; i < sample_count; i++) {
            double y_pred = model->bias;
            for (int j = 0; j < feature_count; j++) {
                y_pred += model->weights[j] * train_data->data[i].features[j];
            }
            double error = y_pred - train_data->data[i].label;

            // 累积梯度
            db += error;
            for (int j = 0; j < feature_count; j++) {
                dw[j] += error * train_data->data[i].features[j];
            }
        }

        // 平均梯度
        db /= sample_count;
        for (int j = 0; j < feature_count; j++) {
            dw[j] /= sample_count;
        }

        // 更新参数（带权重衰减）
        model->bias -= learning_rate * db;
        for (int j = 0; j < feature_count; j++) {
            model->weights[j] -= learning_rate * (dw[j] + weight_decay * model->weights[j]);
        }

        // 每100个epoch打印一次训练进度
        if (epoch % 100 == 0) {
            double rmse = calculate_rmse(model, train_data);
            printf("Epoch %d, RMSE: %.6f\n", epoch, rmse);
        }
    }

    free(dw);
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

// 计算均方根误差（RMSE）
double calculate_rmse(LinearRegressionModel* model, Dataset* dataset) {
    if (!model || !dataset || dataset->count == 0) return 0.0;

    double sum_squared_error = 0.0;
    for (int i = 0; i < dataset->count; i++) {
        double y_pred = predict(model, dataset->data[i].features);
        double error = y_pred - dataset->data[i].label;
        sum_squared_error += error * error;
    }

    return sqrt(sum_squared_error / dataset->count);
}

// K折交叉验证
double k_fold_cross_validation(Dataset* dataset, int k, double learning_rate, int epochs, double weight_decay) {
    if (!dataset || k <= 1 || dataset->count < k) return 0.0;

    int fold_size = dataset->count / k;
    double total_rmse = 0.0;

    // 创建临时数据集用于交叉验证
    Dataset* temp_data = (Dataset*)malloc(sizeof(Dataset));
    temp_data->feature_count = dataset->feature_count;
    temp_data->data = (HouseData*)malloc(sizeof(HouseData) * dataset->count);
    memcpy(temp_data->data, dataset->data, sizeof(HouseData) * dataset->count);
    temp_data->count = dataset->count;

    shuffle_dataset(temp_data);

    for (int i = 0; i < k; i++) {
        printf("第 %d 折交叉验证...\n", i + 1);

        // 划分训练集和验证集
        int start = i * fold_size;
        int end = (i == k - 1) ? dataset->count : start + fold_size;
        int val_count = end - start;

        // 创建验证集
        Dataset val_set;
        val_set.count = val_count;
        val_set.feature_count = dataset->feature_count;
        val_set.data = &temp_data->data[start];

        // 创建训练集
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

        // 训练模型
        LinearRegressionModel* model = create_linear_regression_model(dataset->feature_count);
        train_linear_regression(model, &train_set, learning_rate, epochs, weight_decay);

        // 评估模型
        double rmse = calculate_rmse(model, &val_set);
        printf("第 %d 折 RMSE: %.6f\n", i + 1, rmse);
        total_rmse += rmse;

        // 清理
        free_linear_regression_model(model);
        free(train_set.data);
    }

    // 清理临时数据
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