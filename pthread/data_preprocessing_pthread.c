#include "house_price_prediction_pthread.h"

// 专门用于处理缺失值的线程函数
void* missing_values_thread(void* arg) {
    PreprocessThreadData* data = (PreprocessThreadData*)arg;
    
    // 处理该线程负责的特征范围
    for (int i = data->start_idx; i < data->end_idx; i++) {
        // 计算均值
        double sum = 0.0;
        int valid_count = 0;
        
        for (int j = 0; j < data->dataset->count; j++) {
            double val = data->dataset->data[j].features[i];
            if (!isnan(val)) {
                sum += val;
                valid_count++;
            }
        }
        
        data->mean[i] = (valid_count > 0) ? sum / valid_count : 0.0;
    }
    
    return NULL;
}

// 预处理线程函数（用于标准化）
void* preprocessing_thread(void* arg) {
    PreprocessThreadData* data = (PreprocessThreadData*)arg;
    
    // 处理该线程负责的特征范围
    for (int i = data->start_idx; i < data->end_idx; i++) {
        // 计算均值
        double sum = 0.0;
        int valid_count = 0;
        
        for (int j = 0; j < data->dataset->count; j++) {
            double val = data->dataset->data[j].features[i];
            if (!isnan(val)) {
                sum += val;
                valid_count++;
            }
        }
        
        data->mean[i] = (valid_count > 0) ? sum / valid_count : 0.0;
        
        // 计算标准差（只在std不为NULL时执行）
        if (data->std != NULL) {
            double variance = 0.0;
            valid_count = 0;
            for (int j = 0; j < data->dataset->count; j++) {
                double val = data->dataset->data[j].features[i];
                if (!isnan(val)) {
                    variance += pow(val - data->mean[i], 2);
                    valid_count++;
                }
            }
            
            data->std[i] = (valid_count > 1) ? sqrt(variance / (valid_count - 1)) : 1.0;
            if (data->std[i] < 1e-6) data->std[i] = 1.0;
        }
    }
    
    return NULL;
}

// 并行处理缺失值
void handle_missing_values_pthread(Dataset* dataset, int num_threads) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    double* mean = (double*)malloc(sizeof(double) * feature_count);
    
    // 并行计算每个特征的均值
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    PreprocessThreadData* thread_data = (PreprocessThreadData*)malloc(sizeof(PreprocessThreadData) * num_threads);
    
    int chunk_size = feature_count / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].dataset = dataset;
        thread_data[i].mean = mean;
        thread_data[i].std = NULL;  // 处理缺失值时不需要std
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? feature_count : (i + 1) * chunk_size;
        
        // 使用专门的缺失值处理线程函数
        pthread_create(&threads[i], NULL, missing_values_thread, &thread_data[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 用均值填充缺失值
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < dataset->count; j++) {
            if (isnan(dataset->data[j].features[i])) {
                dataset->data[j].features[i] = mean[i];
            }
        }
    }
    
    free(threads);
    free(thread_data);
    free(mean);
}

// 并行特征标准化
void normalize_features_pthread(Dataset* dataset, int num_threads) {
    if (!dataset || dataset->count == 0 || dataset->feature_count == 0) return;

    int feature_count = dataset->feature_count;
    double* mean = (double*)malloc(sizeof(double) * feature_count);
    double* std = (double*)malloc(sizeof(double) * feature_count);
    
    // 并行计算均值和标准差
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    PreprocessThreadData* thread_data = (PreprocessThreadData*)malloc(sizeof(PreprocessThreadData) * num_threads);
    
    int chunk_size = feature_count / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].dataset = dataset;
        thread_data[i].mean = mean;
        thread_data[i].std = std;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == num_threads - 1) ? feature_count : (i + 1) * chunk_size;
        
        pthread_create(&threads[i], NULL, preprocessing_thread, &thread_data[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 标准化特征
    for (int i = 0; i < feature_count; i++) {
        for (int j = 0; j < dataset->count; j++) {
            if (std[i] > 1e-6) {
                dataset->data[j].features[i] = (dataset->data[j].features[i] - mean[i]) / std[i];
            } else {
                dataset->data[j].features[i] = 0.0;
            }
        }
    }
    
    free(threads);
    free(thread_data);
    free(mean);
    free(std);
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

// 划分训练集和测试集
Dataset* train_test_split(Dataset* dataset, double test_size) {
    if (!dataset || test_size <= 0 || test_size >= 1) return NULL;

    shuffle_dataset(dataset);
    int test_count = (int)(dataset->count * test_size);
    int train_count = dataset->count - test_count;

    Dataset* test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->count = test_count;
    test_dataset->feature_count = dataset->feature_count;
    test_dataset->data = (HouseData*)malloc(sizeof(HouseData) * test_count);

    for (int i = 0; i < test_count; i++) {
        test_dataset->data[i].id = dataset->data[train_count + i].id;
        test_dataset->data[i].label = dataset->data[train_count + i].label;
        test_dataset->data[i].features = copy_features(
            dataset->data[train_count + i].features, dataset->feature_count
        );
    }

    dataset->count = train_count;
    return test_dataset;
}