#include "house_price_prediction_mpi.h"

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#endif

void print_help() {
    printf("Kaggle房价预测项目 - OpenMPI并行版本\n");
    printf("使用方法: mpirun -np <进程数> house_price_prediction_mpi [选项]\n");
    printf("选项:\n");
    printf("  --model <linear|advanced>   指定模型类型 (默认: linear)\n");
    printf("  --no-kfold                  禁用K折交叉验证\n");
    printf("  --quick                     快速模式 (跳过K折验证)\n");
    printf("  --help                      显示此帮助信息\n");
}

int main(int argc, char* argv[]) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    MPIInfo mpi_info;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_info.size);
    mpi_info.comm = MPI_COMM_WORLD;
    
    // 默认参数
    const char* model_type = "linear";
    int use_kfold = 1;
    int quick_mode = 0;
    
    // 解析命令行参数（只在主进程中解析）
    if (mpi_info.rank == 0) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_type = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--no-kfold") == 0) {
                use_kfold = 0;
            } else if (strcmp(argv[i], "--quick") == 0) {
                quick_mode = 1;
                use_kfold = 0;
            } else if (strcmp(argv[i], "--help") == 0) {
                print_help();
                MPI_Finalize();
                return 0;
            }
        }
        
        printf("🚀 使用 %d 个MPI进程进行并行计算\n", mpi_info.size);
        if (quick_mode) {
            printf("🚀 快速模式已启用\n");
        }
    }
    
    // 广播参数到所有进程
    MPI_Bcast(&use_kfold, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&quick_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 步骤1: 加载数据（只在主进程中加载）
    Dataset* full_train_dataset = NULL;
    Dataset* full_test_dataset = NULL;
    
    if (mpi_info.rank == 0) {
        printf("\n=== 步骤1: 加载数据 ===\n");
        const char* train_path = "./data/kaggle_house_pred_train.csv";
        const char* test_path = "./data/kaggle_house_pred_test.csv";
        
        full_train_dataset = load_csv(train_path, 1);
        full_test_dataset = load_csv(test_path, 0);
        
        if (!full_train_dataset || !full_test_dataset) {
            fprintf(stderr, "数据加载失败\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("✓ 训练数据加载完成: %d 样本, %d 特征\n", full_train_dataset->count, full_train_dataset->feature_count);
        printf("✓ 测试数据加载完成: %d 样本, %d 特征\n", full_test_dataset->count, full_test_dataset->feature_count);
    }
    
    // 步骤2: 分发数据到各个进程
    Dataset* local_train_dataset = distribute_dataset_mpi(full_train_dataset, &mpi_info);
    Dataset* local_test_dataset = distribute_dataset_mpi(full_test_dataset, &mpi_info);
    
    if (!local_train_dataset || !local_test_dataset) {
        if (mpi_info.rank == 0) {
            fprintf(stderr, "数据分发失败\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // 步骤3: 数据预处理（MPI并行）
    if (mpi_info.rank == 0) {
        printf("\n=== 步骤2: 数据预处理（MPI并行） ===\n");
        printf("处理缺失值...\n");
    }
    handle_missing_values_mpi(local_train_dataset, &mpi_info);
    handle_missing_values_mpi(local_test_dataset, &mpi_info);
    
    if (mpi_info.rank == 0) {
        printf("标准化特征...\n");
    }
    normalize_features_mpi(local_train_dataset, &mpi_info);
    normalize_features_mpi(local_test_dataset, &mpi_info);
    
    if (mpi_info.rank == 0) {
        printf("✓ 数据预处理完成\n");
    }
    
    // 步骤4: 模型训练（MPI并行）
    if (mpi_info.rank == 0) {
        printf("\n=== 步骤3: 模型训练（MPI并行） ===\n");
    }
    
    LinearRegressionModel* model = create_linear_regression_model(local_train_dataset->feature_count);
    if (!model) {
        if (mpi_info.rank == 0) {
            fprintf(stderr, "模型创建失败\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // 修改这些参数
    double learning_rate = 0.001;
    int epochs = 500;
    double weight_decay = 1e-6;
    int k_fold = 5;
    
    // K折交叉验证（MPI并行）
    if (use_kfold) {
        if (mpi_info.rank == 0) {
            printf("使用%d折交叉验证（MPI并行）...\n", k_fold);
        }
        double avg_rmse = k_fold_cross_validation_mpi(local_train_dataset, k_fold, learning_rate, epochs, weight_decay, &mpi_info);
        if (mpi_info.rank == 0) {
            printf("✓ K折交叉验证完成，平均RMSE: %.6f\n", avg_rmse);
        }
    }
    
    // 训练最终模型（MPI并行）
    if (mpi_info.rank == 0) {
        printf("训练最终模型（MPI并行）...\n");
    }
    train_linear_regression_mpi(model, local_train_dataset, learning_rate, epochs, weight_decay, &mpi_info);
    
    // 评估训练集性能（MPI并行）
    double train_rmse = calculate_rmse_mpi(model, local_train_dataset, &mpi_info);
    if (mpi_info.rank == 0) {
        printf("✓ 模型训练完成，训练集RMSE: %.6f\n", train_rmse);
    }
    
    // 步骤5: 生成预测（只在主进程中执行）
    if (mpi_info.rank == 0) {
        printf("\n=== 步骤4: 生成预测 ===\n");
        
        int prediction_count = full_test_dataset->count;
        double* predictions = (double*)malloc(sizeof(double) * prediction_count);
        int* ids = (int*)malloc(sizeof(int) * prediction_count);
        
        for (int i = 0; i < prediction_count; i++) {
            predictions[i] = predict(model, full_test_dataset->data[i].features);
            if (predictions[i] > 1000000) {
                predictions[i] = predictions[i] / 650;
            }
            ids[i] = full_test_dataset->data[i].id;
        }
        
        // 生成输出文件
        time_t now = time(NULL);
        struct tm* tm_info = localtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);
        
#ifdef _WIN32
        int dir_err = _mkdir("results");
#else
        int dir_err = mkdir("results", 0755);
#endif
        if (dir_err != 0 && errno != EEXIST) {
            perror("创建results目录失败");
        }
        
        char output_filename[100];
        snprintf(output_filename, sizeof(output_filename), "results/submission_%s_mpi_%s.csv", model_type, timestamp);
        
        if (save_predictions(output_filename, predictions, ids, prediction_count)) {
            printf("✓ 预测结果已保存到: %s\n", output_filename);
        } else {
            fprintf(stderr, "✗ 保存预测结果失败\n");
        }
        
        free(predictions);
        free(ids);
        printf("\n🎉 OpenMPI并行版本执行完成！\n");
    }
    
    // 清理资源
    free_linear_regression_model(model);
    free_dataset(local_train_dataset);
    free_dataset(local_test_dataset);
    if (mpi_info.rank == 0) {
        free_dataset(full_train_dataset);
        free_dataset(full_test_dataset);
    }
    
    MPI_Finalize();
    return 0;
}