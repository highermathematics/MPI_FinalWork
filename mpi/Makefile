CC = mpicc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm
TARGET = house_price_prediction_mpi

# 源文件和目标文件
SRC = main_mpi.c csv_parser.c data_preprocessing_mpi.c linear_regression_mpi.c
OBJ = $(SRC:.c=.o)

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

# 编译源文件生成目标文件
%.o: %.c house_price_prediction_mpi.h
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标文件和可执行文件
clean:
	rm -f $(OBJ) $(TARGET)

# 运行程序
run: $(TARGET)
	mpirun -n 4 ./$(TARGET)

# 调试运行
debug: $(TARGET)
	mpirun -n 2 ./$(TARGET) --quick

.PHONY: all clean run debug