CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm
TARGET = house_price_prediction

# 源文件和目标文件
SRC = main.c csv_parser.c data_preprocessing.c linear_regression.c
OBJ = $(SRC:.c=.o)

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

# 编译源文件生成目标文件
%.o: %.c house_price_prediction.h
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标文件和可执行文件
clean:
	del /f /q $(OBJ) $(TARGET).exe

# 运行程序
run: $(TARGET)
	$(TARGET)

.PHONY: all clean run