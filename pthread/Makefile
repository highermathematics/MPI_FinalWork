CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lpthread -lm

# 目标文件
TARGET = house_price_prediction_pthread

# 源文件
SOURCES = main_pthread.c \
          linear_regression_pthread.c \
          data_preprocessing_pthread.c \
          csv_parser_pthread.c

# 对象文件
OBJECTS = $(SOURCES:.c=.o)

# 默认目标
all: $(TARGET)

# 链接目标文件
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# 编译源文件
%.o: %.c house_price_prediction_pthread.h
	$(CC) $(CFLAGS) -c $< -o $@

#清理
clean:
	rm -f *.o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET) --threads 4

# 快速运行
quick: $(TARGET)
	./$(TARGET) --quick --threads 8

.PHONY: all clean run quick