CC = mpicc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = house_price_prediction_mpi
SOURCES = main_mpi.c csv_parser.c data_preprocessing_mpi.c linear_regression_mpi.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c house_price_prediction_mpi.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

test: $(TARGET)
	mpirun -np 2 ./$(TARGET) --quick

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)