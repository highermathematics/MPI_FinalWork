该部分为c语言版串行实现
```bash
gcc -Wall -Wextra -O2 -std=c99 -c csv_parser.c -o csv_parser.o
gcc -Wall -Wextra -O2 -std=c99 -c data_preprocessing.c -o data_preprocessing.o
gcc -Wall -Wextra -O2 -std=c99 -c linear_regression.c -o linear_regression.o
gcc main.o csv_parser.o data_preprocessing.o linear_regression.o -o house_price_prediction.exe -lm


./house_price_prediction.exe
```
