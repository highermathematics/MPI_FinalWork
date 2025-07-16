# 北京化工大学并行程序设计大作业

> [题目](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
> 该项目思路来源于[d2l-ai](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_multilayer-perceptrons/kaggle-house-price.md)

**.py** 文件夹下是将源代码优化之后的结果（已改为并行程序）

**.c** 文件夹下是.py文件夹项目的c语言版（非并行程序）

**mpi**为使用OpenMPI实现的并行程序版本

```bash
#编译
make

#运行（4个进程）
 make run

#运行（指定线程数）：
mpirun -np <线程数> ./house_price_prediction_mpi

#--help 查看
mpirun -np 4 ./house_price_prediction_mpi --help

#清理编译文件
make clean
```

**pthread**为使用pthread库完成的并行程序版本

```bash
#编译
make

#运行（4个进程）
 make run

#运行（指定线程数）：
./house_price_prediction_pthread --threads <线程数>

#--help 查看
./house_price_prediction_pthread --help

#清理编译文件
make clean
```

**data_analyse**为数据分析部分

```python
python Train.py
python visualize_results.py
```
