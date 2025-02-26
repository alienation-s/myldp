# 学员：南格格
# 时间：2022/4/18 21:14
# 学员：南格格
# 时间：2022/4/18 20:58
#隐私预算大小固定，区间大小固定，不同算法DDHP与LDP-SWHP在不同窗口大小下的均方误差值MSE对比

import opendp.smartnoise.core as sn
import numpy as np
import math
import random
import noisefirst as nf
import matplotlib.pyplot as plt


# 数据集路径
import pandas as pd

data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'
# 字段
var_names = ["age"]
# 字段范围+区间数量
age_edges = list(range(10, 90, 10))
data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
age = list(data[:]['age'])
partition_param=8
epsilon=0.1
#LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
t = 1
T_now = 10
# 窗口大小window
window=0
windowList=[50,100,150,200,250]
for window in windowList:
    T_window = window
    age = np.array(list(data[:]['age'])[0:window])
    print(len(age))
    # 多少时刻
    dataLength = len(list(data[:]['age'])) - window + 1
    print(dataLength)
    # 当前时刻   T_now<dataLength
    # dataT当前t时刻窗口数据
    dataT = np.array(list(data[:]['age'])[0:window])
    print(dataT)
    np.random.laplace()
    with sn.Analysis(protect_floating_point=False) as analysis:
        nsize = 1000
        age_prep = sn.histogram(sn.to_int(dataT, lower=10, upper=90),
                                edges=age_edges, null_value=10)
        age_histogram = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})

    analysis.release()
    n_age, bins, patches = plt.hist(dataT, bins=list(range(10, 100, 10)), color='w')
    # 原始直方图数据频数
    # print(n_age)
    # 第1时刻加噪直方图数据频数
    noiseVal_T1 = [i if i > 0 else 0 for i in age_histogram.value]
    print(noiseVal_T1)
    noiseVal_Last = noiseVal_T1
    while (t <= T_now):
        t = t + 1
        T_window = T_window + 1
        dataT = np.array(list(data[:]['age'])[0:T_window])
        print("11111")
        # print(dataT)
        with sn.Analysis(protect_floating_point=False) as analysis:
            nsize = 1000
            age_prep_T = sn.histogram(sn.to_int(dataT, lower=10, upper=90),
                                      edges=age_edges, null_value=10)
            age_histogram_T = sn.laplace_mechanism(age_prep_T, privacy_usage={"epsilon": epsilon, "delta": .000001})
        analysis.release()
        n_age, bins, patches = plt.hist(dataT, bins=list(range(10, 100, 10)), color='w')
        # 当前时刻原始直方图数据频数
        # print(n_age)
        # 第t时刻加噪直方图数据频数
        noiseVal_TT = [i if i > 0 else 0 for i in age_histogram_T.value]
        # 相关距离阈值
        T0 = 0.04
        # 相关距离计算
        print(np.array(n_age).shape)
        print(np.array(noiseVal_TT).shape)
        T = 1 - np.corrcoef(np.array(n_age), np.array(noiseVal_Last))[0][1]
        # 相关距离对比：大于阈值=》对当前数据加噪发布’小于阈值=》发布上一时刻加噪数据
        # 对T进行概率扰动
        # 概率扰动p
        p = math.exp(epsilon) / (1 + math.exp(epsilon))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = T > T0
        if v > 0:
            # 数据加噪
            # 分组
            with sn.Analysis(protect_floating_point=False) as analysis:
                # data = sn.Dataset(path=data_path, column_names=var_names)
                nsize = 1000
                age_prep = sn.histogram(sn.to_int(dataT, lower=10, upper=90),
                                        edges=age_edges, null_value=-1)
                age_histogram_T = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})
            analysis.release()
            # 输出当前时刻加噪数据
            noiseVal_T = [i if i > 0 else 0 for i in age_histogram_T.value]
            noisefirst = nf.NoiseFirst(noiseVal_T, epsilon)
            optk = noisefirst.findOptK(noiseVal_T)
            result = noisefirst.getResultHist(noiseVal_T, optk)
            print("2222222222")
            print(result)
            if t > 2:
                noiseVal_Last = result
            else:
                noiseVal_Last = noiseVal_Last
                print(noiseVal_Last)
        else:
            print("333333333")
            # 输出上一时刻加噪数据
            result = noiseVal_Last
            print(result)
    # 当前时刻LDP-SWHP均方误差MSE值
    val = sum(math.pow(n_age[index] - result[index], 2) for index in range(partition_param))
    mse = math.sqrt(val) / partition_param
    resultMse.append(mse)
    # DDHP均方误差MSE值
    valD = sum(math.pow(n_age[index] - noiseVal_TT[index], 2) for index in range(partition_param))
    mseD = math.sqrt(valD) / partition_param
    resultMseD.append(mseD)


#画折线图
x1=[50,100,150,200,250]
xk=[50,100,150,200,250]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制

plt.xticks(xk)
plt.xlim(50,250)     # x轴坐标范围
plt.xlabel('different window=')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend()           #图例
plt.show()

