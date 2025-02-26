# 学员：南格格
# 时间：2022/5/14 15:47
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'

# .xlsx转.csv文件
data_xls = pd.read_excel(data_path, index_col=0)
data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
epsilonCList=[1.0,2.0,3.0,4.0,5.0]
partition_param=9
resultMseC=[]
# 窗口内的每个数据分配的隐私预算
for epsilon_w_C in epsilonCList:
    T_now=200
    size_window = 100
    # 上一时刻加噪数据集合
    dataNoise = []
    # 当前时刻处理后数据集合
    dataResult = []
    # 每一个隐私预算集合
    epslionListDD = []
    dataT = np.array(list(data[:]['age'])[0:size_window])
    # 当前窗口最后一个位置
    T_window_D = size_window
    epsilon_D = epsilon_w_C
    # epsilon_w=epsilon/window
    i = 1
    epsilonSent_D = 1 / epsilon_w_C
    epslionListDD.append(epsilon_w_C)
    # laplace_noise_t = np.random.laplace(0, epsilonSent, 1)
    x = random.random()
    print(x)
    if (x < 0.5):
        print("nnnnn")
        print(math.log(2 * x))
        laplace_noise_t = epsilonSent_D * math.log(2 * x)
    else:
        print(math.log(2 - 2 * x))
        laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * x))
    print(laplace_noise_t)
    print("1111111111111")
    dataNoise.append(dataT[0] + laplace_noise_t)
    print(dataT[0])
    print(dataNoise[0])
    print(dataNoise)
    dataResult.append(dataNoise[0])
    print("2222222222")
    print(dataResult)
    # 第一时刻的窗口第二个数据开始对比加噪
    while (i < len(dataT)):
        dataT[i]
        # print(dataNoise[i-1])
        T = dataT[i] - dataNoise[-1]
        T0=2
        print(i)
        print("33333333")
        p = math.exp(epsilon_w_C) / (1 + math.exp(epsilon_w_C))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = abs(T) > T0
        if v < 0:
            print("oooooooo")
            dataResult.append(dataNoise[-1])
            dataNoise.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            if (i == len(dataT) - 1):
                wt = len(epslionListDD) - size_window + 1
                epsilon_w_C = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
                print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                print("ssssss")
                print(epsilon_w_C)
                epsilonSent_D = 1 / epsilon_w_C
            else:
                print("dddddd")
                print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                epsilonSent_D = 1 / epsilon_w_C
            y = random.random()
            print(y)
            if (y < 0.5):
                print("nnnnn")
                print(math.log(2 * y))
                laplace_noise_t = epsilonSent_D * math.log(2 * y)
            else:
                print(math.log(2 - 2 * y))
                laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
            print(laplace_noise_t)
            dataAndNoise = dataT[i] + laplace_noise_t
            dataNoise.append(dataAndNoise)
            dataResult.append(dataAndNoise)
        i = i + 1
    print("444444444")
    print(i)
    print(dataNoise)
    print(dataT)
    print(dataResult)
    print(epslionListDD)
    # 第二时刻开始到设定时刻对之后相邻数据的对比加噪
    len_T = size_window + T_now - 1
    while (T_window_D < len_T):
        T_window_D = T_window_D + 1
        dataT_T = np.array(list(data[:]['age'])[0:T_window_D])
        print(dataT_T)
        print(dataT_T[len(dataT_T) - 1])
        print("kkkkkk")
        # i=window-1,,,, 2
        T = dataT_T[len(dataT_T) - 1] - dataNoise[-1]
        p = math.exp(epsilon_w_C) / (1 + math.exp(epsilon_w_C))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = abs(T) > T0
        if v < 0:
            dataResult.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            wt = len(epslionListDD) - size_window + 1
            epsilon_w_C = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
            if (epsilon_w_C == 0):
                dataResult.append(dataNoise[-1])
                epslionListDD.append(0)
            else:
                print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                print("rrrrr")
                epsilonSent_D = 1 / epsilon_w_C
                y = random.random()
                print(y)
                if (y < 0.5):
                    print("nnnnn")
                    print(math.log(2 * y))
                    laplace_noise_t = epsilonSent_D * math.log(2 * y)
                else:
                    print(math.log(2 - 2 * y))
                    laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
                print(laplace_noise_t)
                dataAndNoise = dataT_T[len(dataT_T) - 1] + laplace_noise_t
                print(dataAndNoise)
                dataNoise.append(dataAndNoise)
                dataResult.append(dataAndNoise)
    print(dataNoise)
    print(i)
    print("poririririrrrrrr")
    print(len(dataT_T))
    print(dataT_T[T_now-1:T_now+size_window-1])
    print(len(dataT_T[T_now-1:T_now+size_window-1]))
    print(dataResult)
    # 当前时刻DDHP均方误差MSE值
    # n_age原始数据频数,n_age_Noise加噪数据频数
    n_age, bins, patches = plt.hist(dataT_T[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    n_age_Noise, bins, patches = plt.hist(dataResult[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    print(n_age)
    print(n_age_Noise)
    print(len(dataResult))
    valC = sum(math.pow(n_age[index] - n_age_Noise[index], 2) for index in range(partition_param))
    mseC = math.sqrt(valC) / partition_param
    resultMseC.append(mseC)

print("#####################")
print(resultMseC)
print("#####################")
print("9999999")

