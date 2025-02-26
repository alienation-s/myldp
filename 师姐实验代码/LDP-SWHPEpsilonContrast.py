# 学员：南格格
# 时间：2022/4/18 20:58
#窗口大小固定，区间大小固定，不同算法DDHP与LDP-SWHP在不同隐私预算下的均方误差值MSE对比

import opendp.smartnoise.core as sn
import numpy as np
import math
import random
import noisefirst as nf
import matplotlib.pyplot as plt


# 数据集路径
import pandas as pd

data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'
# 字段
var_names = ["age"]
# 字段范围+区间数量
age_edges = list(range(0, 100, 10))
# 窗口大小window
window=50
# .xlsx转.csv文件
data_xls = pd.read_excel(data_path, index_col=0)
data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
# print(data)
age=np.array(list(data[:]['age'])[0:window])
# print(age)
partition_param=9
# LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
# t为时刻
t=1
# print(dataLength)
#当前时刻   T_now<dataLength
T_now=100
# 第三部分隐私预算
epsilon3=0
# 第二部分隐私预算
epsilon2=0.5
k=1
epsilon3List=[0.5,1.0,1.5,2.0,2.5]
for epsilon3 in epsilon3List:
    k=k+1
    # 隐私预算集合
    epsilonList = []
    T_window = window
    dataT = np.array(list(data[:]['age'])[0:window])
    # 多少时刻
    dataLength = len(list(data[:]['age'])) - window + 1
    print("1111111111111")
    print(dataT)
    t=1
    # 上一时刻加噪数据
    noiseVal_Last = 0
    epsilon_1 = epsilon3
    # 总的隐私预算
    epsilon = epsilon_1 * dataLength
    epsilonList.append(epsilon3)
    n_age, bins, patches = plt.hist(dataT, bins=list(range(0, 100, 10)),color='w')
    # 原始直方图数据频数
    print(n_age)
    epsilonSent=1/epsilon3
    laplace_noise = np.random.laplace(0, epsilonSent , partition_param)
    # 第1时刻加噪直方图数据频数
    noiseVal_T1 = n_age+laplace_noise
    noiseVal_T1 = [i if i > 0 else 0 for i in noiseVal_T1]
    print("2222222")
    print(noiseVal_T1)
    noiseVal_Last=noiseVal_T1
    while (t <= T_now):
        t = t + 1
        T_window = T_window + 1
        dataT = np.array(list(data[:]['age'])[0:T_window])
        print("3333333")
        print(dataT)
        n_age, bins, patches = plt.hist(dataT, bins=list(range(0, 100, 10)), color='w')
        # 当前时刻原始直方图数据频数
        # print(n_age)
        # 相关距离阈值
        T0 = 0.04
        # 相关距离计算
        print(np.array(n_age).shape)
        print(np.array(noiseVal_Last).shape)
        T = 1 - np.corrcoef(np.array(n_age), np.array(noiseVal_Last))[0][1]
        # 相关距离对比：大于阈值=》对当前数据加噪发布’小于阈值=》发布上一时刻加噪数据
        # 对T进行概率扰动
        #概率扰动p
        p = math.exp(epsilon2) / (1 + math.exp(epsilon2))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = T > T0
        if v>0:
            # 数据加噪
            # 分组
            print(epsilon)
            print(sum(epsilonList[0:len(epsilonList)]))
            epsilon3=(epsilon-sum(epsilonList[0:len(epsilonList)]))/(dataLength-(t-1))
            print("sssssss")
            print(epsilon3)
            epsilonList.append(epsilon3)
            if(epsilon3==0):
                result=noiseVal_Last
            else:
                print(epsilon3)
                epsilonSent = 1 / epsilon3
                laplace_noise = np.random.laplace(0, epsilonSent, partition_param)
                noiseVal_T=n_age+laplace_noise
                # 动态规划分组
                noisefirst = nf.NoiseFirst(noiseVal_T, epsilon3)
                optk = noisefirst.findOptK(noiseVal_T)
                result = noisefirst.getResultHist(noiseVal_T, optk)
                result = [i if i > 0 else 0 for i in result]
                print("444444444")
                print(result)
                noiseVal_Last = result

        else:
            print("55555555555")
            # 输出上一时刻加噪数据
            result=noiseVal_Last
            print(result)
        #当前时刻LDP-SWHP均方误差MSE值
    val = sum(math.pow(n_age[index]-result[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultMse.append(mse)
    print(result)
    print(n_age)
    print(epsilonList)
print("9999999")
print(k)
print(resultMse)
print(dataLength)
# #画折线图
x1=[0.5,1.0,1.5,2.0,2.5]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
# plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制

plt.xticks(x1)
plt.xlim(0.5,2.5)     # x轴坐标范围
plt.ylim(0,5)     # x轴坐标范围
plt.xlabel('different ε')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend()           #图例
plt.show()
#
