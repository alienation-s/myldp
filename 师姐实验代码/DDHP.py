# 学员：南格格
# 时间：2022/4/20 9:42
# 学员：南格格
# 时间：2022/4/18 20:58
#窗口大小固定，区间大小固定，算法DDHP隐私预算下的均方误差值MSE对比，噪音产生方式不同

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
age_edges = list(range(10, 90, 10))
# 窗口大小window
window=200
# .xlsx转.csv文件
data_xls = pd.read_excel(data_path, index_col=0)
data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
# print(data)
age=np.array(list(data[:]['age'])[0:window])
# print(age)
partition_param=8
# LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
# t为时刻
t=1
#多少时刻
dataLength=len(list(data[:]['age']))-window+1
# print(dataLength)
#当前时刻   T_now<dataLength
T_now=20
T_window=window
dataT=np.array(list(data[:]['age'])[0:window])
print(dataT)
print(dataT[2])
# 上一时刻加噪数据
noiseVal_Last=0
dataNoise=[]
dataResult=[]
epslionList=[]
T_window=window
# 第一时刻的第一个数据加噪
# 窗口数据分配的隐私
epsilon=1
# 窗口中每一个数据分配的隐私
epsilon_w=epsilon/window
i=1
epsilonSent=1/epsilon_w
epslionList.append(epsilon_w)
laplace_noise_t = np.random.laplace(0, epsilonSent, 1)
print("1111111111111")
print(laplace_noise_t)
dataNoise.append(dataT[0]+laplace_noise_t)
print(dataT[0])
print(dataNoise[0][0])
dataResult.append(dataNoise[0][0])
print("2222222222")
print(dataResult)
# 第一时刻的窗口第二个数据开始对比加噪
while(i<len(dataT)):
    dataT[i]
    print(np.array([dataT[i]]))
    T=[dataT[i]]-dataNoise[i-1]
    print(i)
    print("33333333")
    if(abs(T)<0.06):
        dataResult.append(dataNoise[i-1][0])
        epslionList.append(0)
    else:
        if(i==len(dataT)-1):
            wt=len(epslionList)-window+1
            epsilon_w=epsilon-sum(epslionList[wt:len(epslionList)])
            print(epsilon_w)
            epslionList.append(epsilon_w)
            print("ssssss")
            print(epsilon_w)
            epsilonSent = 1 / epsilon_w
        else:
            print("dddddd")
            print(epsilon_w)
            epslionList.append(epsilon_w)
            epsilonSent = 1 / epsilon_w
        dataAndNoise=dataT[i]+np.random.laplace(0, epsilonSent, 1)
        dataNoise.append(dataAndNoise[0])
        dataResult.append(dataAndNoise[0])
    i = i + 1
print("444444444")
print(i)
print(dataNoise)
print(dataT)
print(dataResult)
print(epslionList)
# 第二时刻开始到设定时刻对之后相邻数据的对比加噪
len_T=window+T_now-1
while(T_window<len_T):
    # 当前窗口的第一个数据
    T_window = T_window + 1
    # 当前窗口的所有数据集合
    dataT_T=np.array(list(data[:]['age'])[0:T_window])
    print(dataT_T)
    # 输出当前窗口的最后一个数据
    print(dataT_T[len(dataT_T)-1])
    print("kkkkkk")
    # i=window-1,,,, 2
    T = dataT_T[len(dataT_T)-1] - dataNoise[i - 1]
    if(abs(T)<0.06):
        dataResult.append(dataNoise[i - 1][0])
        epslionList.append(0)
    else:
        wt = len(epslionList) - window + 1
        epsilon_w = epsilon - sum(epslionList[wt:len(epslionList)])
        print(epsilon_w)
        epslionList.append(epsilon_w)
        print("rrrrr")
        epsilonSent = 1 / epsilon_w
        dataAndNoise = dataT_T[len(dataT_T)-1]+np.random.laplace(0, epsilonSent, 1)
        print(dataAndNoise)
        dataNoise.append(dataAndNoise[0])
        dataResult.append(dataAndNoise[0])
print(dataNoise)
print(dataT_T)
print(dataResult)
# n_age原始数据频数,n_age_Noise加噪数据频数
n_age, bins, patches = plt.hist(dataT_T, bins=list(range(10, 100, 10)))
n_age_Noise, bins, patches = plt.hist(dataResult, bins=list(range(10, 100, 10)))
print(n_age)
print(n_age_Noise)
valD = sum(math.pow(n_age[index]-n_age_Noise[index],2) for index in range(partition_param))
mseD=math.sqrt(valD)/partition_param
resultMseD.append(mseD)
print(mseD)

# #画折线图
# x1=[0.1,0.5,1.0,1.5,2.0,2.5]
# plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
# plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制
#
# plt.xticks(x1)
# plt.xlim(0.1,2.5)     # x轴坐标范围
# plt.xlabel('different ε')   # x轴标注
# plt.ylabel('Mean Square Error')   # y轴标注
# plt.title('MSE under different algorithm')
# plt.legend()           #图例
# plt.show()
# #
#
