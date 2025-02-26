# 学员：南格格
# 时间：2022/4/20 9:42
# 学员：南格格
# 时间：2022/4/18 20:58
#窗口大小固定，区间大小固定，算法DDHP隐私预算下的均方误差值MSE对比，最新版

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
#DDHP下的MSE集合
resultMseD=[]
# t为时刻
t=1
#多少时刻
dataLength=len(list(data[:]['age']))-window+1
# print(dataLength)
#当前时刻   T_now<dataLength
T_now=1000
T_window=window
dataT=np.array(list(data[:]['age'])[0:window])
print(dataT)
print(dataT[2])
# 上一时刻加噪数据集合
dataNoise=[]
# 当前时刻处理后数据集合
dataResult=[]
# 每一个隐私预算集合
epslionList=[]
# 当前窗口最后一个位置
T_window=window
# 第一时刻的第一个数据加噪
epsilon=1.5
# epsilon_w=epsilon/window
# 窗口内的每个数据分配的隐私预算是1
epsilon_w=1.5
i=1
epsilonSent=1/epsilon_w
epslionList.append(epsilon_w)
# laplace_noise_t = np.random.laplace(0, epsilonSent, 1)
x=random.random()
print(x)
if(x<0.5):
    print("nnnnn")
    print(math.log(2*x))
    laplace_noise_t=epsilonSent*math.log(2*x)
else:
    print(math.log(2-2*x))
    laplace_noise_t=-epsilonSent*(math.log(2-2*x))
print(laplace_noise_t)
print("1111111111111")
dataNoise.append(dataT[0]+laplace_noise_t)
print(dataT[0])
print(dataNoise[0])
print(dataNoise)
dataResult.append(dataNoise[0])
print("2222222222")
print(dataResult)
# 第一时刻的窗口第二个数据开始对比加噪
while(i<len(dataT)):
    dataT[i]
    # print(dataNoise[i-1])
    T=dataT[i]-dataNoise[-1]
    print(i)
    print("33333333")
    if(abs(T)<2):
        print("oooooooo")
        dataResult.append(dataNoise[-1])
        dataNoise.append(dataNoise[-1])
        epslionList.append(0)
    else:
        if(i==len(dataT)-1):
            wt=len(epslionList)-window+1
            epsilon_w=epsilon*window-sum(epslionList[wt:len(epslionList)])
            if(epsilon_w==0):
                dataResult.append(dataNoise[-1])
                dataNoise.append(dataNoise[-1])
                epslionList.append(0)
            else:
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
        y = random.random()
        print(y)
        if (y < 0.5):
            print("nnnnn")
            print(math.log(2 * y))
            laplace_noise_t = epsilonSent * math.log(2 * y)
        else:
            print(math.log(2 - 2 * y))
            laplace_noise_t = -epsilonSent * (math.log(2 - 2 * y))
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
print(epslionList)
# 第二时刻开始到设定时刻对之后相邻数据的对比加噪
len_T=window+T_now-1
while(T_window<len_T):
    T_window = T_window + 1
    dataT_T=np.array(list(data[:]['age'])[0:T_window])
    print(dataT_T)
    print(dataT_T[len(dataT_T)-1])
    print("kkkkkk")
    # i=window-1,,,, 2
    T = dataT_T[len(dataT_T)-1] - dataNoise[-1]
    if(abs(T)<2):
        dataResult.append(dataNoise[-1])
        epslionList.append(0)
    else:
        wt = len(epslionList) - window + 1
        epsilon_w = epsilon*window - sum(epslionList[wt:len(epslionList)])
        if(epsilon_w==0):
            dataResult.append(dataNoise[-1])
            epslionList.append(0)
        else:
            print(epsilon_w)
            epslionList.append(epsilon_w)
            print("rrrrr")
            epsilonSent = 1 / epsilon_w
            y = random.random()
            print(y)
            if (y < 0.5):
                print("nnnnn")
                print(math.log(2 * y))
                laplace_noise_t = epsilonSent * math.log(2 * y)
            else:
                print(math.log(2 - 2 * y))
                laplace_noise_t = -epsilonSent * (math.log(2 - 2 * y))
            print(laplace_noise_t)
            dataAndNoise = dataT_T[len(dataT_T)-1] + laplace_noise_t
            print(dataAndNoise)
            dataNoise.append(dataAndNoise)
            dataResult.append(dataAndNoise)
print(dataNoise)
print(i)
print(dataT_T)
print(dataResult)
# n_age原始数据频数,n_age_Noise加噪数据频数
n_age, bins, patches = plt.hist(dataT_T, bins=list(range(0, 100, 10)))
n_age_Noise, bins, patches = plt.hist(dataResult, bins=list(range(0, 100, 10)))
print(n_age)
print(n_age_Noise)
print(len(dataResult))
valD = sum(math.pow(n_age[index]-n_age_Noise[index],2) for index in range(partition_param))
mseD=math.sqrt(valD)/partition_param
resultMseD.append(mseD)
print(mseD)

# #画折线图
# x1=[0.5]
# plt.plot(x1, mseD ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
# # plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制
#
# plt.xticks(x1)
# plt.xlim(0.5,2.5)     # x轴坐标范围
# plt.ylim(0,5)     # x轴坐标范围
# plt.xlabel('different ε')   # x轴标注
# plt.ylabel('Mean Square Error')   # y轴标注
# plt.title('MSE under different algorithm')
# plt.legend()           #图例
# plt.show()
# #
