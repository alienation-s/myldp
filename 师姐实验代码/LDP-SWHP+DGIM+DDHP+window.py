# 学员：南格格
# 时间：2022/5/1 20:4
import numpy as np
import math
import random

import pandas as pd

import noisefirst as nf
import matplotlib.pyplot as plt
import DGIM as dj
data_path = 'E:/系统缓存/桌面/实验代码/数据集区间文件流/'
partition_param=9
# LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
# 窗口的大小
windowList=[50,100,150,200,250]
for size_window in windowList:
    epsilon3 = 0.5
    # 第二部分隐私预算
    epsilon2=epsilon3
    # 隐私预算集合
    epsilonList = []
    # 输入最大相同桶的数量
    n_max_bucket = 10
    # 当前时刻
    time_location = 1
    # print("1111111111111")
    # 上一时刻加噪数据
    noiseVal_Last = 0
    epsilon_1 = epsilon3
    # 总的隐私预算
    T_now=500
    epsilon = epsilon_1 * T_now
    epsilonList.append(epsilon3)
    epsilonSent=1/epsilon3
    laplace_noise = np.random.laplace(0, epsilonSent , partition_param)
    # 第1时刻加噪直方图数据频数
    # print(len(laplace_noise))
    djim = dj.DGIM(n_max_bucket, size_window,time_location,data_path)
    noise_count,original_count=djim.getNoiseCount()
    noiseVal_T1 = noise_count+laplace_noise
    noiseVal_T1 = [i if i > 0 else 0 for i in noiseVal_T1]
    # print("2222222")
    # print(noiseVal_T1)
    noiseVal_Last=noiseVal_T1
    # print(time_location)
    while (time_location < T_now):
        time_location = time_location + 1
        # print("3333333")
        print(time_location)
        djim = dj.DGIM(n_max_bucket, size_window, time_location, data_path)
        noise_count, original_count = djim.getNoiseCount()
        # 相关距离阈值
        T0 = 0.04
        # 相关距离计算
        # print(np.array(noise_count).shape)
        # print(np.array(noiseVal_Last).shape)
        T = 1 - np.corrcoef(np.array(noise_count), np.array(noiseVal_Last))[0][1]
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
            # print(epsilon)
            # print(sum(epsilonList[0:len(epsilonList)]))
            epsilon3=(epsilon-sum(epsilonList[0:len(epsilonList)]))/(T_now-(time_location-1))
            # print("sssssss")
            # print(epsilon3)
            epsilonList.append(epsilon3)
            if(epsilon3==0):
                result=noiseVal_Last
            else:
                # print(epsilon3)
                epsilonSent = 1 / epsilon3
                laplace_noise = np.random.laplace(0, epsilonSent, partition_param)
                noiseVal_T=noise_count+laplace_noise
                # 动态规划分组
                noisefirst = nf.NoiseFirst(noiseVal_T, epsilon3)
                optk = noisefirst.findOptK(noiseVal_T)
                result = noisefirst.getResultHist(noiseVal_T, optk)
                result = [i if i > 0 else 0 for i in result]
                # print("444444444")
                # print(result)
                noiseVal_Last = result

        else:
            # print("55555555555")
            # 输出上一时刻加噪数据
            result=noiseVal_Last
            # print(result)
    #当前时刻LDP-SWHP均方误差MSE值
    # print("ppppp")
    # print(noise_count)
    # print(result)
    val = sum(math.pow(original_count[index]-result[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultMse.append(mse)

########################################DDHP算法########################################
data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'

# .xlsx转.csv文件
data_xls = pd.read_excel(data_path, index_col=0)
data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
windowList=[50,100,150,200,250]
for size_window in windowList:
    T_now = 500
    epsilon_w_D = 1
    # 上一时刻加噪数据集合
    dataNoise = []
    # 当前时刻处理后数据集合
    dataResult = []
    # 每一个隐私预算集合
    epslionListDD = []
    dataT = np.array(list(data[:]['age'])[0:size_window])
    # 当前窗口最后一个位置
    T_window_D = size_window
    epsilon_D = epsilon_w_D
    # epsilon_w=epsilon/window
    i = 1
    epsilonSent_D = 1 / epsilon_w_D
    epslionListDD.append(epsilon_w_D)
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
        print(i)
        print("33333333")
        if (abs(T) < 5):
            print("oooooooo")
            dataResult.append(dataNoise[-1])
            dataNoise.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            if (i == len(dataT) - 1):
                wt = len(epslionListDD) - size_window + 1
                epsilon_w_D = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
                print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                print("ssssss")
                print(epsilon_w_D)
                epsilonSent_D = 1 / epsilon_w_D
            else:
                print("dddddd")
                print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                epsilonSent_D = 1 / epsilon_w_D
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
        if (abs(T) < 5):
            dataResult.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            wt = len(epslionListDD) - size_window + 1
            epsilon_w_D = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
            if (epsilon_w_D == 0):
                dataResult.append(dataNoise[-1])
                epslionListDD.append(0)
            else:
                print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                print("rrrrr")
                epsilonSent_D = 1 / epsilon_w_D
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
    print(dataT_T[T_now - 1:T_now + size_window - 1])
    print(len(dataT_T[T_now - 1:T_now + size_window - 1]))
    print(dataResult)
    # 当前时刻DDHP均方误差MSE值
    # n_age原始数据频数,n_age_Noise加噪数据频数
    n_age, bins, patches = plt.hist(dataT_T[T_now - 1:T_now + size_window - 1], bins=list(range(0, 100, 10)), color='w')
    n_age_Noise, bins, patches = plt.hist(dataResult[T_now - 1:T_now + size_window - 1], bins=list(range(0, 100, 10)),
                                          color='w')
    print(n_age)
    print(n_age_Noise)
    print(len(dataResult))
    valD = sum(math.pow(n_age[index] - n_age_Noise[index], 2) for index in range(partition_param))
    mseD = math.sqrt(valD) / partition_param
    resultMseD.append(mseD)
print("#####################")
print(resultMse)
print("#####################")
print(epsilonList)
print("9999999")
# #画折线图
x1=[50,100,150,200,250]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制

plt.xticks(x1)
plt.xlim(50,250)     # x轴坐标范围
plt.ylim(0,5)     # y轴坐标范围
plt.xlabel('different w')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend()
plt.show()