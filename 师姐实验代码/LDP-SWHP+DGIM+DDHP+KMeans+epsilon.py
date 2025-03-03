# 学员：南格格
# 时间：2022/5/1 20:4
import time

import numpy as np
import math
import random

import pandas as pd

import noisefirst as nf
import matplotlib.pyplot as plt
import DGIM as dj
data_path = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流/'
data_path = '/Users/alasong/Documents/workspace/PP_LDP/师姐实验代码/数据集区间文件流/'

partition_param=9
# LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
resultMseC=[]
# 时间集合
timeList=[]
timeDDList=[]
timeCCList=[]
# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


# 创建数据集
def createDataSet():
    data_path = '/Users/alasong/Documents/workspace/PP_LDP/师姐实验代码/Car1.csv'
    df = pd.read_csv(data_path)
    window=100
    t=1000
    age = list(df[:]['age'])
    data = np.array(age[t-1:window+t-1])
    print(data)
    result = []
    for i in range(len(data)):
        datalist = []
        datalist.append(data[i])
        result.append(datalist)
    return result,data
    # return [1,1,2,50,60,55]
    # return [[1], [1], [2], [6], [6], [5]]

def dataMNoise(dataSet,epsilon):
    noiseVal = []
    epsilonSent = 1 / epsilon
    for i in range(len(dataSet)):
        x = random.random()
        if (x < 0.5):
            laplace_noise = epsilonSent * math.log(2 * x)
        else:
            laplace_noise = -epsilonSent * (math.log(2 - 2 * x))
        noiseVal.append(dataSet[i] + laplace_noise)
    print(noiseVal)
    dataSet, bins, patches = plt.hist(noiseVal, bins=list(range(0, 100, 10)),color='w')
    dataSet = [i if i > 0 else 0 for i in dataSet]
    return dataSet

def countNoise(dataSet,epsilon):
    noiseVal = []
    epsilonSent = 1 / epsilon
    lap_list=[]
    dataSet, bins, patches = plt.hist(dataSet, bins=list(range(0, 100, 10)),color='w')
    print(dataSet)
    for i in range(len(dataSet)):
        x = random.random()
        print("ppppp")
        print(x)
        if (x < 0.5):
            laplace_noise = epsilonSent * math.log(2 * x)
        else:
            laplace_noise = -epsilonSent * (math.log(2 - 2 * x))
        lap_list.append(laplace_noise)
        noiseVal.append(dataSet[i] + laplace_noise)
    print("ffff")
    print(noiseVal)
    print(lap_list)
    dataSet = [i if i > 0 else 0 for i in noiseVal]
    return dataSet
epsilon3List=[0.5,1.0,1.5,2.0,2.5]
for epsilon3 in epsilon3List:
    # 第二部分隐私预算
    start_time = time.time()
    epsilon2=epsilon3
    # 隐私预算集合
    epsilonList = []
    # 输入最大相同桶的数量
    n_max_bucket = 10
    # 窗口的大小
    size_window = 100
    # 当前时刻
    time_location = 100
    # print("1111111111111")
    # 上一时刻加噪数据
    noiseVal_Last = 0
    epsilon_1 = epsilon3
    T_now=1000
    # 总的隐私预算
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
        # print(time_location)
        djim = dj.DGIM(n_max_bucket, size_window, time_location, data_path)
        noise_count, original_count = djim.getNoiseCount()
        # 相关距离阈值
        T0 = 0.05
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
    timeList.append(time.time() - start_time)
    # print(time.time() - start_time)
    # print("ppppp")
    # print(noise_count)
    # print(result)
    val = sum(math.pow(original_count[index]-result[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultMse.append(mse)

########################################DDHP算法########################################
# data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'

# .xlsx转.csv文件
# data_xls = pd.read_excel(data_path, index_col=0)
# data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = '/Users/alasong/Documents/workspace/PP_LDP/师姐实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
epsilonDList=[0.5,1.0,1.5,2.0,2.5]
# 窗口内的每个数据分配的隐私预算
for epsilon_w_D in epsilonDList:
    T_now=1000
    size_window=100
    start_time = time.time()
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
    # x = random.random()
    # print(x)
    # if (x < 0.5):
    #     print("nnnnn")
    #     print(math.log(2 * x))
    #     laplace_noise_t = epsilonSent_D * math.log(2 * x)
    # else:
    #     print(math.log(2 - 2 * x))
    #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * x))
    # print("1111111111111")
    laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
    dataNoise.append(list(dataT[0] + laplace_noise_t)[0])
    # print(dataT[0])
    # print(dataNoise[0])
    # print(dataNoise)
    dataResult.append(dataNoise[0])
    # print("2222222222")
    # print(dataResult)
    # 第一时刻的窗口第二个数据开始对比加噪
    while (i < len(dataT)):
        dataT[i]
        # print(dataNoise[i-1])
        T = dataT[i] - dataNoise[-1]
        # print(i)
        # print("33333333")
        if (abs(T) < 0.05):
            # print("oooooooo")
            dataResult.append(dataNoise[-1])
            dataNoise.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            if (i == len(dataT) - 1):
                wt = len(epslionListDD) - size_window + 1
                epsilon_w_D = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
                # print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                # print("ssssss")
                # print(epsilon_w_D)
                epsilonSent_D = 1 / epsilon_w_D
            else:
                # print("dddddd")
                # print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                epsilonSent_D = 1 / epsilon_w_D
            # y = random.random()
            # print(y)
            # if (y < 0.5):
            #     print("nnnnn")
            #     print(math.log(2 * y))
            #     laplace_noise_t = epsilonSent_D * math.log(2 * y)
            # else:
            #     print(math.log(2 - 2 * y))
            #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
            # print(laplace_noise_t)
            laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
            dataAndNoise = dataT[i] + laplace_noise_t
            dataNoise.append(list(dataAndNoise)[0])
            dataResult.append(list(dataAndNoise)[0])
        i = i + 1
    # print("444444444")
    # print(i)
    # print(dataNoise)
    # print(dataT)
    # print(dataResult)
    # print(epslionListDD)
    # 第二时刻开始到设定时刻对之后相邻数据的对比加噪
    len_T = size_window + T_now - 1
    while (T_window_D < len_T):
        T_window_D = T_window_D + 1
        dataT_T = np.array(list(data[:]['age'])[0:T_window_D])
        # print(dataT_T)
        # print(dataT_T[len(dataT_T) - 1])
        # print("kkkkkk")
        # i=window-1,,,, 2
        T = dataT_T[len(dataT_T) - 1] - dataNoise[-1]
        if (abs(T) < 0.05):
            dataResult.append(dataNoise[-1])
            epslionListDD.append(0)
        else:
            wt = len(epslionListDD) - size_window + 1
            epsilon_w_D = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
            if (epsilon_w_D == 0):
                dataResult.append(dataNoise[-1])
                epslionListDD.append(0)
            else:
                # print(epsilon_w_D)
                epslionListDD.append(epsilon_w_D)
                # print("rrrrr")
                epsilonSent_D = 1 / epsilon_w_D
                # y = random.random()
                # print(y)
                # if (y < 0.5):
                #     print("nnnnn")
                #     print(math.log(2 * y))
                #     laplace_noise_t = epsilonSent_D * math.log(2 * y)
                # else:
                #     print(math.log(2 - 2 * y))
                #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
                # print(laplace_noise_t)
                laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
                dataAndNoise = dataT_T[len(dataT_T) - 1] + laplace_noise_t
                # print(dataAndNoise)
                dataNoise.append(list(dataAndNoise)[0])
                dataResult.append(list(dataAndNoise)[0])
    timeDDList.append(time.time() - start_time)
    # print(time.time() - start_time)
    # print(dataNoise)
    # print(i)
    # print("poririririrrrrrr")
    # print(len(dataT_T))
    # print(dataT_T[T_now-1:T_now+size_window-1])
    # print(len(dataT_T[T_now-1:T_now+size_window-1]))
    # print(dataResult)
    # 当前时刻DDHP均方误差MSE值
    # n_age原始数据频数,n_age_Noise加噪数据频数
    n_age, bins, patches = plt.hist(dataT_T[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    n_age_Noise, bins, patches = plt.hist(dataResult[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    # print(n_age)
    # print(n_age_Noise)
    # print(len(dataResult))
    valD = sum(math.pow(n_age[index] - n_age_Noise[index], 2) for index in range(partition_param))
    mseD = math.sqrt(valD) / partition_param
    resultMseD.append(mseD)
########################################CSLS-LDP算法########################################

# data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'

# # .xlsx转.csv文件
# data_xls = pd.read_excel(data_path, index_col=0)
# data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = '/Users/alasong/Documents/workspace/PP_LDP/师姐实验代码/Car1.csv'

data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
epsilonCList=[0.5,1.0,1.5,2.0,2.5]
# 窗口内的每个数据分配的隐私预算
for epsilon_w_C in epsilonCList:
    T_now=1000
    epsilon4=epsilon_w_C
    size_window = 100
    start_time = time.time()
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
    # x = random.random()
    # print(x)
    # if (x < 0.5):
    #     print("nnnnn")
    #     print(math.log(2 * x))
    #     laplace_noise_t = epsilonSent_D * math.log(2 * x)
    # else:
    #     print(math.log(2 - 2 * x))
    #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * x))
    # print(laplace_noise_t)
    laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
    # print("1111111111111")
    dataNoise.append(list(dataT[0] + laplace_noise_t)[0])
    # print(dataT[0])
    # print(dataNoise[0])
    # print(dataNoise)
    dataResult.append(dataNoise[0])
    # print("2222222222")
    # print(dataResult)
    # 第一时刻的窗口第二个数据开始对比加噪
    while (i < len(dataT)):
        dataT[i]
        # print(dataNoise[i-1])
        T = dataT[i] - dataNoise[-1]
        T0=0.05
        # print(i)
        # print("33333333")
        p = math.exp(epsilon4) / (1 + math.exp(epsilon4))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = abs(T) > T0
        if v > 0:
            if (i == len(dataT) - 1):
                wt = len(epslionListDD) - size_window + 1
                epsilon_w_C = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
                # print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                # print("ssssss")
                # print(epsilon_w_C)
                epsilonSent_D = 1 / epsilon_w_C
            else:
                # print("dddddd")
                # print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                epsilonSent_D = 1 / epsilon_w_C
            # y = random.random()
            # print(y)
            # if (y < 0.5):
            #     print("nnnnn")
            #     print(math.log(2 * y))
            #     laplace_noise_t = epsilonSent_D * math.log(2 * y)
            # else:
            #     print(math.log(2 - 2 * y))
            #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
            # print(laplace_noise_t)
            laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
            dataAndNoise = dataT[i] + laplace_noise_t
            dataNoise.append(list(dataAndNoise)[0])
            dataResult.append(list(dataAndNoise)[0])
        else:
            # print("oooooooo")
            dataResult.append(dataNoise[-1])
            dataNoise.append(dataNoise[-1])
            epslionListDD.append(0)
        i = i + 1
    # print("444444444")
    # print(i)
    # print(dataNoise)
    # print(dataT)
    # print(dataResult)
    # print(epslionListDD)
    # 第二时刻开始到设定时刻对之后相邻数据的对比加噪
    len_T = size_window + T_now - 1
    while (T_window_D < len_T):
        T_window_D = T_window_D + 1
        dataT_T = np.array(list(data[:]['age'])[0:T_window_D])
        # print(dataT_T)
        # print(dataT_T[len(dataT_T) - 1])
        # print("kkkkkk")
        # i=window-1,,,, 2
        T = dataT_T[len(dataT_T) - 1] - dataNoise[-1]
        p = math.exp(epsilon4) / (1 + math.exp(epsilon4))
        if np.random.rand() <= p:
            v = random.choice([0, 1])
        else:
            v = abs(T) > T0
        if v > 0:
            wt = len(epslionListDD) - size_window + 1
            epsilon_w_C = epsilon_D * size_window - sum(epslionListDD[wt:len(epslionListDD)])
            if (epsilon_w_C == 0):
                dataResult.append(dataNoise[-1])
                epslionListDD.append(0)
            else:
                # print(epsilon_w_C)
                epslionListDD.append(epsilon_w_C)
                # print("rrrrr")
                epsilonSent_D = 1 / epsilon_w_C
                # y = random.random()
                # print(y)
                # if (y < 0.5):
                #     print("nnnnn")
                #     print(math.log(2 * y))
                #     laplace_noise_t = epsilonSent_D * math.log(2 * y)
                # else:
                #     print(math.log(2 - 2 * y))
                #     laplace_noise_t = -epsilonSent_D * (math.log(2 - 2 * y))
                # print(laplace_noise_t)
                laplace_noise_t = np.random.laplace(0, epsilonSent_D, 1)
                dataAndNoise = dataT_T[len(dataT_T) - 1] + laplace_noise_t
                # print(dataAndNoise)
                dataNoise.append(list(dataAndNoise)[0])
                dataResult.append(list(dataAndNoise)[0])
        else:
            dataResult.append(dataNoise[-1])
            epslionListDD.append(0)
    timeCCList.append(time.time() - start_time)
    # print(time.time() - start_time)
    # print(dataNoise)
    # print(i)
    # print("poririririrrrrrr")
    # print(len(dataT_T))
    # print(dataT_T[T_now-1:T_now+size_window-1])
    # print(len(dataT_T[T_now-1:T_now+size_window-1]))
    # print(dataResult)
    # 当前时刻DDHP均方误差MSE值
    # n_age原始数据频数,n_age_Noise加噪数据频数
    n_age, bins, patches = plt.hist(dataT_T[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    n_age_Noise, bins, patches = plt.hist(dataResult[T_now-1:T_now+size_window-1], bins=list(range(0, 100, 10)),color='w')
    # print(n_age)
    # print(n_age_Noise)
    # print(len(dataResult))
    valC = sum(math.pow(n_age[index] - n_age_Noise[index], 2) for index in range(partition_param))
    mseC = math.sqrt(valC) / partition_param
    resultMseC.append(mseC)
########################################KMeans算法########################################
dataset,data = createDataSet()
centroids, cluster = kmeans(dataset, 9)
mean_data=[]
same_data=cluster[0]
for j in range(len(cluster)):
    for i in range(len(cluster[j])):
        mean_data.append(centroids[j][0])
print('质心为：%s' % centroids)
print('集群为：%s' % cluster)
print('最终数据为：%s' % mean_data)
# noiseData=countNoise(mean_data,0.1)
kmeansMse=[]
kepsilon=[0.5,1.0,1.5,2.0,2.5]
for i in kepsilon:
    noiseData=dataMNoise(mean_data,i)
    origendata, bins, patches = plt.hist(data, bins=list(range(0, 100, 10)),color='w')
    print(origendata)
    print(noiseData)
    val = sum(math.pow(origendata[index] - noiseData[index], 2) for index in range(9))
    mse = math.sqrt(val) / 9
    kmeansMse.append(mse)
print(kmeansMse)
print("#####################")
print(resultMse)
print(resultMseD)
print(resultMseC)
print("#####################")
print(timeList)
print(timeDDList)
print(timeDDList)
print("9999999")
# #画折线图
x1=[0.5,1.0,1.5,2.0,2.5]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制
plt.plot(x1, resultMseC ,marker='v',markerfacecolor='none',ms=5.5,color='yellow',label='CSLS-LDP',linestyle='-') #绘制
plt.plot(x1, kmeansMse ,marker='v',markerfacecolor='none',ms=5.5,color='green',label='Kmeans',linestyle='-') #绘制

plt.xticks(x1)
plt.xlim(0.5,2.5)     # x轴坐标范围
plt.ylim(0,5)     # y轴坐标范围
plt.xlabel('different ε')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend() #图例
plt.show()