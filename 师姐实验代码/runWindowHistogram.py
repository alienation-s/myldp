# 学员：南格格
# 时间：2022/4/18 20:18
# 学员：南格格
# 时间：2022/4/11 21:42
#隐私预算大小固定，区间大小固定，不同滑动窗口下的均方误差值MSE对比

import opendp.smartnoise.core as sn
import numpy as np
import math
import random
import noisefirst as nf
import matplotlib.pyplot as plt


# 数据集路径
import pandas as pd

data_path = 'E:/系统缓存/桌面/data.csv'
# 字段
var_names = ["age"]
# 字段范围+区间数量
age_edges = list(range(10, 110, 10))
data = np.genfromtxt(data_path, delimiter=',', names=True)
age = list(data[:]['age'])
partition_param=9
# 不同隐私预算下MSE集合
resultMse=[]
epsilon=0.5
# 窗口大小window
window=0
windowList=[50,100,150,200,250]
for window in windowList:
    a=np.array(age)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    re=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    print(len(re))
    # t为当前时刻
    t=1
    # dataT当前t时刻窗口数据
    dataT=re[t]
    print(dataT)
    with sn.Analysis(protect_floating_point=False) as analysis:
        # data = sn.Dataset(path=data_path, column_names=var_names)
        nsize = 1000
        age_prep = sn.histogram(sn.to_int(dataT, lower=0, upper=110),
                                edges=age_edges, null_value=-1)
        age_histogram = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})

    analysis.release()
    n_age, bins, patches = plt.hist(dataT, bins=list(range(10, 110, 10)),color='w')
    # 原始直方图数据频数
    print(n_age)
    # 加噪直方图数据频数
    noiseVal = [i if i > 0 else 0 for i in age_histogram.value]
    # 相关距离阈值
    T0 = 0.04
    # 相关距离计算
    T = 1 - np.corrcoef(np.array(n_age), np.array(noiseVal))[0][1]
    # 相关距离对比：大于阈值=》对当前数据加噪发布’小于阈值=》发布上一时刻加噪数据
    # 对T进行概率扰动
    #概率扰动p
    p = math.exp(epsilon) / (1 + math.exp(epsilon))
    if np.random.rand() <= p:
        v = random.choice(["True", "False"])
    else:
        v = T > T0
    if v:
        # 数据加噪
        # 分组TODO
        with sn.Analysis(protect_floating_point=False) as analysis:
            data = sn.Dataset(path=data_path, column_names=var_names)
            nsize = 1000
            age_prep = sn.histogram(sn.to_int(dataT, lower=0, upper=100),
                                    edges=age_edges, null_value=-1)
            age_histogram_T = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})
        analysis.release()
        # 输出当前时刻加噪数据
        noiseVal_T = [i if i > 0 else 0 for i in age_histogram_T.value]
    else:
        # 输出上一时刻加噪数据
        noiseVal_T=noiseVal
    noisefirst = nf.NoiseFirst(noiseVal_T, epsilon)
    optk = noisefirst.findOptK(noiseVal_T)
    result = noisefirst.getResultHist(noiseVal_T, optk)
    #分组后均方误差MSE值
    val = sum(math.pow(n_age[index]-result[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultMse.append(mse)
print(resultMse)

#画折线图
x1=[50,100,150,200,250]
xk=[50,100,150,200,250]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制

plt.xticks(xk)
plt.xlim(50,250)     # x轴坐标范围
plt.xlabel('different window')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different window')
plt.legend()           #图例
plt.show()
