# 学员：南格格
# 时间：2022/5/10 20:15
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 生成估计值
def generateEstimates(n_data,epsilon,window):
    disturbance_data = []
    for i in range(len(n_data)):
        op = ['0', '1']
        d = random.choice(op)
        random_num = random.random()
        if d == '0':
            dis_data = n_data[i]+random_num*epsilon*window
        else:
            dis_data = n_data[i]-random_num*epsilon*window
        disturbance_data.append(dis_data)
    return disturbance_data

data_path = 'E:/系统缓存/桌面/南宝研二/差分隐私/基础/莫磊/AHPM-SW coding/Car1.xlsx'

# .xlsx转.csv文件
data_xls = pd.read_excel(data_path, index_col=0)
data_xls.to_csv('Car1.csv', encoding='utf-8')
data_path_csv = 'E:/系统缓存/桌面/实验代码/Car1.csv'
partition_param=9
window=100
data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
dataT = np.array(list(data[:]['age'])[0:window])
# 获取到原始数据频数n_data
n_data, bins, patches = plt.hist(dataT, bins=list(range(0, 100, 10)),color='w')
epsilon=0.01
generate_data=generateEstimates(n_data,epsilon,window)
# 加噪
epsilonSent=1/epsilon
x = random.random()
if (x < 0.5):
    print("nnnnn")
    print(math.log(2 * x))
    laplace_noise = epsilonSent * math.log(2 * x)
else:
    print(math.log(2 - 2 * x))
    laplace_noise = -epsilonSent * (math.log(2 - 2 * x))
print(laplace_noise)
noiseVal_data=[]
for i in range(len(generate_data)):
    noise_data = generate_data[i]+laplace_noise
    noiseVal_data.append(noise_data)
noiseVal_data = [i if i > 0 else 0 for i in noiseVal_data]
print(generate_data)
print(noiseVal_data)

# 第二时刻
dataT = np.array(list(data[:]['age'])[1:window+1])
# 获取到原始数据频数n_data
n_data, bins, patches = plt.hist(dataT, bins=list(range(0, 100, 10)),color='w')
epsilon2=0.01
generate_data=generateEstimates(n_data,epsilon,window)
# 相关距离阈值
T0 = 0.04
T = 1 - np.corrcoef(np.array(generate_data), np.array(noiseVal_data))[0][1]
p = math.exp(epsilon2) / (1 + math.exp(epsilon2))
if T >T0:
    # 数据加噪
    # 贪心分组
    print("eeee")
else:
    print("qqqqqqq")