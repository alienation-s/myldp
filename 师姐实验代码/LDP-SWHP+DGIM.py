# 学员：南格格
# 时间：2022/5/1 20:4
import numpy as np
import math
import random
import noisefirst as nf
import matplotlib.pyplot as plt
import DGIM as dj
# 输入最大相同桶的数量
n_max_bucket =4
# 窗口的大小
size_window = 150
# 当前时刻
time_location = 200
data_path = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流/'
partition_param=9
# LDP-SHWP下MSE集合
resultMse=[]
#DDHP下的MSE集合
resultMseD=[]
# 第三部分隐私预算
# epsilon3=0.5
# 第二部分隐私预算
epsilon2=0.5
# 隐私预算集合
epsilonList = []
print("1111111111111")
# 上一时刻加噪数据
noiseVal_Last = 0
# epsilon_1 = epsilon3
# 总的隐私预算
T_now=600
# epsilon = epsilon_1 * T_now=1
epsilon =1
epsilon3=epsilon/T_now
epsilonList.append(epsilon3)
epsilonSent=1/epsilon3
laplace_noise = np.random.laplace(0, epsilonSent , partition_param)
# 第1时刻加噪直方图数据频数
print(len(laplace_noise))
djim = dj.DGIM(n_max_bucket, size_window,time_location,data_path)
noise_count,original_count=djim.getNoiseCount()
noiseVal_T1 = noise_count+laplace_noise
noiseVal_T1 = [i if i > 0 else 0 for i in noiseVal_T1]
print("2222222")
print(noiseVal_T1)
noiseVal_Last=noiseVal_T1
print(time_location)
while (time_location < T_now):
    time_location = time_location + 1
    print("3333333")
    print(time_location)
    djim = dj.DGIM(n_max_bucket, size_window, time_location, data_path)
    noise_count, original_count = djim.getNoiseCount()
    # 相关距离阈值
    T0 = 0.04
    # 相关距离计算
    print(np.array(noise_count).shape)
    print(np.array(noiseVal_Last).shape)
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
        print(epsilon)
        print(sum(epsilonList[0:len(epsilonList)]))
        # epsilon3=(epsilon-sum(epsilonList[0:len(epsilonList)]))/(T_now-(time_location-1))
        epsilon3=(epsilon-sum(epsilonList[0:len(epsilonList)]))/T_now
        # epsilon3=epsilon-sum(epsilonList[len(epsilonList)-size_window:len(epsilonList)])
        print("sssssss")
        print(epsilon3)
        epsilonList.append(epsilon3)
        if(epsilon3==0):
            result=noiseVal_Last
        else:
            print(epsilon3)
            epsilonSent = 1 / epsilon3
            laplace_noise = np.random.laplace(0, epsilonSent, partition_param)
            noiseVal_T=noise_count+laplace_noise
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
print("ppppp")
print(noise_count)
print(result)
val = sum(math.pow(original_count[index]-result[index],2) for index in range(partition_param))
mse=math.sqrt(val)/partition_param
resultMse.append(mse)
print("#####################")
print(original_count)
print(noise_count)
print(result)
print(mse)
print("#####################")
print(epsilonList)
print(len(epsilonList))
print(T_now)
print("9999999")
#画直方图
# from pyecharts import options as opts
# from pyecharts.charts import Bar
# from pyecharts.faker import Faker
#
# c = (
#     Bar()
#     .add_xaxis(list(range(0, 90, 10)))
#     .add_yaxis("年龄", result, category_gap=0, color=Faker.rand_color())
#     .set_global_opts(title_opts=opts.TitleOpts(title="LDP-SWHP+DGIM直方图"))
#     .render("LDP-SWHP+DGIM_histogram.html")
# )

x=[]
for i in range(len(epsilonList)):
    x.append(i)
list_2=[]
for i in range(len(epsilonList)):
    list_2.append(epsilon-sum(epsilonList[0:i]))
epsilon_list_1=[]
for i in range(1,len(list_2)):
    epsilon_list_1.append(epsilon/(i*(i+1)))
list=[]
for i in range(len(list_2)):
    list.append(epsilon-sum(epsilon_list_1[0:i]))
x=[]
# list_2.pop(0)
print(list)
print(epsilonList)
print("list")
# list.pop(0)
for i in range(len(list_2)):
    x.append(i)
print(len(list))
print(len(epsilonList))
# plt.plot(x, list_2 ,marker="8",markerfacecolor='none',ms=5.5,color='#81aaf1',label='LDP-SWHP',linestyle='-') #绘制
plt.plot(x, epsilonList ,marker="8",markerfacecolor='none',ms=5.5,color='red',label='dd',linestyle='-') #绘制
plt.legend() #图例
plt.show()
