# 学员：南格格
# 时间：2022/8/11 9:55
# 学员：南格格
# 时间：2022/5/1 20:4
import numpy as np
import math
import random
import noisefirst as nf
import matplotlib.pyplot as plt
import DGIM as dj

def getData(age):
    for k in range(9):
        dataPath = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流LDP/' + str(k)
        file = open(dataPath, 'w').close()
        t = len(age)
        # 建立二维数组，
        graph=[[0 for j in range(10)] for i in range(t)]
        # i为第几时刻
        i=0
        while(i<t):
            if(int(age[i])<10):
                # 有则为1
                graph[i][0]=graph[i][0]+1
            else:
                # 获取到数据的十位数字
                # print(int(age[i]))
                ten_digits,single_digit=map(int,str(int(age[i])))
                # print(ten_digits)
                # 有则为1
                graph[i][ten_digits]=graph[i][ten_digits]+1
            i=i+1
        # print(graph)
        graph=np.array(graph)
        datalist=graph[:,k].tolist() #第k列数据
        # print(graph[:,k].tolist())
        for j in range(0, len(age), 1):
            file = open(dataPath, 'a')
            file.write("{}\n".format(datalist[j]))
            file.close()
    return True


# 输入最大相同桶的数量
n_max_bucket =4
# 窗口的大小
size_window = 150
# 当前时刻
time_location = 150
time_now=500
data_path = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流LDP/'
partition_param=9
# LDP-SHWP下MSE集合
resultMse=[]
print("1111111111111")
data_path_csv = 'E:/系统缓存/桌面/论文修改/实验代码/Car1.csv'
data = np.genfromtxt(data_path_csv, delimiter=',', names=True)
dataT = np.array(list(data[:]['age'])[0:size_window])
epsilonLdpList=[0.5,1.0,1.5,2.0,2.5]
for epsilon in epsilonLdpList:
    # print(dataT)
    data_data=[]
    data_list=[]
    noiseVal_Last = []
    for i in range(len(dataT)):
        epsilonSent = 1/ epsilon
        laplace_noise = np.random.laplace(0, epsilonSent, 1)
        data_list.append(dataT[i]+laplace_noise)
    print("data_ldp")
    count=1
    while(count<=time_now):
        if(count==1):
            data_list = [i if i > 0 else 0 for i in data_list]
            data_list = [i if i < 100 else 99 for i in data_list]
            # print(data_list)
            if (getData(data_list)):
                djim = dj.DGIM(n_max_bucket, size_window, time_location, data_path)
                noise_count, original_count = djim.getNoiseCount()
                noise_count = [i if i > 0 else 0 for i in noise_count]
                noiseVal_Last = noise_count
                # print("oooooo")
                # print(noiseVal_Last)
        else:
            data_now=np.array(list(data[:]['age']))[size_window + count - 2]
            epsilonSent = 1 / epsilon
            laplace_noise = np.random.laplace(0, epsilonSent, 1)
            data_list.append(data_now+laplace_noise)
            # print(data_list)
            data_data=data_list[count-1:size_window + count-1]
            data_data = [i if i > 0 else 0 for i in data_data]
            data_data = [i if i < 100 else 99 for i in data_data]
            # print("ppppp")
            # print(data_data)
            # print(len(data_data))
            # print(data_data[size_window-1])
            if (getData(data_data)):
                djim = dj.DGIM(n_max_bucket, size_window, time_location, data_path)
                noise_count, original_count = djim.getNoiseCount()
                noise_count = [i if i > 0 else 0 for i in noise_count]
                # print(noise_count)
                # print(noiseVal_Last)
                # 相关距离阈值
                T0 = 0.04
                # 相关距离计算
                # print(np.array(noise_count).shape)
                # print(np.array(noiseVal_Last).shape)
                T = 1 - np.corrcoef(np.array(noise_count), np.array(noiseVal_Last))[0][1]
                # print("eeeeeeeee")
                # print(np.corrcoef(np.array(noise_count), np.array(noiseVal_Last)))
                noiseVal_Last = noise_count if T > T0 else noiseVal_Last
        count=count+1
    # print(noiseVal_Last)
    # #当前时刻LDP均方误差MSE值
    val = sum(math.pow(original_count[index]-noiseVal_Last[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultMse.append(mse)
    print(resultMse)
###########################################LDP-SWHP##################################################
epsilon3List=[0.5,1.0,1.5,2.0,2.5]
resultDMse=[]
data_path = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流/'
for epsilon3 in epsilon3List:
    # 第二部分隐私预算
    epsilon2=epsilon3
    # 隐私预算集合
    epsilonList = []
    time_location=150
    # print("1111111111111")
    # 上一时刻加噪数据
    noiseVal_Last = 0
    epsilon_1 = epsilon3
    T_now=time_now+size_window-1
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
    # print(time.time() - start_time)
    # print("ppppp")
    # print(noise_count)
    # print(result)
    val = sum(math.pow(original_count[index]-result[index],2) for index in range(partition_param))
    mse=math.sqrt(val)/partition_param
    resultDMse.append(mse)
print(resultDMse)
#画直方图
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker

c = (
    Bar()
    .add_xaxis(list(range(0, 90, 10)))
    .add_yaxis("年龄", noise_count, category_gap=0, color=Faker.rand_color())
    .set_global_opts(title_opts=opts.TitleOpts(title="LDP+DGIM直方图"))
    .render("LDP+DGIM_histogram.html")
)

# #画折线图
x1=[0.5,1.0,1.5,2.0,2.5]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP',linestyle='-') #绘制
plt.plot(x1, resultDMse ,marker='o',markerfacecolor='none',ms=5.5,color='red',label='LDP-SWHP',linestyle='-') #绘制
plt.xticks(x1)
plt.xlim(0.5,2.5)     # x轴坐标范围
plt.ylim(0,10)     # y轴坐标范围
plt.xlabel('different ε')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend() #图例
plt.show()
