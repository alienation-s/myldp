# 学员：南格格
# 时间：2022/7/11 22:03
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    data_path = 'E:/系统缓存/桌面/论文修改/实验代码/Car1.csv'
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

def dataNoise(dataSet,epsilon):
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

if __name__ == '__main__':
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
    kmeansMse = []
    kepsilon = [0.5, 1.0, 1.5, 2.0, 2.5]
    for i in kepsilon:
        noiseData=dataNoise(mean_data,i)
        origendata, bins, patches = plt.hist(data, bins=list(range(0, 100, 10)),color='w')
        print(origendata)
        print(noiseData)
        val = sum(math.pow(origendata[index] - noiseData[index], 2) for index in range(9))
        mse = math.sqrt(val) / 9
        kmeansMse.append(mse)
print(kmeansMse)
# #画折线图
x1=[0.5,1.0,1.5,2.0,2.5]
plt.plot(x1, kmeansMse ,marker='v',markerfacecolor='none',ms=5.5,color='green',label='Kmeans',linestyle='-') #绘制

plt.xticks(x1)
plt.xlim(0.5,2.5)     # x轴坐标范围
plt.xlabel('different ε')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend() #图例
plt.show()