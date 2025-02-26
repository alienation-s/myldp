import os
import random
import noisefirst as nf
import matplotlib.pyplot as plt
import numpy as np

# 数据集路径  E:\系统缓存\桌面\论文修改\实验代码\Car1.csv
data_path = 'E:/系统缓存/桌面/论文修改/实验代码/Car1.csv'
data = np.genfromtxt(data_path, delimiter=',', names=True)
age = list(data[:]['age'])

def getData(age,k):
    t = len(age)
    # 建立二维数组，
    graph=[[0 for j in range(10)] for i in range(t)]
    # i为第几时刻
    i=0
    while(i<t):
        if(age[i]<10):
            # 有则为1
            graph[i][0]=graph[i][0]+1
        else:
            # 获取到数据的十位数字
            ten_digits,single_digit=map(int,str(int(age[i])))
            print(ten_digits)
            # 有则为1
            graph[i][ten_digits]=graph[i][ten_digits]+1
        i=i+1
    print(graph)
    graph=np.array(graph)
    datalist=graph[:,k].tolist() #第k列数据
    print(graph[:,k].tolist())
    for j in range(0, len(age), 1):
        dataPath = 'E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流/' + str(k)
        file = open(dataPath, 'a')
        file.write("{}\n".format(datalist[j]))
        file.close()
    return True




# # 第几列
#k=8
datalist=getData(age,8)
print(datalist)
# # 将数据写入文件  E:\系统缓存\桌面\论文修改\实验代码\数据集区间文件流
# for j in range(0,len(age),1):
#     dataPath='E:/系统缓存/桌面/论文修改/实验代码/数据集区间文件流/'+str(k)
#     file=open(dataPath,'a')
#     file.write("{}\n".format(datalist[j]))
#     file.close()