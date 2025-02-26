# 学员：南格格
# 时间：2022/4/18 21:54
import numpy as np
from matplotlib import pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
import  random
epsilon=1
t=5000
w=50
epsilon_list=[]
epsilon_1=epsilon/t/w
epsilon_list.append(epsilon_1)
k=0
index_list=[]
index_list.append(1)
count=1
for i in range(t):
    index=random.choice([0, 1])
    index_list.append(index)
    if(index>0):
        if(count<=w):
            if(k==0):
                epsilon_list.append(epsilon_1)
            else:
                epsilon_list.append(k * epsilon_1)
                k=0
        if(count>w):
            epsilon_list.append(epsilon-sum(epsilon_list[count-w:count-1]))
    else:
        epsilon_list.append(0)
        k=k+1
print(epsilon_list)
print(index_list)
epsilon_list = list(filter(lambda x : x != 0, epsilon_list))
list=[]
for i in range(len(epsilon_list)):
    list.append(epsilon-sum(epsilon_list[0:i-1]))
x=[]
for i in range(len(list)):
    x.append(i)
plt.plot(x, list ,marker="8",markerfacecolor='none',ms=5.5,color='#81aaf1',label='LDP-SWHP',linestyle='-') #绘制
plt.legend() #图例
plt.show()