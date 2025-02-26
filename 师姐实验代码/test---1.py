# 学员：南格格
# 时间：2022/4/18 21:54
import numpy as np
from matplotlib import pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
import random

epsilon = 1
t = 5000
w = 50
epsilon_list = []
epsilon_1 = epsilon / t / w
epsilon_list.append(epsilon_1)
k = 0
index_list = []
index_list.append(1)
count = 1
for i in range(t):
    index = random.choice([0, 1])
    index_list.append(index)
    if (index > 0):
        if (count <= w):
            if (k == 0):
                epsilon_list.append(epsilon_1)
            else:
                epsilon_list.append(k * epsilon_1)
                k = 0
        if (count > w):
            epsilon_list.append(epsilon - sum(epsilon_list[count - w:count - 1]))
    else:
        epsilon_list.append(0)
        k = k + 1
print(epsilon_list)
print(index_list)
epsilon_list = list(filter(lambda x: x != 0, epsilon_list))
list = []
for i in range(len(epsilon_list)):
    list.append(epsilon - sum(epsilon_list[0:i - 1]))

#####################二分###############################
epsilon_list_1 = []
epsilon_list_2 = []
epsilon = 1
epsilon_1 = epsilon / 2
epsilon_list_1.append(epsilon_1)
for i in range(1, 100):
    epsilon_list_1.append(epsilon_list_1[len(epsilon_list_1) - 1] / 2)
    epsilon_list_2.append(epsilon / (i * (i + 1)))
epsilon_list_2.append(epsilon/(100*101))
print(i)
list_1 = []
list_2 = []
for i in range(len(epsilon_list_1)):
    list_1.append(epsilon - sum(epsilon_list_1[0:i]))
    list_2.append(epsilon - sum(epsilon_list_2[0:i]))
x = []
a=[1,2,3]
print(a[0:2])
list_1.pop(0)
list_2.pop(0)
for i in range(len(epsilon_list_1)):
    x.append(i)
print(epsilon_list_1)
print(epsilon_list_2)
plt.plot(x, epsilon_list_1, marker="8", markerfacecolor='none', ms=5.5, color='#81aaf1', label='2', linestyle='-')  # 绘制
plt.plot(x, epsilon_list_2, marker="8", markerfacecolor='none', ms=5.5, color='red', label='j', linestyle='-')  # 绘制
plt.legend()  # 图例
plt.show()
