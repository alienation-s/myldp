# 学员：南格格
# 时间：2022/11/27 15:28
import math

from matplotlib import pyplot as plt

L=1000
W_list=[50,100,150,200,250]
r=11
y1=[]
y2=[]
for w in W_list:
    y1.append(L*math.log(w)/1000)
    y2.append((L/(r-1))*math.pow(math.log(w),2)/1000)
plt.plot(W_list, y1 ,marker="o", linewidth="3", label='others') #绘制
plt.plot(W_list, y2 ,marker="x", linewidth="3",markerfacecolor='none',ms=5.5,label='EOHP',linestyle='-') #绘制

# plt.plot(x, y_3, marker="o", linewidth="3", label='k-means-dp')
# plt.plot(x, y_4, marker="x", linewidth="3", label='EOHP')
plt.xlabel("Different w",fontdict={"family": "Times New Roman", "size": 16})
plt.ylabel("Storage Cost",fontdict={"family": "Times New Roman", "size": 16})

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
tit='×'+'10$^{4}$'
plt.legend(loc="upper left",fontsize=10)
plt.title(tit,loc='left',fontsize=16)
plt.show()
plt.legend() #图例
plt.show()