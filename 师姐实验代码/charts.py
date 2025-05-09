# 学员：南格格
# 时间：2022/5/15 16:4
from matplotlib import pyplot as plt

x1=[0.5,1.0,1.5,2.0,2.5]
resultMse=[]
resultMseD=[]
resultMseC=[]
plt.plot(x1, resultMse ,marker='o',markerfacecolor='none',ms=5.5,color='blue',label='LDP-SWHP',linestyle='-') #绘制
plt.plot(x1, resultMseD ,marker='v',markerfacecolor='none',ms=5.5,color='red',label='DDHP',linestyle='-') #绘制
plt.plot(x1, resultMseC ,marker='v',markerfacecolor='none',ms=5.5,color='yellow',label='CSLS-LDP',linestyle='-') #绘制

plt.xticks(x1)
plt.xlim(0.5,2.5)     # x轴坐标范围
plt.ylim(0,5)     # y轴坐标范围
plt.xlabel('different ε')   # x轴标注
plt.ylabel('Mean Square Error')   # y轴标注
plt.title('MSE under different algorithm')
plt.legend() #图例
plt.show()