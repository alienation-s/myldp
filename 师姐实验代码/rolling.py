# 学员：南格格
# 时间：2022/4/12 10:11
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import blackman
import matplotlib.pyplot as plt
import numpy as np

#x=np.arange(1,7,1)  # x坐标
x= [7, 23,36, 49,62, 75,88,100] # x坐标
#隐私预算为0.01时
#y= [0.5, 1.0,1.5, 2.0,2.5] # y坐标刻度
#隐私预算为0.1时
#y= [0.4, 0.6,0.8, 1.0,1.2,1.4] # y坐标刻度
#隐私预算为1时
y= [0.5, 1.0,1.5, 2.0,2.5] # y坐标刻度

x1=[20,40,60,80,100]

#隐私预算为0.01时
# y1=[2.5 , 2.75 , 2.85 ,2.85, 2.74 , 2.76,2.78,2.80]
# y2=[0.25 , 0.27 , 0.26 ,0.265, 0.266 , 0.25,0.225,0.245]
# y3=[0.25 , 0.8 , 0.9 ,1.05, 1.15 , 1.35,1.45,1.40]

#隐私预算为0.1时
# y1=[0.7 , 0.8 , 0.94 ,0.91, 0.9 , 0.89,0.9,0.88]
# y2=[0.4 , 0.45 , 0.5 ,0.46, 0.45 , 0.44,0.42,0.39]
# y3=[0.8 , 1.2 , 1.3 ,1.35, 1.36 , 1.37,1.38,1.39]

#隐私预算为1时
y1=[0.505 , 0.425 , 0.5 ,0.42, 0.419 , 0.418,0.418,0.419]
y2=[0.54 , 1.75 , 2.115,2.32, 2.55 , 2.55,2.3,2.2]
y3=[0.58 , 1.80 , 2.32 ,2.55,2.65 , 2.62,2.40,2.3]

#plt.plot(x, y1 ,marker='o',markerfacecolor='white' ,ms=5,color='black',label='Laplace',linestyle='dashdot') #绘制y1 lpls
plt.plot(x, y3 ,marker='v',markerfacecolor='none',ms=5.5,color='black',label='Geometric',linestyle='-') #绘制y2 简单几何
plt.plot(x, y2 ,marker='s',markerfacecolor='none',ms=5,color='black',label='GM-NoiseFirst',linestyle=':') #绘制y2 几何加指数
plt.plot(x, y1 ,marker='o',markerfacecolor='none' ,ms=5,color='black',label='Laplace',linestyle='dashdot') #绘制y1 拉普拉斯机制


plt.xticks(x1)          # x轴的刻度
plt.yticks(y)           # y轴的刻度
plt.xlim(0,105)     # x轴坐标范围
plt.ylim(0.25,2.75)    # y轴坐标范围（0.3）（）（0.25，2.75）
plt.xlabel('Range size')   # x轴标注
plt.ylabel('Mean Square Error(e+02)')   # y轴标注
plt.title('ε=1',loc="left")
plt.legend(fontsize=7.5)           #图例
plt.show()
