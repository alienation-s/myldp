# 学员：南格格
# 时间：2022/4/11 21:42
# 滑动窗口大小固定，区间大小固定，固定隐私预算下的加噪后的直方图
import opendp.smartnoise.core as sn
import numpy as np
import math
import random

import noisefirst as nf
import matplotlib.pyplot as plt


# 数据集路径
data_path = 'E:/系统缓存/桌面/data.csv'
# data_path = os.path.join('.', 'data', os.path.basename(path))
# 字段
var_names = ["age"]
# 隐私预算
epsilon = 0.1
# 字段范围+区间数量
age_edges = list(range(10, 110, 10))

data = np.genfromtxt(data_path, delimiter=',', names=True)
age = list(data[:]['age'])
partition_param=9
# 窗口大小window
window=200
a=np.array(age)
shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
strides = a.strides + (a.strides[-1],)
re=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
print(len(re))
# t为当前时刻
t=1
# dataT当前t时刻窗口数据
dataT=re[t]
print(dataT)

with sn.Analysis(protect_floating_point=False) as analysis:
    # data = sn.Dataset(path=data_path, column_names=var_names)
    nsize = 1000
    age_prep = sn.histogram(sn.to_int(dataT, lower=0, upper=110),
                            edges=age_edges, null_value=-1)
    age_histogram = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})

analysis.release()

n_age, bins, patches = plt.hist(dataT, bins=list(range(10, 110, 10)), color='w',
                                alpha=0.7, rwidth=0.85)
# 原始直方图数据频数
print(n_age)
# 加噪直方图数据频数
noiseVal = [i if i > 0 else 0 for i in age_histogram.value]
# 相关距离阈值
T0 = 0.04
# 相关距离计算
T = 1 - np.corrcoef(np.array(n_age), np.array(noiseVal))[0][1]
# 相关距离对比：大于阈值=》对当前数据加噪发布’小于阈值=》发布上一时刻加噪数据
# 对T进行概率扰动
#概率扰动p
p = math.exp(epsilon) / (1 + math.exp(epsilon))
if np.random.rand() <= p:
    v = random.choice(["True", "False"])
else:
    v = T > T0
if v:
    # 数据加噪
    # 分组TODO
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=data_path, column_names=var_names)
        nsize = 1000
        age_prep = sn.histogram(sn.to_int(dataT, lower=0, upper=100),
                                edges=age_edges, null_value=-1)
        age_histogram_T = sn.laplace_mechanism(age_prep, privacy_usage={"epsilon": epsilon, "delta": .000001})
    analysis.release()
    # 输出当前时刻加噪数据
    noiseVal_T = [i if i > 0 else 0 for i in age_histogram_T.value]

else:
    # 输出上一时刻加噪数据
    noiseVal_T=noiseVal
noisefirst = nf.NoiseFirst(noiseVal_T, epsilon)
optk = noisefirst.findOptK(noiseVal_T)
result = noisefirst.getResultHist(noiseVal_T, optk)

print(result)

#画直方图
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker

c = (
    Bar()
    .add_xaxis(list(range(10, 110, 10)))
    .add_yaxis("年龄", result, category_gap=0, color=Faker.rand_color())
    .set_global_opts(title_opts=opts.TitleOpts(title="LDP-SWHP直方图"))
    .render("bar_histogram.html")
)
