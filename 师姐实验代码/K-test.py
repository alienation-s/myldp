# 学员：南格格
# 时间：2022/7/11 21:45
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import pandas as pd
n_samples=1000
x,y=datasets.make_blobs(n_samples=1000,n_features=4,random_state=8)
print(x)
data_path = 'E:/系统缓存/桌面/data.csv'
df = pd.read_csv(data_path)
age = list(df[:]['age'])
data = np.array(age)
result = []
for i in range(len(data)):
    datalist = []
    datalist.append(data[i])
    result.append(datalist)
result=pd.DataFrame(StandardScaler().fit_transform(result))
sil_score=[]
inertia=[]
for k in range(2,9):
    kmeans=KMeans(n_clusters=k,random_state=0).fit(result)
    print(x)
    sil_score.append(silhouette_score(x,kmeans.labels_))
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(2,9),inertia,'o-')
plt.xlabel('k')
plt.show()

