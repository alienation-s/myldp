import numpy as np
print(list(range(0, 100, 10)))
# data_list=[1,2,3,2,1,4]
# print(data_list)
# laplace_ = np.random.laplace(0, 10, 6)
# laplace_r = np.random.laplace(0, 5, 6)
# print(laplace_)
# print(laplace_r)
# print(laplace_+data_list)
# laplace_noise = np.random.laplace(0, 1, 1)
# print(laplace_noise)
# window=3
# epsilon=0.1
# epsilon_w=epsilon/window
# epsilonSent=1/epsilon_w
# laplace_noise_t = np.random.laplace(0, epsilonSent, 1)
# print(laplace_noise_t)
# data=[2,2,4]
# print(data[0:3])
# print(sum(data[1:3]))
# function [u] = laplace_noise1(epsilon)
# %function to generate laplace noise//函数生成拉普拉斯噪声
# % u = output laplace noise//u表示输出拉普拉斯噪声
# sigma = 1/epsilon;
# rand(1,1)产生一行一列的数据，find(x<1/2)找到小于0.5的数
# x = rand(1,1);% j = find(x < 1/2);% k = find(x >= 1/2);
# if(x<1/2)
#    %u =  *log(1+2*(x-0.5));
#    u = sigma*log(2*x);
# else
#    u = -(sigma)*log(2-2*x); %
# end
# data=[1,2,3]
# su=sum(data[0:3])
# print(su)
