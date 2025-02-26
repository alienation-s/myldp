# 学员：南格格
# 时间：2022/5/2 10:31
# 传统的DGIM算法的桶结构
# 精确统计,返回时间和统计的 1 的数目
import time

def Count_bit_act():
    bit_sum = 0  # 统计1-bit个数
    start_time = time.time()
    with open('E:/系统缓存/桌面/实验代码/数据集区间文件流/8', 'r') as f:
        f.seek(0 if time_location <= window_size else 2 * (time_location - window_size))  # 跳转到窗口大小之前行位置
        for i in range(time_location if time_location <= window_size else window_size):
            temp = f.readline()  # 读取值
            if temp and temp == '1\n':
                # if temp and int(temp.strip('\n'))==1:
                bit_sum += 1
    return bit_sum, time.time() - start_time


# 判断桶是否到期，如果到期则删除桶
def Is_due(time_now):
    if bucket_list and time_now - window_size == bucket_list[0]['t']:  # 最左边的桶的时间戳等于当前时间减去窗口大小，到期了
        del bucket_list[0]


# 桶合并，连续的桶是相似的
def Merge():
    for i in range(len(bucket_list) - 1, max_same_bucket - 1, -1):
        if bucket_list[i]['b'] == bucket_list[i - max_same_bucket]['b']:
            # 存在n_max_bucket个大小相同的桶
            bucket_list[i - max_same_bucket]['b'] += bucket_list[i - max_same_bucket + 1]['b']
            bucket_list[i - max_same_bucket]['t'] = bucket_list[i - max_same_bucket + 1]['t']
            del bucket_list[i - max_same_bucket + 1]


# 估计 1 的数量
def Count_bit():
    bit_sum = 0
    flag_half = 1  #
    start_time = time.time()
    with open('E:/系统缓存/桌面/实验代码/数据集区间文件流/8', 'r') as f:
        for i in range(time_location):
            temp = f.readline()  # 读取文件的值
            if temp:
                Is_due(i + 1)  # 判断是否有桶到期
                if temp == '1\n':
                    # if int(temp.strip('\n'))==1:
                    bucket = {"t": i + 1, "b": 1}  # 桶的结构
                    bucket_list.append(bucket)
                    Merge()  # 合并大小相同的桶
    for i in range(len(bucket_list)):
        # print(bucket_n[i])
        bit_sum += bucket_list[i]['b']
    bit_sum -= bucket_list[0]['b'] / 2
    return bit_sum if bucket_list else 0, time.time() - start_time


# 传统DGIM算法，相同 位置，相同 窗体大小，不同相似桶大小
bit_sum_list=[]
bit_time_list=[]
time_location=501
window_size = 150
max_same_bucket_list=[x for x in range(2,22)]
bit_act_sum,bit_act_time=Count_bit_act()
bucket_length_list_1=[]
for max_same_bucket in max_same_bucket_list:
    bucket_list = []#桶的列表
    bit_sum,bit_time=Count_bit()
    bit_sum_list.append(bit_sum)
    bit_time_list.append(bit_time)
    print(max_same_bucket)
    bucket_length_list_1.append(len(bucket_list))
    for index in bucket_list:
        print(index)

# djim = dj.DGIM(n_max_bucket, size_window,time_location,data_path)
# noise_count,original_count=djim.getNoiseCount()
# def getNoiseCount(n_max_bucket,size_window,time_location):
#     original_count = []
#     noise_count = []
#     for i in range(0,9,1):
#         # data_path = 'E:/系统缓存/桌面/实验代码/数据集区间文件流/'+str(i)
#         print(data_path)
#         djim = dj.DGIM(n_max_bucket, size_window,time_location,data_path)
#         # 当前窗口中1的估计个数,运行时间
#         bit_sum, bit_time = djim.Count_bit()
#         noise_count.append(bit_sum)
#         # 当前窗口中1的精确个数,运行时间
#         bit_act_sum, bit_act_time = djim.Count_bit_act()
#         original_count.append(bit_act_sum)
#         # print("当前窗口中1的估计个数为：%d,运行时间为:%f" % (bit_sum, bit_time))
#         # print(i)
#         # print("当前窗口中1的精确个数为：%d,运行时间为:%f" % (bit_act_sum, bit_act_time))
#         # print("误差值：", abs((bit_act_sum - bit_sum) / bit_act_sum))
#         print("原始频数为：",original_count )
#         print("加噪频数为：",noise_count )
#     return noise_count,original_count
#
# noise_count,original_count=getNoiseCount(n_max_bucket,size_window,time_location)
#