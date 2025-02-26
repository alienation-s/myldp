import time

class DGIM:
    def __init__(self, n_max_bucket, size_window,time_location,data_path):
        self.bucket_n = []  # 桶的列表
        self.data_path = data_path # 文件路径
        # 输入最大相同桶的数量
        self.n_max_bucket = n_max_bucket
        # 窗口的大小
        self.size_window = size_window
        # 当前时刻
        self.time_location = time_location


    def Count_bit_act(self,data_path):
        bit_sum = 0  # 统计1-bit个数

        start_time = time.time()
        with open(data_path, 'r') as f:

            f.seek(0 if self.time_location <= self.size_window else (self.time_location - self.size_window))  # 跳转到窗口大小之前行位置

            for i in range(self.time_location if self.time_location <= self.size_window else self.size_window):

                temp = f.readline()  # 读取值

                if temp and temp=='1\n':
                    bit_sum += 1


        return bit_sum, time.time() - start_time


    def Is_due(self,time_now):
        if len(self.bucket_n) > 0 and time_now - self.size_window == self.bucket_n[0]['timestamp']:  # 最左边的桶的时间戳等于当前时间减去窗口大小，到期了

            del self.bucket_n[0]


    def Merge(self):
        for i in range(len(self.bucket_n) - 1, self.n_max_bucket - 1, -1):

            if self.bucket_n[i]['bit_sum'] == self.bucket_n[i - self.n_max_bucket]['bit_sum']:
                # 存在n_max_bucket个大小相同的桶

                self.bucket_n[i - self.n_max_bucket]['bit_sum'] += self.bucket_n[i - self.n_max_bucket + 1]['bit_sum']

                self.bucket_n[i - self.n_max_bucket]['timestamp'] = self.bucket_n[i - self.n_max_bucket + 1]['timestamp']

                del self.bucket_n[i - self.n_max_bucket + 1]


    def Count_bit(self,data_path):
        bit_sum = 0

        flag_half = 1

        start_time = time.time()

        with open(data_path, 'r') as f:

            for i in range(self.time_location):

                temp = f.readline()  # 读取文件的值

                if temp:

                    self.Is_due(i + 1)  # 判断是否有桶到期

                    if temp=='1\n':
                        bucket = {"timestamp": i + 1, "bit_sum": 1}  # 桶的结构

                        self.bucket_n.append(bucket)
                        self.Merge()  # 合并大小相同的桶

        if len(self.bucket_n)>0:
            for i in range(len(self.bucket_n)):
                bit_sum += self.bucket_n[i]['bit_sum']
            bit_sum -= self.bucket_n[0]['bit_sum'] / 2
        return bit_sum if len(self.bucket_n) > 0 else 0, time.time() - start_time

    def getNoiseCount(self):
        original_count = []
        noise_count = []
        for i in range(9):
            self.__init__(self.n_max_bucket, self.size_window, self.time_location, self.data_path)
            data_path = self.data_path+str(i)
            # print(data_path)
            # 当前窗口中1的估计个数,运行时间
            bit_sum, bit_time = self.Count_bit(data_path)
            noise_count.append(bit_sum)
            # 当前窗口中1的精确个数,运行时间
            bit_act_sum, bit_act_time = self.Count_bit_act(data_path)
            original_count.append(bit_act_sum)
            # print("原始频数为：",original_count )
            # print("估计频数为：",noise_count )
        return noise_count,original_count