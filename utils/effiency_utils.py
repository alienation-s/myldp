import time

# 用于计时函数
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    return result

# 示例用法
# time_function(run_experiment, file_path, output_dir, sample_fraction=1.0, total_budget=1.0)

from memory_profiler import memory_usage

# 用于监控内存消耗的函数
def memory_function(func, *args, **kwargs):
    mem_usage = memory_usage((func, args, kwargs))
    print(f"Memory usage (in MiB): {max(mem_usage)}")
    return func(*args, **kwargs)

# 示例用法
# memory_function(run_experiment, file_path, output_dir, sample_fraction=1.0, total_budget=1.0)

import psutil

def monitor_resources(func, *args, **kwargs):
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=0.1)
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    result = func(*args, **kwargs)
    
    cpu_after = process.cpu_percent(interval=0.1)
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    
    print(f"CPU usage before: {cpu_before}%")
    print(f"CPU usage after: {cpu_after}%")
    print(f"Memory usage before: {memory_before} MB")
    print(f"Memory usage after: {memory_after} MB")
    
    return result

# 示例用法
# monitor_resources(run_experiment, file_path, output_dir, sample_fraction=1.0, total_budget=1.0)

import matplotlib.pyplot as plt

# 对比不同预算下的时间和内存消耗
# budgets = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
# times = []  # 存储每个预算的时间
# memories = []  # 存储每个预算的内存

# for budget in budgets:
#     result = time_function(monitor_resources, run_experiment, file_path, output_dir, total_budget=budget)
#     times.append(result['dtw_distance'])  # 这里只是示例，可以记录时间
#     memories.append(result['mre'])  # 这里只是示例，可以记录内存使用情况

# # 绘制时间和内存对比图
# plt.plot(budgets, times, label="Time")
# plt.plot(budgets, memories, label="Memory")
# plt.xlabel("Privacy Budget")
# plt.ylabel("Resources")
# plt.legend()
# plt.show()