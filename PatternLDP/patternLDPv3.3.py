import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils
import utils.effiency_utils as effiency_utils

# 重要性计算模块 - 使用PID控制
def compute_importance(data):
    # PID控制：第一差分
    data['first_difference'] = data['value'].diff().abs()
    data['first_difference'].fillna(0, inplace=True)
    
    # 第二差分
    data['second_difference'] = data['first_difference'].diff().abs()
    data['second_difference'].fillna(0, inplace=True)
    
    # 计算每个点的重要性
    data['importance'] = data['first_difference'] + data['second_difference']
    
    # 归一化处理
    max_importance = data['importance'].max()
    min_importance = data['importance'].min()
    data['importance'] = (data['importance'] - min_importance) / (max_importance - min_importance)
    
    # PID控制：动态调整重要性
    Kp, Ki, Kd = 0.8, 0.1, 0.1
    data['pid_error'] = Kp * data['first_difference'] + Ki * data['second_difference'] + Kd * data['first_difference'].diff().fillna(0)
    data['importance'] += data['pid_error'].fillna(0)
    
    return data['importance'].values

# 模式感知采样模块 - 使用Piecewise Linear Approximation（PLA）
def pattern_aware_sampling(data, delta=0.5):
    sampled_indices = []  # 存储显著点的索引
    
    last_point = data.iloc[0]
    sampled_indices.append(0)  # 将第一个点的索引加入
    
    for i in range(1, len(data)):
        current_point = data.iloc[i]
        error = abs(current_point['value'] - last_point['value'])
        
        # 只有当误差大于阈值delta时，才将点标记为显著点
        if error >= delta:
            sampled_indices.append(i)  # 记录显著点的索引
            last_point = current_point
    
    return sampled_indices  # 返回显著点的索引列表

# 隐私预算分配模块 - 按照论文中的描述进行调整
def allocate_privacy_budget(data, total_budget, w):
    num_points = len(data)
    total_importance = data['importance'].sum()

    # 计算每个点的隐私预算
    data['privacy_budget'] = (data['importance'] / total_importance) * total_budget
    
    # 窗口内预算分配
    for i in range(num_points):
        window_start = max(0, i - w + 1)
        window_end = i + 1
        window_budget = data['privacy_budget'][window_start:window_end].sum()
        
        # 如果窗口内预算超过总预算，进行调整
        if window_budget > total_budget:
            excess_budget = window_budget - total_budget
            adjustment = excess_budget / (window_end - window_start)
            data['privacy_budget'][window_start:window_end] -= adjustment

    return data

# 重要性感知随机化模块 - 适应性扰动
def importance_aware_randomization(data, importance, total_budget, w, significant_indices):
    num_points = len(significant_indices)  # 显著点的数量
    perturbed_values = []
    
    for i in range(num_points):
        index = significant_indices[i]  # 获取显著点的索引
        value = data.iloc[index]['normalized_value']  # 获取对应的值
        
        gamma = importance[index]  # 获取该显著点的重要性
        epsilon = total_budget * gamma  # 根据重要性分配预算
        
        # 使用适应性扰动（正态扰动）
        perturbation = np.random.normal(0, epsilon)  # 正态扰动
        perturbed_values.append(value + perturbation)  # 扰动后的值
    
    return perturbed_values

# 总接口模块 run_experiment
def run_experiment(file_path, output_dir, w=160, total_budget=1.0, sample_fraction=1.0, delta=0.5, DTW_MRE=True):
    data, _ = data_utils.prerun_experiment_HKHS_data(file_path, sample_fraction)
    normalized_data = data['normalized_value'].values

    # data, original_values = data_utils.preprocess_heartrate_data(
    #     file_path, 
    #     sample_fraction,
    # )
    importance = compute_importance(data)
    
    data = allocate_privacy_budget(data, total_budget, w)
    
    significant_indices = pattern_aware_sampling(data, delta)
    perturbed_values = importance_aware_randomization(data, importance, total_budget, w, significant_indices)
    
    fitted_values = data_utils.generate_piecewise_linear_curve(data['date'], significant_indices, perturbed_values)
    dtw_distance = None
    mre = None
    if(DTW_MRE):
        dtw_distance = data_utils.calculate_fdtw(normalized_data, fitted_values)
        mre = data_utils.calculate_mre(perturbed_values, normalized_data[significant_indices])

    return {
        'normalized_data': normalized_data,
        'importance': importance,
        'significant_indices': significant_indices,
        'perturbed_values': perturbed_values,
        'fitted_values': fitted_values,
        'dtw_distance': dtw_distance,
        'mre': mre,
    }

# 高层次接口函数 compare_experiments
def compare_experiments(file_path, output_dir, target):
    sample_fraction = 1.0
    
    if target == "sample_fraction":
        # 实验 0: 只改变数据量
        sample_fractions = np.arange(0.5, 1.05, 0.05)  # 生成从 0.5 到 1.0，步长为 0.05 的数组
        results = []
        for sample_fraction in sample_fractions:
            result_sample = run_experiment(
                file_path, 
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=1.0, 
                w=160, 
                DTW_MRE=True)
            print(f"DTW for sample fraction {sample_fraction}: {result_sample['dtw_distance']}, MRE for sample fraction {sample_fraction}: {result_sample['mre']}")
            results.append(result_sample)
    elif target == "e":
        es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        results = []
        for e in es:
            result_budget = run_experiment(file_path, output_dir, sample_fraction=sample_fraction, total_budget=e, w=160, delta=0.5, DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        ws = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
        results = []
        for w in ws:
            result_window = run_experiment(file_path, output_dir, sample_fraction=sample_fraction, total_budget=1.0, w=w, delta=0.5, DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)

if __name__ == "__main__":
    file_path = "../data/HKHS.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    compare_experiments(file_path, output_dir, target="e")
    compare_experiments(file_path, output_dir,target="w")
