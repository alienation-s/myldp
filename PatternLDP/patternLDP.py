import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils


# 重要性计算模块
def compute_importance(data):
    # 计算一阶差分，并填充NaN（第一行无法计算差分）
    data['first_difference'] = data['value'].diff().abs()
    data['first_difference'].fillna(0, inplace=True)  # 填充第一行的NaN值为0
    
    # 计算二阶差分，并填充NaN（第一行和第二行无法计算二阶差分）
    data['second_difference'] = data['first_difference'].diff().abs()
    data['second_difference'].fillna(0, inplace=True)  # 填充NaN值为0
    
    # 综合计算重要性：一阶差分和二阶差分的加权
    data['importance'] = data['first_difference'] + data['second_difference']
    
    # 对重要性进行归一化（确保重要性值在0到1之间）
    max_importance = data['importance'].max()
    min_importance = data['importance'].min()
    data['importance'] = (data['importance'] - min_importance) / (max_importance - min_importance)
    
    return data['importance'].values

# 模式感知采样模块
def pattern_aware_sampling(data, delta=0.5):
    sampled_data = []
    
    last_point = data.iloc[0]
    sampled_data.append(last_point)
    
    for i in range(1, len(data)):
        current_point = data.iloc[i]
        # 计算当前点与上一点的表示误差
        error = abs(current_point['value'] - last_point['value'])
        
        if error >= delta:
            sampled_data.append(current_point)
            last_point = current_point
    
    sampled_data = pd.DataFrame(sampled_data)
    return sampled_data

# 隐私预算分配模块
def allocate_privacy_budget(data, total_budget, w):
    num_points = len(data)
    total_importance = data['importance'].sum()  # 总重要性

    # 计算每个数据点的预算权重，根据其重要性进行调整
    data['privacy_budget'] = (data['importance'] / total_importance) * total_budget
    
    # 确保在每个窗口内，隐私预算的总和不超过总预算
    for i in range(num_points):
        window_start = max(0, i - w + 1)  # 滑动窗口的起始位置
        window_end = i + 1  # 当前点作为窗口的一部分
        window_budget = data['privacy_budget'][window_start:window_end].sum()
        
        # 如果窗口内的预算超出了总预算，则进行调整
        if window_budget > total_budget:
            excess_budget = window_budget - total_budget
            adjustment = excess_budget / (window_end - window_start)
            data['privacy_budget'][window_start:window_end] -= adjustment

    return data

# 重要性感知随机化模块
def importance_aware_randomization(data, importance, total_budget, w):
    num_points = len(data)
    perturbed_values = []
    
    # 根据重要性分配预算
    for i, row in data.iterrows():
        gamma = importance[i]  # 获取当前数据点的重要性
        epsilon = total_budget * gamma  # 根据重要性分配预算（重要性越高，预算越多）
        perturbation = np.random.normal(0, epsilon)  # 正态扰动
        perturbed_values.append(row['normalized_value'] + perturbation)  # 使用标准化后的值进行扰动
    # data['restored_value'] = (data['perturbed_value'] * np.std(data['value'])) + np.mean(data['value']) # 反标准化操作
    data['perturbed_value'] = perturbed_values
    return data

# 总接口模块 process
def process(file_path, output_dir, w=160, total_budget=1.0, sample_fraction=1.0, delta=0.5):
    # Step 1: 数据预处理
    data, original_values = data_utils.preprocess_HKHS_data(file_path, sample_fraction)
    
    # Step 2: 计算数据点的重要性
    importance = compute_importance(data)
    
    # Step 3: 分配隐私预算
    data = allocate_privacy_budget(data, total_budget, w)
    
    # Step 4: 模式感知采样
    sampled_data = pattern_aware_sampling(data, delta)
    
    # Step 5: 重要性感知随机化
    perturbed_data = importance_aware_randomization(sampled_data, importance, total_budget, w)
    
    # Step 6: 保存处理后的数据
    output_path = f"{output_dir}/processed_data.csv"
    perturbed_data.to_csv(output_path, index=False)
    
    return perturbed_data

if __name__ == "__main__":
    # 示例调用
    file_path = "data/HKHS.csv"
    output_dir = "results"
    processed_data = process(file_path, output_dir, w=160, total_budget=1.0)