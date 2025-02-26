import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils
import utils.effiency_utils as effiency_utils

# =========================================================================
# Step 3: 自适应预算分配（根据LBD方案）
# =========================================================================
def adaptive_w_event_budget_allocation_lbd(slopes, fluctuation_rates, total_budget, w, min_budget=1e-5):
    """
    根据LBD方案实现自适应预算分配，确保w个显著点的总预算不超过total_budget
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    
    for i in range(num_points):
        # 获取当前点的滑动窗口范围
        window_start = max(0, i - w + 1)
        window_end = i
        # 已使用的预算
        used_budget = np.sum(allocated_budgets[window_start:window_end])
        remaining_budget = total_budget - used_budget
        
        if remaining_budget <= 0:
            allocated_budgets[i] = min_budget
            continue
        
        # 计算当前点的预算
        if slopes[i] >= 0:
            pk = 1 - np.exp(-slopes[i])
        else:
            pk = 1 - np.exp(slopes[i])
        
        pgamma = 1 - np.exp(-fluctuation_rates[i])
        
        if slopes[i] >= 0:
            pky = 1 - np.exp(-1 / (slopes[i] * fluctuation_rates[i] + 1e-8))
        else:
            pky = 1 - np.exp(-slopes[i] / (fluctuation_rates[i] + 1e-8))
        
        p = 1 - np.exp(-((pk + pgamma) / (pky + 1e-8)))
        epsilon_i = p * remaining_budget
        epsilon_i = max(epsilon_i, min_budget)
        allocated_budgets[i] = epsilon_i

    return allocated_budgets

# Step 4: SW扰动机制
def sw_perturbation_w_event_lbd(values, budgets, min_budget=0.01):
    """
    对显著点应用SW机制，确保按照LBD方案进行扰动
    """
    perturbed_values = []
    assert len(values) == len(budgets), "Values and budgets arrays have different lengths"
    for value, epsilon in zip(values, budgets):
        epsilon = max(epsilon, min_budget)
        denominator = 2 * np.exp(epsilon) * (np.exp(epsilon) - 1 - epsilon)
        if denominator <= 1e-10:
            perturbed_value = value
        else:
            b = (epsilon * np.exp(epsilon) - np.exp(epsilon) + 1) / denominator
            perturb_prob = np.exp(epsilon) / (2 * b * np.exp(epsilon) + 1)
            if np.random.random() <= perturb_prob:
                perturbed_value = value
            else:
                perturbed_value = value + np.random.laplace(scale=b)
        perturbed_values.append(perturbed_value)
    return perturbed_values

# =========================================================================
# 主实验接口：控制不同的实验变量
def run_experiment_lbd(file_path, output_dir, sample_fraction=1.0, total_budget=1.0, w=160, delta=0.5, kp=0.8, ks=0.1, kd=0.1, DTW_MRE=True):
    """
    统一接口：控制实验中的各个变量，如数据量、隐私预算、窗口大小等，返回实验结果。
    """
    current_date = datetime.now().strftime('%Y%m%d%H%M')
    # Step 1: 数据预处理
    data, original_values = data_utils.preprocess_HKHS_data(file_path, sample_fraction)
    normalized_data = data['sample_normalized_value'].values
    original_data = data['normalized_value'].values

    # Step 3: 自适应预算分配（LBD方案）
    slopes = np.gradient(normalized_data)
    fluctuation_rates = np.abs(slopes)
    allocated_budgets = adaptive_w_event_budget_allocation_lbd(slopes, fluctuation_rates, total_budget=total_budget, w=w)

    # Step 4: SW扰动机制
    perturbed_values = sw_perturbation_w_event_lbd(normalized_data, allocated_budgets)

    # 计算并返回所需的度量（如 DTW 和 MRE）
    dtw_distance = None
    mre = None
    if(DTW_MRE==True):
        fitted_values = data_utils.generate_piecewise_linear_curve(data['date'], np.arange(len(normalized_data)), perturbed_values)
        dtw_distance = data_utils.calculate_fdtw(normalized_data, fitted_values)
        mre = data_utils.calculate_mre(perturbed_values, normalized_data)

    return {
        'normalized_data': normalized_data,
        'perturbed_values': perturbed_values,
        'dtw_distance': dtw_distance,
        'mre': mre
    }

# =========================================================================
# 统一接口调用
def compare_experiments(file_path, output_dir, target):
    """
    调用统一接口进行不同变量的实验，返回并对比结果。
    """
    sample_fraction = 0.8
    if target == "e":
        # 实验 1: 只改变隐私预算
        es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        results = []
        for e in es:
            result_budget = run_experiment_lbd(file_path, output_dir, sample_fraction=sample_fraction, total_budget=e, w=160, DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        # 实验 2: 只改变窗口大小
        ws = [80,100,120,140,160,180,200,220,240,260]
        results = []
        for w in ws:
            result_window = run_experiment_lbd(file_path, output_dir, sample_fraction=sample_fraction, total_budget=1.0, w=w, DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)

if __name__ == "__main__":
    file_path = '../data/HKHS.csv'  # 输入数据路径
    output_dir = 'results'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    compare_experiments(file_path, output_dir, target="e")
    compare_experiments(file_path, output_dir, target="w")