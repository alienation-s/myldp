import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils
import utils.effiency_utils as effiency_utils
# 需要在文件开头添加以下导入
from concurrent.futures import ThreadPoolExecutor

def calculate_slope(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b
def calculate_angle(k1, k2):

    return abs((k2 - k1) / (1 + k1 * k2 + 1e-8))
def adaptive_angle_threshold(k, gamma, kp=0.8, ks=0.1, kd=0.1):
    denominator = max(abs(k) + gamma + 1e-8, 1e-3)
    exponent = -1 / denominator
    # 修正条件判断，当分母极小时触发
    if denominator < 1e-3:
        lambda_val = 1.0
    else:
        lambda_val = 1 - np.exp(exponent)
    alpha = lambda_val * np.pi / 2
    alpha = np.clip(alpha, 0, np.pi/2 - 1e-8)
    tan_alpha = np.tan(alpha)
    return tan_alpha

def calculate_error(fitted_value, actual_value):
    return abs(fitted_value - actual_value)

def calculate_fluctuation_rate(errors, current_idx, kp=0.8, ks=0.1, kd=0.1, pi=5):
    if current_idx == 0:  # 第一个点没有历史数据
        return 0.0

    e_ti = errors[current_idx]  # 当前点误差
    e_ti_1 = errors[current_idx - 1] if current_idx - 1 >= 0 else 0  # 前一个点误差

    # 计算比例控制项
    proportional_term = kp * e_ti

    # 计算积分控制项
    integral_term = 0
    start_idx = max(0, current_idx - pi)  # 确保窗口不越界
    for n in range(start_idx, current_idx + 1):
        integral_term += errors[n]
    integral_term = (ks / pi) * sum(errors[start_idx:current_idx+1])
    # integral_term = (ks / pi) * (sum(errors[start_idx:current_idx+1]) / pi)

    # 计算微分控制项
    differential_term = 0
    if current_idx > 0:
        differential_term = kd * (e_ti - e_ti_1)
        # differential_term = kd * (e_ti - e_ti_1) / (t_i - t_i_1)

    # 总波动率
    gamma = proportional_term + integral_term + differential_term
    return gamma

def remarkable_point_sampling(data, original_data, kp=0.8, ks=0.1, kd=0.1, pi=5):
    n = len(data)
    points = [0]  # 初始化为包含第一个点
    i = 0
    errors = np.zeros(n)
    
    while i < n - 1:
        start = i
        for j in range(i + 1, n):
            segment = np.array([[x, data[x]] for x in range(start, j + 1)])
            k, b = calculate_slope(segment)
            fitted_value = k * j + b
            errors[j] = calculate_error(fitted_value, data[j])
            
            if j + 1 < n:
                k_next = (data[j + 1] - data[j]) / 1  # 时间间隔为1
                if np.sign(k_next) != np.sign(k) and abs(k_next - k) > 0.1:
                    points.append(j)
                    i = j
                    break
                tan_theta = calculate_angle(k, k_next)
                gamma = calculate_fluctuation_rate(errors, j, kp, ks, kd, pi)
                tan_alpha = adaptive_angle_threshold(k, gamma)
                if tan_theta > tan_alpha:
                    points.append(j)
                    i = j
                    break
        else:  # 如果内层循环没有break，则i递增
            i += 1
    
    # 确保最后一个点被包含
    if points[-1] != n - 1:
        points.append(n - 1)
    
    return points

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w):
    num_points = len(slopes)
    allocated = np.zeros(num_points)
    window_sum = 0.0  # 滑动窗口预算总和
    
    for i in range(num_points):
        # 维护滑动窗口预算总和
        if i >= w:
            window_sum -= allocated[i - w]
        remaining = max(total_budget - window_sum, 0)
        
        k = slopes[i]
        gamma = fluctuation_rates[i]
        
        # 计算pk（严格遵循公式15）
        if k >= 0:
            pk = 1 - np.exp(-k)
        else:
            pk = 1 - np.exp(k)  # 注意符号
        
        # 计算pgamma（公式16）
        pgamma = 1 - np.exp(-gamma)
        
        # 计算pkγ（严格遵循公式17）
        if k != 0:  # 避免除以零
            if k > 0:
                pky = 1 - np.exp(-1 / (k * gamma + 1e-8))
            else:
                pky = 1 - np.exp(-abs(k) / (gamma + 1e-8))  # 注意绝对值处理
        else:
            pky = 1.0  # 当k=0时的特殊处理
        
        # 计算p（公式18）
        numerator = pk + pgamma
        denominator = pky + 1e-8
        p = 1 - np.exp(-numerator / denominator)
        p = np.clip(p, 0, 1)  # 确保概率在合理范围
        
        # 分配预算
        epsilon_i = p * remaining
        allocated[i] = epsilon_i
        window_sum += epsilon_i  # 更新窗口总和
    
    return allocated

def sw_perturbation_w_event(values, budgets, min_budget=0.01):
    epsilons = np.maximum(budgets, min_budget)
    denominators = 2 * np.exp(epsilons) * (np.exp(epsilons) - 1 - epsilons)
    valid_mask = denominators > 1e-10
    
    # 向量化计算参数
    b = np.zeros_like(epsilons)
    b[valid_mask] = (epsilons[valid_mask] * np.exp(epsilons[valid_mask]) 
                    - np.exp(epsilons[valid_mask]) + 1) / denominators[valid_mask]
    
    perturb_probs = np.exp(epsilons) / (2 * b * np.exp(epsilons) + 1)
    perturb_probs = np.nan_to_num(perturb_probs, nan=1.0)  # 处理无效值
    
    # 批量生成随机数
    rand = np.random.rand(len(values))
    perturb_mask = rand <= perturb_probs
    
    # 向量化扰动
    laplace_noise = np.random.laplace(scale=b, size=len(values))
    perturbed = np.where(perturb_mask, values, values + laplace_noise)
    
    return perturbed

def kalman_filter(perturbed_values, process_variance=5e-4, measurement_variance=5e-3):
    n = len(perturbed_values)
    estimates = np.empty(n)
    variance = 1.0
    estimates[0] = perturbed_values[0]
    
    for t in range(1, n):
        # 预测步骤
        predicted_estimate = estimates[t-1]
        predicted_variance = variance + process_variance
        
        # 更新步骤
        kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
        estimates[t] = predicted_estimate + kalman_gain * (perturbed_values[t] - predicted_estimate)
        variance = (1 - kalman_gain) * predicted_variance
    return estimates
# =========================================================================
# 主实验接口：控制不同的实验变量
def run_experiment(file_path, output_dir, 
                   sample_fraction=1.0, 
                   total_budget=1.0, 
                   w=160, 
                   delta=0.5, 
                   kp=0.7, 
                   ks=0.15, 
                   kd=0.1,
                   process_variance=5e-4,
                   measurement_variance=5e-3,
                   DTW_MRE=True):
    # 数据预处理优化
    data, original_values = data_utils.preprocess_HKHS_data(
        file_path, 
        sample_fraction,
    )
    normalized_data = data['sample_normalized_value'].values
    original_data = data['normalized_value'].values
    # plot_utils.plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date)
    
    # 显著点采样（使用优化后的版本）
    significant_indices = remarkable_point_sampling(
        normalized_data, 
        original_data, 
        kp=kp, 
        ks=ks, 
        kd=kd
    )
    # plot_utils.plot_significant_points(data, normalized_data, significant_indices, output_dir, current_date, sample_fraction)
    
    # 特征计算优化
    slopes = np.gradient(normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes) + 1e-8  # 避免零值
    
    # 预算分配（使用优化后的版本）
    allocated_budgets = adaptive_w_event_budget_allocation(
        slopes, 
        fluctuation_rates, 
        total_budget=total_budget, 
        w=w
    )
    # plot_utils.plot_budget_allocation(allocated_budgets, output_dir, current_date, sample_fraction)
    
    # 扰动（向量化版本）
    perturbed_values = sw_perturbation_w_event(
        normalized_data[significant_indices], 
        allocated_budgets
    )
    # plot_utils.plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date, sample_fraction)
    
    # 卡尔曼滤波（优化版本）
    smoothed_values = kalman_filter(
        perturbed_values,
        process_variance=process_variance,
        measurement_variance=measurement_variance
    )
    # plot_utils.plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date, sample_fraction)
    
    # 结果生成与评估
    fitted_values = data_utils.generate_piecewise_linear_curve(
        data['date'], 
        significant_indices, 
        smoothed_values
    )
    
    # 并行计算评估指标
    dtw_distance = None
    mre = None
    if DTW_MRE:
        with ThreadPoolExecutor() as executor:
            dtw_future = executor.submit(
                data_utils.calculate_fdtw,
                normalized_data,
                fitted_values
            )
            mre_future = executor.submit(
                data_utils.calculate_mre,
                smoothed_values,
                normalized_data[significant_indices]
            )
            dtw_distance = dtw_future.result()
            mre = mre_future.result()
    
    return {
        'normalized_data': normalized_data,
        'smoothed_values': smoothed_values,
        'significant_indices': significant_indices,
        'perturbed_values': perturbed_values,
        'fitted_values': fitted_values,
        'dtw_distance': dtw_distance,
        'mre': mre
    }

# 统一接口调用
def compare_experiments(file_path, output_dir, target):
    """
    调用统一接口进行不同变量的实验，返回并对比结果。
    """
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
    if target == "e":
        # 实验 1: 只改变隐私预算
        es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        results = []
        for e in es:
            result_budget = run_experiment(
                file_path,
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=e, 
                w=160, 
                DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        # 实验 2: 只改变窗口大小
        ws = [80,100,120,140,160,180,200,220,240,260]
        results = []
        for w in ws:
            result_window = run_experiment(
                file_path, 
                output_dir,
                sample_fraction=sample_fraction, 
                total_budget=1.0,
                w=w,
                DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)

if __name__ == "__main__":
    file_path = '../data/HKHS.csv'  # 输入数据路径
    output_dir = 'results'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    # eff_result = effiency_utils.memory_function(run_experiment, file_path, output_dir, sample_fraction=1.0, total_budget=1.0, w=160, delta=0.5, kp=0.8, ks=0.1, kd=0.1, DTW_MRE=False)
    compare_experiments(file_path, output_dir,target="e")
    compare_experiments(file_path, output_dir,target="w")
    # normalized_data_80, sample_80_smoothed_values, significant_indices_80, perturbed_values_80, piecewise_fitted_values_80 = process(file_path, output_dir, 0.8, 160, 1.0)