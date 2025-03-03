import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils
import utils.effiency_utils as effiency_utils
from concurrent.futures import ThreadPoolExecutor

# 使用向量化计算代替循环
def calculate_slope(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b
def calculate_angle(k1, k2):
    """
    计算两条直线斜率之间的角度（tan θ）
    """
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
    """
    计算当前点的误差 e(t_i)
    """
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

def remarkable_point_sampling(data, kp=0.8, ks=0.1, kd=0.1, pi=5):
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

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, data_length):
    """
    论文方法：对所有数据点分配预算，而不仅是显著点
    """
    allocated = np.zeros(data_length)  # 初始化所有数据点预算
    window_sum = 0.0  # 滑动窗口预算总和

    for i in range(data_length):
        # 维护滑动窗口预算总和
        if i >= w:
            window_sum -= allocated[i - w]
        remaining = max(total_budget - window_sum, 0)  # 计算当前可分配预算
        
        # 计算 k 和 gamma
        k = slopes[i] if i < len(slopes) else 0  # 避免索引超界
        gamma = fluctuation_rates[i] if i < len(fluctuation_rates) else 0

        # 计算 pk
        if k >= 0:
            pk = 1 - np.exp(-k)
        else:
            pk = 1 - np.exp(k)

        # 计算 pγ
        pγ = 1 - np.exp(-gamma)

        # 计算 pkγ
        if k != 0:
            pkγ = 1 - np.exp(-1 / (abs(k) * gamma + 1e-8))
        else:
            pkγ = 1.0  # 处理 k=0 情况

        # 计算最终的 p
        p = 1 - np.exp(- (pk + pγ) / (pkγ + 1e-8))
        p = np.clip(p, 0, 1)  # 预算比例在 [0,1] 之间

        # 计算最终预算 εi
        allocated[i] = p * remaining
        window_sum += allocated[i]  # 更新滑动窗口总预算

    return allocated

def sw_perturbation_w_event(values, budgets, min_budget=0.001):
    """
    论文中的 SW 机制，对所有数据点添加扰动
    """
    epsilons = np.maximum(budgets, min_budget)  # 确保每个点至少有最小预算
    denominators = 2 * np.exp(epsilons) * (np.exp(epsilons) - 1 - epsilons)
    valid_mask = denominators > 1e-10  # 过滤有效值

    # 计算 b[i]，根据论文公式
    b = np.zeros_like(epsilons)
    b[valid_mask] = (epsilons[valid_mask] * np.exp(epsilons[valid_mask]) 
                    - np.exp(epsilons[valid_mask]) + 1) / denominators[valid_mask]

    # 计算扰动概率
    perturb_probs = np.exp(epsilons) / (2 * b * np.exp(epsilons) + 1)

    # 生成随机扰动
    rand = np.random.rand(len(values))
    perturb_mask = rand <= perturb_probs

    # 添加噪声
    laplace_noise = np.random.laplace(scale=b, size=len(values))
    perturbed = np.where(perturb_mask, values, values + laplace_noise)

    return perturbed

def kalman_filter(perturbed_values, process_variance=5e-4, measurement_variance=5e-3):
    """
    论文中的 Kalman 滤波器，优化扰动数据
    """
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
        variance = (1 - kalman_gain) * predicted_variance  # 更简洁的更新

    return estimates

# =========================================================================
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
    # sample_data, origin_data = data_utils.preprocess_HKHS_data(
    #     file_path, 
    #     sample_fraction,
    # )
    sample_data, origin_data = data_utils.preprocess_heartrate_data(
        file_path, 
        sample_fraction,
    ) # ['date', 'normalized_value'] 两个都是这样的格式
    sample_normalized_data = sample_data['normalized_value'].values
    sample_data_length = len(sample_normalized_data)
    origin_normalized_data = origin_data['normalized_value'].values
    origin_data_length = len(sample_normalized_data)
    # plot_utils.plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date)
    
    # 显著点采样（使用优化后的版本）
    significant_indices = remarkable_point_sampling(
        sample_normalized_data, 
        kp=kp, 
        ks=ks, 
        kd=kd
    ) # 获取显著点的索引，是采样后的数据的显著点索引
    # plot_utils.plot_significant_points(data, normalized_data, significant_indices, output_dir, current_date, sample_fraction)
    
    # 特征计算优化
    slopes = np.gradient(sample_normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes) + 1e-8  # 避免零值
    
    # 预算分配（使用优化后的版本）
    allocated_budgets = adaptive_w_event_budget_allocation(
        slopes, fluctuation_rates, total_budget, w, sample_data_length
    )
    # plot_utils.plot_budget_allocation(allocated_budgets, output_dir, current_date, sample_fraction)
    
    # 扰动（向量化版本）
    perturbed_values = sw_perturbation_w_event(
        sample_normalized_data, allocated_budgets
    )
    # plot_utils.plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date, sample_fraction)
    
    # 卡尔曼滤波（优化版本）
    smoothed_values = kalman_filter(
        perturbed_values, process_variance=process_variance, measurement_variance=measurement_variance
    )
    sample_data["smoothed_value"] = smoothed_values
    # plot_utils.plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date, sample_fraction)

     # 插值
    interpolated_data = data_utils.interpolate_missing_points(origin_data, sample_data)
    interpolated_values = interpolated_data['smoothed_value']
    # 并行计算评估指标
    dtw_distance = None
    mre = None
    if DTW_MRE:
        with ThreadPoolExecutor() as executor:
            dtw_future = executor.submit(
                data_utils.calculate_fdtw,
                origin_normalized_data,
                interpolated_values
            ) # original_series, fitted_series
            mre_future = executor.submit(
                data_utils.calculate_mre,
                interpolated_values,
                origin_normalized_data
            ) # perturbed_values, normalized_values
            dtw_distance = dtw_future.result()
            mre = mre_future.result()
    
    return {
        'sample_normalized_data': sample_normalized_data,
        'sample_significant_indices': significant_indices,
        'sample_perturbed_values': perturbed_values,
        'sample_smoothed_values': smoothed_values,
        'dtw_distance': dtw_distance,
        'mre': mre
    }