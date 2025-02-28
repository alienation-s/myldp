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
''' 无注释用于问gpt用 '''
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
    if current_idx == 0:  
        return 0.0
    e_ti = errors[current_idx]  
    e_ti_1 = errors[current_idx - 1] if current_idx - 1 >= 0 else 0 
    proportional_term = kp * e_ti
    integral_term = 0
    start_idx = max(0, current_idx - pi)  
    for n in range(start_idx, current_idx + 1):
        integral_term += errors[n]
    integral_term = (ks / pi) * sum(errors[start_idx:current_idx+1])
    differential_term = 0
    if current_idx > 0:
        differential_term = kd * (e_ti - e_ti_1)
    gamma = proportional_term + integral_term + differential_term
    return gamma

def remarkable_point_sampling(data, original_data, kp=0.8, ks=0.1, kd=0.1, pi=5):
    n = len(data)
    points = [0]  
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
                k_next = (data[j + 1] - data[j]) / 1
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
        else:
            i += 1
    if points[-1] != n - 1:
        points.append(n - 1)
    return points

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w):
    num_points = len(slopes)
    allocated = np.zeros(num_points)
    window_sum = 0.0 
    for i in range(num_points):
        if i >= w:
            window_sum -= allocated[i - w]
        remaining = max(total_budget - window_sum, 0)
        k = slopes[i]
        gamma = fluctuation_rates[i]
        if k >= 0:
            pk = 1 - np.exp(-k)
        else:
            pk = 1 - np.exp(k)
        pgamma = 1 - np.exp(-gamma)
        if k != 0: 
            if k > 0:
                pky = 1 - np.exp(-1 / (k * gamma + 1e-8))
            else:
                pky = 1 - np.exp(-abs(k) / (gamma + 1e-8))
        else:
            pky = 1.0 
        numerator = pk + pgamma
        denominator = pky + 1e-8
        p = 1 - np.exp(-numerator / denominator)
        p = np.clip(p, 0, 1)
        epsilon_i = p * remaining
        allocated[i] = epsilon_i
        window_sum += epsilon_i  
    return allocated

def sw_perturbation_w_event(values, budgets, min_budget=0.01):
    epsilons = np.maximum(budgets, min_budget)
    denominators = 2 * np.exp(epsilons) * (np.exp(epsilons) - 1 - epsilons)
    valid_mask = denominators > 1e-10
    b = np.zeros_like(epsilons)
    b[valid_mask] = (epsilons[valid_mask] * np.exp(epsilons[valid_mask]) 
                    - np.exp(epsilons[valid_mask]) + 1) / denominators[valid_mask]
    perturb_probs = np.exp(epsilons) / (2 * b * np.exp(epsilons) + 1)
    perturb_probs = np.nan_to_num(perturb_probs, nan=1.0)
    rand = np.random.rand(len(values))
    perturb_mask = rand <= perturb_probs
    laplace_noise = np.random.laplace(scale=b, size=len(values))
    perturbed = np.where(perturb_mask, values, values + laplace_noise) 
    return perturbed

def kalman_filter(perturbed_values, process_variance=5e-4, measurement_variance=5e-3):
    n = len(perturbed_values)
    estimates = np.empty(n)
    variance = 1.0
    estimates[0] = perturbed_values[0]
    for t in range(1, n):
        predicted_estimate = estimates[t-1]
        predicted_variance = variance + process_variance
        kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
        estimates[t] = predicted_estimate + kalman_gain * (perturbed_values[t] - predicted_estimate)
        variance = (1 - kalman_gain) * predicted_variance  # 更简洁的更新
    return estimates

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
    slopes = np.gradient(normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes) + 1e-8  # 避免零值
    allocated_budgets = adaptive_w_event_budget_allocation(
        slopes, 
        fluctuation_rates, 
        total_budget=total_budget, 
        w=w
    )
    # plot_utils.plot_budget_allocation(allocated_budgets, output_dir, current_date, sample_fraction)
    perturbed_values = sw_perturbation_w_event(
        normalized_data[significant_indices], 
        allocated_budgets
    )
    # plot_utils.plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date, sample_fraction)
    smoothed_values = kalman_filter(
        perturbed_values,
        process_variance=process_variance,
        measurement_variance=measurement_variance
    )
    # plot_utils.plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date, sample_fraction)
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
