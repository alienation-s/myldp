import numpy as np
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
# Step 2: 显著点采样
# =========================================================================
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
    参数:
        k1: 当前拟合直线的斜率
        k2: 新增点形成的直线的斜率
    返回:
        tan_theta: 两条直线的夹角的tan值
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
    参数:
        fitted_value (float): 拟合值
        actual_value (float): 实际值
    返回:
        error (float): 当前点的误差
    """
    return abs(fitted_value - actual_value)

def calculate_fluctuation_rate(errors, current_idx, kp=0.8, ks=0.1, kd=0.1, pi=5):
    """
    根据公式（12）计算波动率 γ(t_i)
    参数:
        errors (list or np.ndarray): 误差序列
        current_idx (int): 当前时间点索引
        kp (float): 比例控制参数
        ks (float): 积分控制参数
        kd (float): 微分控制参数
        pi (int): 滑动窗口的长度
    返回:
        gamma (float): 当前点的波动率 γ(t_i)
    """
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
    integral_term = (ks / pi) * (sum(errors[start_idx:current_idx+1]) / pi)

    # 计算微分控制项
    differential_term = 0
    if current_idx > 0:
        differential_term = kd * (e_ti - e_ti_1)
        # differential_term = kd * (e_ti - e_ti_1) / (t_i - t_i_1)

    # 总波动率
    gamma = proportional_term + integral_term + differential_term
    return gamma

def remarkable_point_sampling(data, original_data, kp=0.8, ks=0.1, kd=0.1, pi=5):
    """
    基于论文中的LSSLF算法采样显著点。确保每次拟合是基于当前点及其前序点，避免过度拟合。
    """
    n = len(data)
    points = []
    i = 0
    errors = np.zeros(n)
    while i < n - 1:
        start = i
        points.append(start)  # 添加当前显著点起始点
        for j in range(i + 1, n):
            segment = np.array([[x, data[x]] for x in range(start, j + 1)])
            k, b = calculate_slope(segment)
            fitted_value = k * j + b
            errors[j] = calculate_error(fitted_value, data[j])
            
            if j + 1 < n:
                k_next = (data[j + 1] - data[j]) / (j + 1 - j)
                if np.sign(k_next) != np.sign(k):
                    # 增加幅度阈值判断，避免微小波动误判
                    if abs(k_next - k) > 0.1:  # 根据数据特性调整阈值
                        points.append(j)
                        i = j
                        break
                tan_theta = calculate_angle(k, k_next)
                gamma = calculate_fluctuation_rate(errors, j, kp=kp, ks=ks, kd=kd, pi=pi)
                tan_alpha = adaptive_angle_threshold(k, gamma)
                if tan_theta > tan_alpha:
                    i = j
                    break
        i += 1
    if n - 1 < len(data):
        points.append(n - 1)
    return points

# =========================================================================
# Step 3: 自适应预算分配
# =========================================================================
def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, min_budget=1e-5):
    """
    根据论文公式实现自适应预算分配，确保w个显著点的总预算不超过total_budget
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    
    for i in range(num_points):
        window_start = max(0, i - w + 1)
        window_end = i
        used_budget = np.sum(allocated_budgets[window_start:window_end])
        remaining_budget = total_budget - used_budget
        if remaining_budget <= 0:
            allocated_budgets[i] = min_budget
            continue
        
        if slopes[i] >= 0:
            pk = 1 - np.exp(-slopes[i])
        else:
            pk = 1 - np.exp(slopes[i])
        
        pgamma = 1 - np.exp(-fluctuation_rates[i])
        if slopes[i] >= 0:
            pky = 1 - np.exp(-1 / (abs(slopes[i]) * fluctuation_rates[i] + 1e-8))
        else:
            pky = 1 - np.exp(-abs(slopes[i]) / (fluctuation_rates[i] + 1e-8))
        pky = np.clip(pky, 1e-8, 1 - 1e-8)  
        p = 1 - np.exp(-((pk + pgamma) / (pky + 1e-8)))
        epsilon_i = p * remaining_budget
        epsilon_i = max(epsilon_i, min_budget)
        allocated_budgets[i] = epsilon_i

    return allocated_budgets

# =========================================================================
# Step 4: SW扰动机制
# =========================================================================
def sw_perturbation_w_event(values, budgets, min_budget=0.01):
    """
    对显著点应用SW机制，确保按照论文公式进行扰动
    """
    perturbed_values = []
    assert len(values) == len(budgets), "Values and budgets arrays have different lengths"
    for value, epsilon in zip(values, budgets):
        epsilon = max(epsilon, min_budget)
        denominator = 2 * np.exp(epsilon) * (np.exp(epsilon) - 1 - epsilon)
        if denominator <= 1e-10:
            perturbed_values.append(value)
            continue
        b = (epsilon * np.exp(epsilon) - np.exp(epsilon) + 1) / denominator
        perturb_prob = np.exp(epsilon) / (2 * b * np.exp(epsilon) + 1)
        if np.random.rand() <= perturb_prob:
            perturbed_values.append(value)
        else:
            perturbed_values.append(value + np.random.laplace(scale=b))
    return perturbed_values

# =========================================================================
# Step 5: 卡尔曼滤波
# =========================================================================
def kalman_filter(perturbed_values, process_variance=5e-4, measurement_variance=5e-3):
    """
    根据论文公式进行卡尔曼滤波平滑。
    """
    n = len(perturbed_values)
    estimates = np.zeros(n)
    variance = 1.0
    estimates[0] = perturbed_values[0]
    for t in range(1, n):
        predicted_estimate = estimates[t - 1]
        predicted_variance = variance + process_variance
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
                   process_variance=5e-4,  # 新增卡尔曼参数
                   measurement_variance=5e-3,  # 新增卡尔曼参数
                   DTW_MRE=True):
    """
    统一接口：控制实验中的各个变量，如数据量、隐私预算、窗口大小等，返回实验结果。
    """
    current_date = datetime.now().strftime('%Y%m%d%H%M')
    # Step 1: 数据预处理
    data, original_values = data_utils.preprocess_HKHS_data(file_path, sample_fraction)
    normalized_data = data['sample_normalized_value'].values
    original_data = data['normalized_value'].values

    # 保存归一化图表
    # plot_utils.plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date)

    # Step 2: 显著点采样
    significant_indices = remarkable_point_sampling(normalized_data, original_data, kp=kp, ks=ks, kd=kd)
    # plot_utils.plot_significant_points(data, normalized_data, significant_indices, output_dir, current_date, sample_fraction)

    # Step 3: 自适应预算分配
    slopes = np.gradient(normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes)
    allocated_budgets = adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget=total_budget, w=w)
    # plot_utils.plot_budget_allocation(allocated_budgets, output_dir, current_date, sample_fraction)

    # Step 4: SW扰动机制
    perturbed_values = sw_perturbation_w_event(normalized_data[significant_indices], allocated_budgets)
    # plot_utils.plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date, sample_fraction)

    # Step 5: 卡尔曼滤波平滑
    smoothed_values = kalman_filter(perturbed_values, process_variance=process_variance, measurement_variance=measurement_variance)
    # plot_utils.plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date, sample_fraction)

    # Step 6: 显著点拟合曲线生成
    fitted_values = data_utils.generate_piecewise_linear_curve(data['date'], significant_indices, smoothed_values)
    dtw_distance = None
    mre = None
    # 计算并返回所需的度量（如 DTW 和 MRE）
    if(DTW_MRE):
        dtw_distance = data_utils.calculate_fdtw(normalized_data, fitted_values)
        mre = data_utils.calculate_mre(smoothed_values, normalized_data[significant_indices])

    return {
        'normalized_data': normalized_data,
        'smoothed_values': smoothed_values,
        'significant_indices': significant_indices,
        'perturbed_values': perturbed_values,
        'fitted_values': fitted_values,
        'dtw_distance': dtw_distance,
        'mre': mre
    }
