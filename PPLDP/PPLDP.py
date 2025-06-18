import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
from concurrent.futures import ThreadPoolExecutor
import datetime
import time
import tracemalloc
# 使用向量化计算代替循环
def calculate_slope(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

def calculate_slope_fast(start, end, sum_x, sum_y, sum_xy, sum_xx):
    n = end - start + 1
    Sx = sum_x[end + 1] - sum_x[start]
    Sy = sum_y[end + 1] - sum_y[start]
    Sxy = sum_xy[end + 1] - sum_xy[start]
    Sxx = sum_xx[end + 1] - sum_xx[start]
    
    denominator = n * Sxx - Sx**2 + 1e-8  # 避免除以0
    k = (n * Sxy - Sx * Sy) / denominator
    b = (Sy - k * Sx) / n
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
    points = [0]  # 初始点
    i = 0
    errors = np.zeros(n)

    # 前缀和（用于拟合加速）
    x = np.arange(n)
    y = np.array(data)
    sum_x = np.concatenate([[0], np.cumsum(x)])
    sum_y = np.concatenate([[0], np.cumsum(y)])
    sum_xy = np.concatenate([[0], np.cumsum(x * y)])
    sum_xx = np.concatenate([[0], np.cumsum(x * x)])

    while i < n - 1:
        start = i
        for j in range(i + 1, n):
            # 拟合 i~j 段的斜率 k 和截距 b
            k, b = calculate_slope_fast(start, j, sum_x, sum_y, sum_xy, sum_xx)
            fitted_value = k * j + b
            errors[j] = abs(fitted_value - data[j])

            if j + 1 < n:
                # 计算下一个斜率
                k_next = data[j + 1] - data[j]
                tan_theta = calculate_angle(k, k_next)
                gamma = calculate_fluctuation_rate(errors, j, kp, ks, kd, pi)
                tan_alpha = adaptive_angle_threshold(k, gamma)

                if tan_theta > tan_alpha:
                    # 拟合失败，当前点为显著点，重新开始新段拟合
                    points.append(j)
                    i = j
                    break
            else:
                # 最后一个点也要加入
                if points[-1] != n - 1:
                    points.append(n - 1)
                i = n  # 终止外循环
                break

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

def sw_perturbation_forced(values, budgets):
    epsilons = np.maximum(budgets, 1e-3)  # 确保每个点至少有最小预算
    # b = np.maximum(1.0 / epsilons, 1e-3) # 限制最小扰动范围
    b = 1.0 / epsilons  
    noise = np.random.laplace(scale=b, size=len(values))
    return values + noise
def sw_perturbation(values, budgets):
    perturbed = []
    for v, eps in zip(values, budgets):
        eps = max(eps, 1e-3)
        exp_eps = np.exp(eps)

        # 计算扰动范围 b[i]（论文公式 (19)）
        numerator = eps * exp_eps - exp_eps + 1
        denominator = 2 * exp_eps * (exp_eps - 1 - eps) + 1e-8  # 避免除0
        b = numerator / denominator
        b = max(b, 1e-6)

        # 概率定义（论文公式 (20)）
        p = exp_eps / (2 * b * exp_eps + 1)
        q = 1 / (2 * b * exp_eps + 1)

        # 按概率采样扰动值
        if np.random.rand() < p:
            # 添加噪声在 [-b, b] 区间内
            noise = np.random.uniform(-b, b)
        else:
            # 添加较大噪声（论文未明确定义，可选择合理范围）
            noise = np.random.uniform(-3*b, 3*b)
        perturbed.append(v + noise)
    return np.array(perturbed)

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

def run_single_experiment(x, eps, file_path, w, sample_method="uniform"):
    # 开始时间和内存监控
    start_time = time.time()
    tracemalloc.start()
    sample_data, origin_data = data_utils.preprocess_data(file_path, x, sample_method) # ['date', 'normalized_value'] 两个都是这样的格式
    sample_normalized_data = sample_data['normalized_value'].values
    sample_data_length = len(sample_normalized_data)
    origin_normalized_data = origin_data['normalized_value'].values
    origin_data_length = len(sample_normalized_data)
    
    # 显著点采样（使用优化后的版本）
    significant_indices = remarkable_point_sampling(
        sample_normalized_data, 
        kp=0.8, 
        ks=0.1, 
        kd=0.1
    ) # 获取显著点的索引，是采样后的数据的显著点索引
    
    # 特征计算优化
    slopes = np.gradient(sample_normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes) + 1e-8  # 避免零值
    
    # 预算分配（使用优化后的版本）
    allocated_budgets = adaptive_w_event_budget_allocation(
        slopes, fluctuation_rates, eps, w, sample_data_length
    )
    
    # 扰动（向量化版本）
    perturbed_values = sw_perturbation(sample_normalized_data, allocated_budgets)
    # perturbed_values = sw_perturbation_w_event(sample_normalized_data, allocated_budgets)
    
    # 卡尔曼滤波（优化版本）
    smoothed_values = kalman_filter(
        perturbed_values, process_variance=5e-4, measurement_variance=5e-3
    )
    sample_data["smoothed_value"] = smoothed_values

     # 插值
    interpolated_data = data_utils.interpolate_missing_points(origin_data, sample_data)
    interpolated_values = interpolated_data['smoothed_value']
    # 统计耗时、内存峰值
    elapsed_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 并行计算评估指标
    dtw_distance = None
    mre = None
    if True:
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
        "sampling_rate": x,
        "epsilon": eps,
        "dtw": dtw_distance,
        "mre": mre,
        "runtime": elapsed_time,
        "peak_memory": peak_mem / 10**6
    }


if __name__ == "__main__":
    # file_path = 'data/HKHS.csv'
    # file_path = 'data/heartrate.csv'
    file_path = 'data/LD.csv'
    # file_path = "data/ETTh1.csv" #可用！！！电力变压器温度 (ETT) 是电力长期部署的关键指标。该数据集由来自中国两个分离县的2年数据组成。为了探索长序列时间序列预测 (LSTF) 问题的粒度，创建了不同的子集，{ETTh1，ETTh2} 为1小时级，ETTm1为15分钟级。每个数据点由目标值 “油温” 和6个功率负载特征组成。火车/val/测试为12/4/4个月。https://opendatalab.com/OpenDataLab/ETT
    # file_path = "data/exchange_rate.csv"  
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    res = run_single_experiment(
        x=1, 
        eps=1.0, 
        file_path=file_path, 
        w=160
    )
    print(res)