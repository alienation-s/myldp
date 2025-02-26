import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

# 设置随机种子以确保可重复性
np.random.seed(42)

# =========================================================================
# Step 0: 统一数据处理接口
# =========================================================================
def process_hkhs_data(file_path, output_dir, sample_fraction=1.0, w=160, total_budget=1.0):
    """
    主函数：处理 HKHS 数据并保存处理结果。
    """
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d%H%M')

    # Step 1: 数据预处理
    data, _, = preprocess_data(file_path, sample_fraction)
    normalized_data = data['sample_normalized_value'].values
    original_data = data['normalized_value'].values

    # 保存归一化图表
    plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date)

    # Step 2: 显著点采样
    significant_indices = remarkable_point_sampling(normalized_data, original_data)  # 调用新采样函数
    plot_significant_points(data, normalized_data, significant_indices, output_dir, current_date, sample_fraction)

    # Step 3: 自适应预算分配
    slopes = np.gradient(normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes)
    allocated_budgets = adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget=total_budget, w=w)
    plot_budget_allocation(allocated_budgets, output_dir, current_date, sample_fraction)

    # Step 4: SW扰动机制
    perturbed_values = sw_perturbation_w_event(normalized_data[significant_indices], allocated_budgets)
    plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date, sample_fraction)

    # Step 5: 卡尔曼滤波平滑
    smoothed_values = kalman_filter(perturbed_values)
    plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date, sample_fraction)

    return normalized_data, smoothed_values, significant_indices, perturbed_values

# =========================================================================
# Step 1: 数据预处理
# =========================================================================
def preprocess_data(file_path, sample_fraction=1.0):
    data = pd.read_csv(file_path)
    max_val = data[' value'].max()
    min_val = data[' value'].min()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')

    data['normalized_value'] = (data[' value'] - min_val) / (max_val - min_val)
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42).sort_values(by='date')

    # data['sample_normalized_value'] = (data[' value'] - data[' value'].min()) / (
    #     data[' value'].max() - data[' value'].min())
    # 归一化后的采样数据列
    data['sample_normalized_value'] = (data[' value'] - min_val) / (max_val - min_val)
    return data, data[' value'].values

def plot_normalized_data(data, normalized_data, sample_fraction, output_dir, current_date):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], normalized_data, label='Normalized Data')
    plt.title(f'Step 1: Normalized HKHS Index (sample_fraction={sample_fraction})')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{current_date}_{sample_fraction}_Step1_Normalized_HKHS_Index.png"), dpi=300)
    plt.close()

# =========================================================================
# Step 2: 显著点采样
# =========================================================================
def calculate_slope(points):
    """
    使用最小二乘法计算直线的斜率(k)和截距(b)
    参数:
        points (np.ndarray): 包含 (x, y) 坐标的点集
    返回:
        k: 斜率
        b: 截距
    """
    x = points[:, 0]
    y = points[:, 1]
    n = len(points)
    k = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(y) - k * np.sum(x)) / n
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
    """
    动态调整角度阈值 alpha，根据趋势斜率(k)和波动率(gamma)
    参数:
        k: 当前直线的斜率
        gamma: 数据的波动率（fluctuation rate）
        kp, ks, kd: 动态调整参数
    返回:
        tan_alpha: 动态调整后的角度阈值（tan值）
    """
    lambda_val = 1 - np.exp(-1 / (abs(k) + gamma))  # 公式（13）
    alpha = lambda_val * np.pi / 2  # 根据公式（13）计算 alpha
    return np.tan(alpha)
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
    integral_term = (ks / pi) * integral_term

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
    基于改进的分段最小二乘拟合 (LSSLF) 采样显著点，并使用公式（12）计算波动率 γ(t_i)
    参数:
        data (np.ndarray): 时间序列数据，假设为一维数组
        kp (float): 比例控制参数
        ks (float): 积分控制参数
        kd (float): 微分控制参数
        pi (int): 滑动窗口长度，用于积分项计算
    返回:
        points (list): 显著点的索引列表
    """
    n = len(data)
    points = []
    i = 0

    # 初始化误差序列
    errors = np.zeros(n)

    while i < n - 1:
        start = i
        points.append(start)  # 当前拟合过程的起点
        for j in range(i + 1, n):
            # 拟合当前片段的数据
            segment = np.array([[x, data[x]] for x in range(start, j + 1)])
            k, b = calculate_slope(segment)
            
            # 更新拟合值并计算误差
            fitted_value = k * j + b
            errors[j] = calculate_error(fitted_value, data[j])

            # 检查下一个点是否超出角度阈值或趋势变化
            if j + 1 < n:
                k_next = (data[j + 1] - data[j]) / (j + 1 - j)  # 下一个点与当前点的斜率

                # 检查趋势变化（正负斜率变化）
                if np.sign(k_next) != np.sign(k):  # 趋势发生变化
                    points.append(j)  # 将当前点作为显著点
                    i = j  # 重新开始拟合过程
                    break

                # 计算当前点与下一点的角度
                tan_theta = calculate_angle(k, k_next)  # 当前斜率与下一斜率的夹角

                # 使用公式（12）计算波动率
                gamma = calculate_fluctuation_rate(errors, j, kp=kp, ks=ks, kd=kd, pi=pi)

                # 动态调整角度阈值
                tan_alpha = adaptive_angle_threshold(k, gamma)  # 动态调整角度阈值

                if tan_theta > tan_alpha:  # 若超出角度阈值，则当前点为显著点
                    i = j
                    break
        i += 1

    if n - 1 < len(data):
        points.append(n - 1)  # 将最后一个点作为显著点
    # 映射到原始数据的索引
    # mapped_points = [np.where(original_data == data.iloc[point]['value'])[0][0] for point in points]
    # mapped_points = [np.where(original_data == data[point])[0][0] for point in points]
    return points

def plot_significant_points(data, normalized_data, significant_indices,output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], normalized_data, label='Normalized Data')
    plt.scatter(data['date'].iloc[significant_indices], normalized_data[significant_indices],
                color='red', label='Significant Points', marker='x')
    plt.title('Step 2: Significant Points Sampling')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{current_date}_{sample_fraction}_Step2_Significant_Points_Sampling.png"), dpi=300)
    plt.close()

# =========================================================================
# Step 3: 自适应预算分配
# =========================================================================
def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, min_budget=1e-5):
    """
    基于论文公式实现的自适应预算分配，确保任意 w 个显著点内总预算不超过 total_budget。
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    
    for i in range(num_points):
        # 计算剩余预算 ε'
        window_start = max(0, i - w + 1)
        window_end = i
        used_budget = np.sum(allocated_budgets[window_start:window_end])
        remaining_budget = total_budget - used_budget
        if remaining_budget <= 0:
            allocated_budgets[i] = min_budget
            continue
        
        # 计算 pk（数据流幅度）
        if slopes[i] >= 0:
            pk = 1 - np.exp(-slopes[i])
        else:
            pk = 1 - np.exp(slopes[i])
        
        # 计算 pγ（数据流波动率）
        pgamma = 1 - np.exp(-fluctuation_rates[i])
        
        # 计算 pky（潜在采样点分布）
        if slopes[i] >= 0:
            pky = 1 - np.exp(-1 / (slopes[i] * fluctuation_rates[i] + 1e-8))
        else:
            pky = 1 - np.exp(-slopes[i] / (fluctuation_rates[i] + 1e-8))
        
        # 计算最终分配比例 p
        p = 1 - np.exp(-((pk + pgamma) / (pky + 1e-8)))
        
        # 根据论文公式 (18) 计算当前点分配的预算 ε_i
        epsilon_i = p * remaining_budget
        
        # 确保预算值大于最小预算
        epsilon_i = max(epsilon_i, min_budget)
        
        allocated_budgets[i] = epsilon_i

    # 打印分配预算的统计信息
    print("分配的隐私预算统计指标:")
    print(pd.Series(allocated_budgets).describe())
    
    return allocated_budgets

def plot_budget_allocation(budgets, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(budgets)), budgets, color='blue')
    plt.title('Step 3: Adaptive Budget Allocation with w-event Privacy')
    plt.xlabel('Significant Point Index')
    plt.ylabel('Privacy Budget (epsilon)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{current_date}_{sample_fraction}_Step3_Budget_Allocation.png"), dpi=300)
    plt.close()

# =========================================================================
# Step 4: SW扰动机制
# =========================================================================
def sw_perturbation_w_event(values, budgets, min_budget=0.01):
    """
    对显著点应用SW机制，满足w-event隐私，按照论文公式扰动
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
    print(f"最终扰动值数量: {len(perturbed_values)}")
    return perturbed_values

def plot_perturbed_values(data, normalized_data, significant_indices, perturbed_values, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'].iloc[significant_indices], perturbed_values,
         label='Perturbed Points', color='red', linestyle='--', marker='x')
    plt.plot(data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1)
    plt.title('Step 4: SW Mechanism Perturbation')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{current_date}_{sample_fraction}_Step4_Perturbation.png"), dpi=300)
    plt.close()

# =========================================================================
# Step 5: 卡尔曼滤波
# =========================================================================
def kalman_filter(perturbed_values, process_variance=1e-4, measurement_variance=1e-2):
    """
    基于论文公式的 Kalman 滤波器实现。
    """
    n = len(perturbed_values)
    estimates = np.zeros(n)
    variance = 1.0  # 初始协方差
    estimates[0] = perturbed_values[0]  # 初始化估计值

    for t in range(1, n):
        # 预测过程
        predicted_estimate = estimates[t - 1]
        predicted_variance = variance + process_variance

        # 更新过程
        kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
        estimates[t] = predicted_estimate + kalman_gain * (perturbed_values[t] - predicted_estimate)
        variance = (1 - kalman_gain) * predicted_variance

    return estimates

def plot_kalman_smoothing(data, normalized_data, significant_indices, perturbed_values, smoothed_values, output_dir, current_date,sample_fraction):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1, 
         zorder=3, marker='.', markersize=3)
    plt.scatter(data['date'].iloc[significant_indices], perturbed_values,
            label='Perturbed Points', color='red', marker='x')
    plt.plot(data['date'].iloc[significant_indices], smoothed_values,
         label='Smoothed Points', color='blue', linestyle='-', linewidth=1, zorder=1)
    plt.title('Step 5: Kalman Filter Smoothing')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{current_date}_{sample_fraction}_Step5_Kalman_Smoothing.png"), dpi=300)
    plt.close()

def calculate_dtw_numpy(series1, series2):
    """
    使用 NumPy 计算 DTW 距离
    参数:
        series1 (np.ndarray): 时间序列 1
        series2 (np.ndarray): 时间序列 2
    返回:
        distance (float): DTW 距离
    """
    n, m = len(series1), len(series2)
    dtw_matrix = np.zeros((n + 1, m + 1)) + np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series1[i - 1] - series2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1]) # match
    return dtw_matrix[n, m]

from dtaidistance import dtw, dtw_visualisation as dtwvis
def visualize_dtw(series1, series2, output_path):
    """
    可视化 DTW 匹配路径
    参数:
        series1 (np.ndarray): 时间序列 1
        series2 (np.ndarray): 时间序列 2
        output_path (str): 保存图像路径
    """
    path = dtw.warping_path(series1, series2)
    dtwvis.plot_warping(series1, series2, path, filename=output_path)

def calculate_mre(perturbed_values, normalized_values):
    # 方法 1：添加小值 epsilon 避免除以零
    # mre = np.mean(np.abs((perturbed_values - normalized_values) / (normalized_values + 1e-8)))
    
    # 方法 2：跳过零值
    # non_zero_indices = normalized_values != 0
    # mre = np.mean(np.abs((perturbed_values[non_zero_indices] - normalized_values[non_zero_indices]) / normalized_values[non_zero_indices]))
    
    # 方法 3：改用替代公式
    mre = np.mean(np.abs(perturbed_values - normalized_values) / (np.abs(normalized_values) + np.abs(perturbed_values) + 1e-8))
    
    return mre
if __name__ == "__main__":
    file_path = 'data/HKHS.csv'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取采样比例为100%和80%的数据
    normalized_data_100, sample_100_smoothed_values, significant_indices_100, perturbed_values_100 = process_hkhs_data(file_path, output_dir, 1.0, 160, 1.0)
    normalized_data_80, sample_80_smoothed_values, significant_indices_80, perturbed_values_80 = process_hkhs_data(file_path, output_dir, 0.8, 160, 1.0)
    
    # 输出显著点结果
    mre_100 = calculate_mre(sample_100_smoothed_values, normalized_data_100[significant_indices_100])
    mre_80 = calculate_mre(sample_80_smoothed_values, normalized_data_80[significant_indices_80])
    print(f"MRE for sample_fraction=1.0: {mre_100:.4f}")
    print(f"MRE for sample_fraction=0.8: {mre_80:.4f}")

    # 使用 NumPy 计算 DTW 距离
    dtw_distance_100 = calculate_dtw_numpy(normalized_data_100[significant_indices_100], sample_100_smoothed_values)
    dtw_distance_80 = calculate_dtw_numpy(normalized_data_80[significant_indices_80], sample_80_smoothed_values)

    print(f"DTW distance (NumPy) for sample_fraction=1.0: {dtw_distance_100:.4f}")
    print(f"DTW distance (NumPy) for sample_fraction=0.8: {dtw_distance_80:.4f}")

    # 可视化 DTW 匹配路径
    visualize_dtw(
        normalized_data_100[significant_indices_100], 
        sample_100_smoothed_values, 
        os.path.join(output_dir, "dtw_100.png")
    )
    visualize_dtw(
        normalized_data_80[significant_indices_80], 
        sample_80_smoothed_values, 
        os.path.join(output_dir, "dtw_80.png")
    )