import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

file_path = 'data/HKHS.csv'
# 设置保存路径
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 获取当前日期
current_date = datetime.now().strftime('%Y%m%d')
# --------------------
# Step 1: 数据预处理
# --------------------

def preprocess_data(file_path):
    """读取数据并进行标准化"""
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    data['normalized_value'] = (data[' value'] - data[' value'].min()) / (
        data[' value'].max() - data[' value'].min())
    return data, data[' value'].values  # 返回原始值

hkhs_data, original_data = preprocess_data(file_path)
normalized_data = hkhs_data['normalized_value'].values

# 数据可视化
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')
plt.title('Step 1: Normalized HKHS Index')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step1_Normalized_HKHS_Index.png"), dpi=300)
plt.show()

# --------------------
# Step 2: 显著点采样
# --------------------

def dynamic_lambda(slope, fluctuation_rate):
    """动态计算 λ 参数"""
    return 1 - np.exp(-1 / (abs(slope) + fluctuation_rate + 1e-8))

def calculate_dynamic_threshold(slope, fluctuation_rate, min_alpha=0.05, max_alpha=0.8):
    """计算动态角度阈值 α"""
    lambda_value = dynamic_lambda(slope, fluctuation_rate)
    alpha = lambda_value * np.pi / 2  # 动态角度阈值
    return np.clip(alpha, min_alpha, max_alpha)

def adaptive_significant_points(data, max_search=10, min_alpha=0.05, max_alpha=0.8, slope_diff_threshold=1e-4):
    """
    基于动态阈值的显著点采样，增加搜索范围限制和容错机制。
    """
    significant_points = [0]  # 起始点
    i = 0
    fluctuation_rate = 0  # 波动率初始化

    while i < len(data) - 2:
        start = i
        found_significant_point = False  # 标记是否找到显著点

        for j in range(i + 1, min(i + max_search, len(data))):
            # 拟合当前段直线
            x = np.arange(start, j + 1)
            y = data[start:j + 1]
            slope, _ = np.polyfit(x, y, 1)

            # 计算下一点的斜率变化
            if j + 1 < len(data):
                next_slope = (data[j + 1] - data[j]) / 1
                tan_theta = abs((next_slope - slope) / (1 + slope * next_slope + 1e-8))
            else:
                break

            # 动态计算阈值 alpha
            lambda_value = dynamic_lambda(slope, fluctuation_rate)
            alpha = np.clip(lambda_value * np.pi / 2, min_alpha, max_alpha)
            threshold = max(np.tan(alpha), slope_diff_threshold)  # 增加固定最小阈值

            # 判断显著点条件
            if tan_theta > threshold:
                significant_points.append(j)
                i = j  # 更新起始点
                found_significant_point = True
                break

        # 如果未找到显著点，强制选择下一个点，防止死循环
        if not found_significant_point:
            i += 1  # 移动到下一个点
            significant_points.append(i)

        # 更新波动率
        if j + 1 < len(data):
            fluctuation_rate = abs(next_slope - slope)

    significant_points.append(len(data) - 1)  # 添加终点
    return significant_points

# 执行显著点采样
significant_indices = adaptive_significant_points(normalized_data)

# 可视化显著点
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')
plt.scatter(hkhs_data['date'].iloc[significant_indices], normalized_data[significant_indices],
            color='red', label='Significant Points',marker='x')
plt.title('Step 2: Significant Points Sampling')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step2_Significant_Points_Sampling.png"), dpi=300)
plt.show()
# --------------------
# Step 3: 自适应预算分配，满足w-event隐私
# --------------------

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, min_budget=1e-5):
    """
    基于w-event隐私的自适应预算分配，加入最小预算阈值
    :param slopes: 显著点的斜率
    :param fluctuation_rates: 显著点的波动率
    :param total_budget: 总隐私预算
    :param w: w-event窗口大小
    :param min_budget: 最小隐私预算阈值，确保所有点分配到非零预算
    :return: 分配给每个显著点的隐私预算
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    remaining_budget = total_budget

    for i in range(num_points):
        # 计算当前w窗口内已使用的预算
        used_budget = np.sum(allocated_budgets[max(0, i - w + 1):i])

        # 当前可用的预算 (ε')
        current_budget = remaining_budget - used_budget
        if current_budget <= 0:
            raise ValueError("隐私预算不足，无法满足w-event隐私要求")

        # 动态分配权重: 考虑斜率与波动率
        slope_weight = 1 - np.exp(-abs(slopes[i]))
        fluctuation_weight = 1 - np.exp(-fluctuation_rates[i])
        combined_weight = (slope_weight + fluctuation_weight) / 2

        # 计算当前点的分配比例，确保不小于最小预算
        epsilon_i = max(combined_weight * current_budget, min_budget)
        allocated_budgets[i] = min(epsilon_i, current_budget)

    return allocated_budgets
# 计算斜率和波动率
slopes = np.gradient(normalized_data[significant_indices])
fluctuation_rates = np.abs(slopes)
total_budget = 1.0
w_window = 10  # 设置w-event窗口大小
allocated_budgets = adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w_window)

# 可视化预算分配
plt.figure(figsize=(12, 6))
plt.bar(range(len(allocated_budgets)), allocated_budgets, color='blue')
plt.title('Step 3: Adaptive Budget Allocation with w-event Privacy')
plt.xlabel('Significant Point Index')
plt.ylabel('Privacy Budget')
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step3_Adaptive_Budget_Allocation.png"), dpi=300)
plt.show()

# --------------------
# Step 4: SW扰动机制，基于w-event隐私预算
# --------------------

def sw_perturbation_w_event(values, budgets, min_budget=1e-5, scale_factor=0.5):
    """
    对显著点应用SW机制，满足w-event隐私，减少扰动强度
    :param values: 原始显著点值
    :param budgets: 每个显著点的隐私预算
    :param min_budget: 最小隐私预算阈值
    :param scale_factor: 噪声缩放系数，默认0.5
    :return: 扰动后的显著点值
    """
    perturbed_values = []
    for value, epsilon in zip(values, budgets):
        epsilon = max(epsilon, min_budget)  # 确保预算不为0
        b = scale_factor * (epsilon / (2 * (np.exp(epsilon) - 1)))  # 调整扰动强度
        noisy_value = value + np.random.laplace(scale=b)
        perturbed_values.append(noisy_value)
    return perturbed_values
# 调整扰动强度
perturbed_values = sw_perturbation_w_event(normalized_data[significant_indices], allocated_budgets, scale_factor=0.5)

# 可视化扰动结果
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'].iloc[significant_indices], perturbed_values, 
         label='Perturbed Points', color='red', linestyle='--', marker='x')
plt.plot(hkhs_data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1)
plt.title('Step 4: SW Mechanism Perturbation with w-event Privacy')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step4_SW_Mechanism_Perturbation.png"), dpi=300)
plt.show()
# --------------------
# Step 5: 卡尔曼滤波
# --------------------
def kalman_filter(data, process_variance=1e-4, measurement_variance=1e-2):
    """应用卡尔曼滤波平滑扰动后的数据"""
    n = len(data)
    estimates = np.zeros(n)
    estimates[0] = data[0]
    variance = 1.0
    for t in range(1, n):
        predicted_estimate = estimates[t - 1]
        predicted_variance = variance + process_variance
        kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
        estimates[t] = predicted_estimate + kalman_gain * (data[t] - predicted_estimate)
        variance = (1 - kalman_gain) * predicted_variance
    return estimates

smoothed_values = kalman_filter(perturbed_values)

# 可视化平滑后的结果
plt.figure(figsize=(12, 6))
# 绘制原始显著点（最上层，用蓝色线连接，不显示大圆点，只用最小的点连接）
plt.plot(hkhs_data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1, zorder=3, marker='.', markersize=3)
# 绘制扰动后的显著点
plt.scatter(hkhs_data['date'].iloc[significant_indices], perturbed_values,
            label='Perturbed Points', color='red', marker='x')
# 绘制平滑后的显著点（最底层，用绿色线连接）
plt.plot(hkhs_data['date'].iloc[significant_indices], smoothed_values,
         label='Smoothed Points', color='blue', linestyle='-', linewidth=1, zorder=1)
plt.title('Step 5: Kalman Filter Smoothing')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step5_Kalman_Filter_Smoothing.png"), dpi=300)
plt.show()