import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

# 设置随机种子以确保可重复性
np.random.seed(42)

file_path = 'data/HKHS.csv'  # 请确保CSV文件路径正确
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

current_date = datetime.now().strftime('%Y%m%d')

# =========================================================================
# Step 1: 数据预处理（增加 sample_fraction 参数）
# =========================================================================
def preprocess_data(file_path, sample_fraction=1.0):
    """
    读取数据并进行归一化处理。
    sample_fraction: 采样比例，取值范围 (0, 1]。
                     默认为1.0表示使用全部数据，
                     0.8表示随机采样80%的数据。
    """
    data = pd.read_csv(file_path)
    # 将日期转换为日期格式，并排序
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    
    # ---------- 新增：随机采样（若 sample_fraction < 1.0） ----------
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42)
        # 采样后再次按日期排序
        data = data.sort_values(by='date')
    # ---------- 新增结束 --------------------------------------------
    
    # 归一化处理
    # 假设原始列名是 " value"，请注意列名前面的空格
    data['normalized_value'] = (data[' value'] - data[' value'].min()) / (
        data[' value'].max() - data[' value'].min())
    
    return data, data[' value'].values  # 返回 (DataFrame, 原始值数组)


# 在此我们只需要把 sample_fraction 改成 0.8 即可
SAMPLE_FRACTION = 0.8  # 这里设置为 0.8，表示只取 80% 的数据进行后续流程
hkhs_data, original_data = preprocess_data(file_path, sample_fraction=SAMPLE_FRACTION)
normalized_data = hkhs_data['normalized_value'].values

plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')
plt.title(f'Step 1: Normalized HKHS Index (sample_fraction={SAMPLE_FRACTION})')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step1_Normalized_HKHS_Index.png"), dpi=300)
plt.show()


# =========================================================================
# Step 2: 显著点采样 (与之前算法保持一致)
# =========================================================================
def dynamic_lambda(slope, fluctuation_rate):
    return 1 - np.exp(-1 / (abs(slope) + fluctuation_rate + 1e-8))

def calculate_dynamic_threshold(slope, fluctuation_rate, min_alpha=0.05, max_alpha=0.8):
    lambda_value = dynamic_lambda(slope, fluctuation_rate)
    alpha = lambda_value * np.pi / 2
    return np.clip(alpha, min_alpha, max_alpha)

def adaptive_significant_points(data, max_search=10, min_alpha=0.2, max_alpha=1.0, 
                                slope_diff_threshold=0.1, min_distance=5):
    significant_points = [0]
    i = 0
    fluctuation_rate = 0

    while i < len(data) - 2:
        start = i
        found_significant_point = False
        diff = data[i + 1:] - data[i]
        
        # 1) 简单间距策略
        for j in range(len(diff)):
            candidate = i + j + 1
            if candidate - significant_points[-1] >= min_distance:
                significant_points.append(candidate)
                i = candidate
                found_significant_point = True
                break

        if found_significant_point:
            continue

        # 2) 基于斜率差异的自适应策略
        for j in range(i + 1, min(i + max_search, len(data))):
            x = np.arange(start, j + 1)
            y = data[start:j + 1]
            slope, _ = np.polyfit(x, y, 1)

            if j + 1 < len(data):
                next_slope = (data[j + 1] - data[j])
                tan_theta = abs((next_slope - slope) / (1 + slope * next_slope + 1e-8))
            else:
                break

            alpha = calculate_dynamic_threshold(slope, fluctuation_rate, min_alpha, max_alpha)
            threshold = max(np.tan(alpha), slope_diff_threshold)

            if tan_theta > threshold:
                candidate = j
                if candidate - significant_points[-1] >= min_distance:
                    significant_points.append(candidate)
                    i = candidate
                    found_significant_point = True
                    break

        if not found_significant_point:
            i += 1
            significant_points.append(i)

        if j + 1 < len(data):
            fluctuation_rate = abs(next_slope - slope)

    significant_points.append(len(data) - 1)
    return significant_points

significant_indices = adaptive_significant_points(normalized_data)

# --------------------
# Step 2.1: 显著点采样可视化
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')
plt.scatter(hkhs_data['date'].iloc[significant_indices], 
            normalized_data[significant_indices],
            color='red', label='Significant Points', marker='x')
plt.title('Step 2.1: Significant Points Sampling')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step2_1_Significant_Points_Sampling.png"), dpi=300)
plt.show()


# --------------------
# Step 2.2: 显著点线性拟合可视化 (改进版)
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')

for k in range(len(significant_indices) - 1):
    start_idx = significant_indices[k]
    end_idx = significant_indices[k + 1]
    
    x_numeric = np.arange(start_idx, end_idx + 1)
    y_segment = normalized_data[start_idx:end_idx + 1]
    
    slope, intercept = np.polyfit(x_numeric, y_segment, 1)
    fitted_y = slope * x_numeric + intercept
    
    x_dates = hkhs_data['date'].iloc[start_idx:end_idx + 1]
    line_label = 'Fitted Lines Between Significant Points' if k == 0 else None
    
    plt.plot(x_dates, fitted_y, color='red', linestyle='-', linewidth=1, label=line_label)

plt.title('Step 2.2: Linear Fit Between Significant Points')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step2_2_Linear_Fit_Significant_Points.png"), dpi=300)
plt.show()

# --------------------
# Step 2.3: 最终拟合结果对比
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'], normalized_data, label='Normalized Data')

for k in range(len(significant_indices) - 1):
    start_idx = significant_indices[k]
    end_idx = significant_indices[k + 1]
    
    x = np.arange(start_idx, end_idx + 1)
    y = normalized_data[start_idx:end_idx + 1]
    
    slope, intercept = np.polyfit(x, y, 1)
    fitted_y = slope * x + intercept
    
    plt.plot(hkhs_data['date'].iloc[start_idx:end_idx + 1], 
             fitted_y, color='red', linestyle='-', linewidth=1)

plt.scatter(hkhs_data['date'].iloc[significant_indices], 
            normalized_data[significant_indices],
            color='red', label='Significant Points', marker='x')
plt.title('Step 2.3: Original Data vs. Fitted Lines')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step2_3_Original_vs_Fitted.png"), dpi=300)
# plt.show()


# =========================================================================
# Step 3: 自适应预算分配 (与之前算法保持一致)
# =========================================================================
DEFAULT_PK_GAMMA_VALUE = 0.1
MIN_BUDGET_VALUE = 1e-5

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, min_budget=MIN_BUDGET_VALUE):
    """
    基于w-event隐私的自适应预算分配，确保任意w个显著点内总预算不超过total_budget。
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    
    for i in range(num_points):
        window_start = max(0, i - w + 1)
        window_end = i + 1  # Python切片不包括window_end
        current_window = slice(window_start, window_end)
        
        used_budget = np.sum(allocated_budgets[current_window])
        
        remaining_budget = total_budget - used_budget
        if remaining_budget <= 0:
            allocated_budgets[i] = min_budget
            continue
        
        # 动态分配权重: 考虑斜率和波动率
        pk = 1 - np.exp(-abs(slopes[i]))
        pgamma = 1 - np.exp(-fluctuation_rates[i])
        if slopes[i] * fluctuation_rates[i] == 0:
            pk_gamma = DEFAULT_PK_GAMMA_VALUE
        else:
            pk_gamma = 1 - np.exp(-1 / (abs(slopes[i] * fluctuation_rates[i])))

        p = 1 - np.exp(-((pk + pgamma) / (pk_gamma + 1e-8)))
        
        epsilon_i = max(p * remaining_budget, min_budget)
        epsilon_i = min(epsilon_i, remaining_budget)
        
        allocated_budgets[i] = epsilon_i
    
    # 打印分配预算的统计信息
    print("分配的隐私预算统计指标:")
    print(pd.Series(allocated_budgets).describe())
    
    return allocated_budgets

slopes = np.gradient(normalized_data[significant_indices])
fluctuation_rates = np.abs(slopes)

epsilon_total = 1.0  # 每w个显著点的总隐私预算
w_window = 160       # w-event窗口大小

allocated_budgets = adaptive_w_event_budget_allocation(slopes, fluctuation_rates, epsilon_total, w_window)

plt.figure(figsize=(12, 6))
plt.bar(range(len(allocated_budgets)), allocated_budgets, color='blue')
plt.title('Step 3: Adaptive Budget Allocation with w-event Privacy')
plt.xlabel('Significant Point Index')
plt.ylabel('Privacy Budget (epsilon)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"{current_date}_Step3_Adaptive_Budget_Allocation.png"), dpi=300)
plt.show()


# =========================================================================
# Step 4: SW扰动机制 (与之前算法保持一致)
# =========================================================================
def sw_perturbation_w_event(values, budgets, min_budget=1e-5):
    """
    对显著点应用SW机制，满足w-event隐私，按照论文公式扰动
    """
    perturbed_values = []
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

perturbed_values = sw_perturbation_w_event(normalized_data[significant_indices], allocated_budgets)

plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'].iloc[significant_indices], perturbed_values,
         label='Perturbed Points', color='red', linestyle='--', marker='x')
plt.plot(hkhs_data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1)
plt.title('Step 4: SW Mechanism Perturbation with w-event Privacy')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step4_SW_Mechanism_Perturbation.png"), dpi=300)
plt.show()

print("SW扰动后，显著点日期的前几个元素:", hkhs_data['date'].iloc[significant_indices][:5].values)
print("SW扰动后，原始显著点值的前几个元素:", normalized_data[significant_indices][:5])
print("扰动后的值的前几个元素:", perturbed_values[:5])


# =========================================================================
# Step 5: 卡尔曼滤波 (与之前算法保持一致)
# =========================================================================
def kalman_filter(data, process_variance=1e-4, measurement_variance=1e-2):
    """
    应用卡尔曼滤波平滑扰动后的数据
    """
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

plt.figure(figsize=(12, 6))
plt.plot(hkhs_data['date'].iloc[significant_indices], normalized_data[significant_indices],
         label='Original Significant Points', color='green', linestyle='-', linewidth=1, 
         zorder=3, marker='.', markersize=3)
plt.scatter(hkhs_data['date'].iloc[significant_indices], perturbed_values,
            label='Perturbed Points', color='red', marker='x')
plt.plot(hkhs_data['date'].iloc[significant_indices], smoothed_values,
         label='Smoothed Points', color='blue', linestyle='-', linewidth=1, zorder=1)
plt.title('Step 5: Kalman Filter Smoothing')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"{current_date}_Step5_Kalman_Filter_Smoothing.png"), dpi=300)
plt.show()

print("original_data 是否存在缺失值:", pd.isnull(original_data).any())
print("hkhs_data 是否存在缺失值:", pd.isnull(hkhs_data).any())


# =========================================================================
# Step 6: 数据分布统计与可视化 (与之前算法保持一致)
# =========================================================================
plt.figure(figsize=(12, 6))
sns.histplot(normalized_data, color='blue', label='Original Data', kde=True, 
             stat="density", bins=50, alpha=0.5)
sns.histplot(perturbed_values, color='red', label='Perturbed Data', kde=True, 
             stat="density", bins=50, alpha=0.5)
plt.title('Step 6: Data Distribution Before and After Perturbation')
plt.xlabel('Normalized Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"{current_date}_Step6_Data_Distribution_Before_After_Perturbation.png"), dpi=300)
plt.show()

print("扰动前数据统计指标:")
print(pd.Series(normalized_data).describe())
print("\n扰动后数据统计指标:")
print(pd.Series(perturbed_values).describe())