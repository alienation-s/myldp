import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

# 固定随机种子，保证可重复性
np.random.seed(42)

# ====================== 全局配置 ======================
file_path = 'data/HKHS.csv'   # 请确保CSV文件路径正确
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
current_date = datetime.now().strftime('%Y%m%d')

# 当日期序列不连续时，是否对采样后的数据进行插值再画线？
INTERPOLATE_FOR_LINE_PLOT = False  # True 表示插值后画线；False 表示仅画散点


# ====================== 辅助函数：评估效用 ======================
def jensen_shannon_divergence(p, q, eps=1e-12):
    """
    计算JS散度，需要先对p, q做归一化（概率分布）。
    p, q: 一维概率分布（如直方图归一化后的频率）
    """
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

def evaluate_utility(original_series, perturbed_series, label=""):
    """
    计算一系列效用指标，并返回结果字典:
    - MSE, RMSE
    - MAE, MAPE
    - Pearson相关系数 (Corr)
    - JS散度 (JS-Div)
    """
    # 1) 去除缺失值（若有）
    mask = ~np.isnan(original_series) & ~np.isnan(perturbed_series)
    o = original_series[mask]
    p = perturbed_series[mask]
    
    # 2) 如果有效数据量太少，直接返回NaN
    if len(o) < 2:
        return {
            'label': label,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MAPE(%)': np.nan,
            'Corr': np.nan,
            'JS-Div': np.nan
        }
    
    # 3) 计算 MSE / RMSE
    mse = mean_squared_error(o, p)
    rmse = np.sqrt(mse)
    
    # 4) 计算 MAE / MAPE
    mae = mean_absolute_error(o, p)
    safe_o = np.where(np.abs(o) < 1e-12, 1e-12, o)  # 避免除0
    mape = np.mean(np.abs((o - p) / safe_o)) * 100
    
    # 5) Pearson 相关系数
    corr = np.corrcoef(o, p)[0, 1]
    
    # 6) JS 散度 (通过直方图频率)
    hist_orig, bin_edges = np.histogram(o, bins=50, density=True)
    hist_pert, _         = np.histogram(p, bins=bin_edges, density=True)
    
    if hist_orig.sum() < 1e-12 or hist_pert.sum() < 1e-12:
        jsd = np.nan
    else:
        jsd = jensen_shannon_divergence(hist_orig, hist_pert)
    
    results = {
        'label': label,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE(%)': mape,
        'Corr': corr,
        'JS-Div': jsd
    }
    return results

# ====================== PPLDP各步骤函数 ======================
def dynamic_lambda(slope, fluctuation_rate):
    return 1 - np.exp(-1 / (abs(slope) + fluctuation_rate + 1e-8))

def calculate_dynamic_threshold(slope, fluctuation_rate, min_alpha=0.05, max_alpha=0.8):
    lambda_value = dynamic_lambda(slope, fluctuation_rate)
    alpha = lambda_value * np.pi / 2
    return np.clip(alpha, min_alpha, max_alpha)

def adaptive_significant_points(data, max_search=10, min_alpha=0.2, max_alpha=1.0, 
                                slope_diff_threshold=0.1, min_distance=5):
    """
    显著点采样算法示例，可根据需要进行改进。
    """
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

DEFAULT_PK_GAMMA_VALUE = 0.1
MIN_BUDGET_VALUE = 1e-5

def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, min_budget=MIN_BUDGET_VALUE):
    """
    基于w-event隐私的自适应预算分配
    """
    num_points = len(slopes)
    allocated_budgets = np.zeros(num_points)
    
    for i in range(num_points):
        window_start = max(0, i - w + 1)
        window_end = i + 1  
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
    
    print("分配的隐私预算统计指标:")
    print(pd.Series(allocated_budgets).describe())
    
    return allocated_budgets

def sw_perturbation_w_event(values, budgets, min_budget=1e-5):
    """
    SW机制扰动
    """
    perturbed_values = []
    for value, epsilon in zip(values, budgets):
        epsilon = max(epsilon, min_budget)
        
        denominator = 2 * np.exp(epsilon) * (np.exp(epsilon) - 1 - epsilon)
        if denominator <= 1e-10:
            # 避免除0
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

def kalman_filter(data, process_variance=1e-4, measurement_variance=1e-2):
    """
    卡尔曼滤波
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

# ============== 额外函数：对日期做插值，便于画线 ==============
def interpolate_time_series(dates, values, freq='D'):
    """
    dates: numpy array of timestamps
    values: numpy array of values (same length as dates)
    freq: pandas frequency string, e.g. 'D', 'H', 'M' 等
    """
    df = pd.DataFrame({'date': dates, 'val': values})
    df = df.set_index('date').sort_index()
    # 将index设为DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # 按 freq 进行重采样，并对缺失值做插值
    df_resampled = df.resample(freq).mean().interpolate(method='linear')
    return df_resampled.index.values, df_resampled['val'].values

# ====================== 核心流程封装 ======================
def ppldp_pipeline(file_path, sample_fraction=1.0,
                   total_budget=1.0, w_window=160,
                   process_variance=1e-4, measurement_variance=1e-2):
    """
    执行从数据读取 -> 采样 -> 归一化 -> 显著点检测 -> 拟合 -> 预算分配 -> SW扰动 -> 卡尔曼滤波
    返回:
      - dates: 时间戳数组 (采样后)
      - original_vals: 采样后对应的原始值
      - normalized_vals: 归一化后数据
      - final_vals: 最终卡尔曼滤波之后的数据（与 normalized_vals 同长度）
      - significant_indices: 显著点在 [0, len(normalized_vals)-1] 上的索引
    """
    # 1) 读取并排序
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')

    # 2) 随机采样
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42)
        data = data.sort_values(by='date')

    # 3) 原始值 + 归一化
    original_vals = data[' value'].values  # 注意列名前空格
    data['normalized_value'] = (data[' value'] - data[' value'].min()) / (
        data[' value'].max() - data[' value'].min()
    )
    normalized_vals = data['normalized_value'].values
    dates = data['date'].values  # 用于可视化
    
    # -------------------------------
    # Step 2: 显著点检测
    # -------------------------------
    significant_indices = adaptive_significant_points(normalized_vals)

    # -------------------------------
    # Step 3: 自适应预算分配
    # -------------------------------
    slopes = np.gradient(normalized_vals[significant_indices])
    fluctuation_rates = np.abs(slopes)
    allocated_budgets = adaptive_w_event_budget_allocation(slopes,
                                                           fluctuation_rates,
                                                           total_budget=total_budget,
                                                           w=w_window)

    # -------------------------------
    # Step 4: SW扰动
    # -------------------------------
    perturbed_values = sw_perturbation_w_event(normalized_vals[significant_indices],
                                               allocated_budgets)

    # 将扰动结果映射回全体点
    final_array = np.array(normalized_vals, copy=True)
    final_array[significant_indices] = perturbed_values

    # -------------------------------
    # Step 5: 卡尔曼滤波
    # -------------------------------
    smoothed_array = kalman_filter(final_array,
                                   process_variance=process_variance,
                                   measurement_variance=measurement_variance)

    return dates, original_vals, normalized_vals, smoothed_array, significant_indices


# ====================== 对多种采样比例进行对比 ======================
def main_comparison():
    # 设定需要对比的采样比例
    sample_fractions = [0.8, 1.0]
    
    # 用于存放对比结果
    utility_records = []

    for frac in sample_fractions:
        # 1) 执行 PPLDP 流程
        dates, original_vals, norm_vals, final_vals, sig_idx = ppldp_pipeline(
            file_path=file_path,
            sample_fraction=frac,
            total_budget=1.0,
            w_window=160,
            process_variance=1e-4,
            measurement_variance=1e-2
        )
        
        # 2) 评估 (与采样后的原始真值对比)
        metrics = evaluate_utility(original_vals, final_vals, label=f"sample={frac}")
        utility_records.append(metrics)

        # ============ 可视化 1：时间序列对比 ============
        plt.figure(figsize=(10, 5))
        
        if INTERPOLATE_FOR_LINE_PLOT:
            # 如果需要插值来画平滑曲线
            interp_dates_orig, interp_vals_orig = interpolate_time_series(dates, original_vals, freq='D')
            interp_dates_final, interp_vals_final = interpolate_time_series(dates, final_vals, freq='D')
            
            plt.plot(interp_dates_orig, interp_vals_orig, label='Original (Interp)', color='black')
            plt.plot(interp_dates_final, interp_vals_final, label='PPLDP Final (Interp)', color='red', linestyle='--')
            
            # 显著点位置：在原有坐标上散点标注
            plt.scatter(dates[sig_idx], final_vals[sig_idx],
                        color='blue', marker='x', label='Significant Points', zorder=5)
            
            plt.title(f"Time Series w/ Interpolation (sample_fraction={frac})")
        else:
            # 仅散点图表示
            plt.scatter(dates, original_vals, label='Original Data', color='black', s=8)
            plt.scatter(dates, final_vals, label='PPLDP Final', color='red', s=8, alpha=0.7)
            
            # 显著点用更醒目的标记
            plt.scatter(dates[sig_idx], final_vals[sig_idx],
                        color='blue', marker='x', s=50, label='Significant Points', zorder=5)
            
            plt.title(f"Time Series (Scatter) (sample_fraction={frac})")

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{current_date}_TimeSeries_{frac}.png"), dpi=300)
        plt.show()

        # ============ 可视化 2：分布对比 (直方图+KDE) ============
        plt.figure(figsize=(10, 5))
        sns.histplot(original_vals, color='blue', label='Original', kde=True, 
                     stat="density", bins=50, alpha=0.5)
        sns.histplot(final_vals, color='red', label='PPLDP Final', kde=True, 
                     stat="density", bins=50, alpha=0.5)
        plt.title(f"Distribution Comparison (sample_fraction={frac})")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{current_date}_Distribution_{frac}.png"), dpi=300)
        plt.show()

    # 3) 汇总各项指标到一个 DataFrame
    df_utility = pd.DataFrame(utility_records)
    print("\n===== Utility Comparison Table =====")
    print(df_utility)
    
    # 保存为CSV
    df_utility.to_csv(os.path.join(output_dir, 
        f"{current_date}_Utility_Comparison.csv"), index=False)


# ====================== 运行入口 ======================
if __name__ == "__main__":
    main_comparison()