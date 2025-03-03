import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import utils.plot_utils as plot_utils
import utils.effiency_utils as effiency_utils
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

''' x和total-budget的实验代码 '''
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

sampling_rates = np.round(np.arange(0.5, 1.0 + 0.05, 0.05), 2)[::-1]  # 降序排列
epsilon_values = np.round(np.arange(0.1, 1.0 + 0.1, 0.1), 1)

def run_single_experiment(x, eps, file_path):
    # 数据预处理
    data, original_values = data_utils.preprocess_HKHS_data(file_path, sample_fraction=x)
    normalized_data = data['sample_normalized_value'].values
    original_data = data['normalized_value'].values

    # 显著点采样
    significant_indices = remarkable_point_sampling(
        normalized_data, 
        original_data, 
        kp=0.7, 
        ks=0.15, 
        kd=0.1, 
        pi=5
    )

    # 特征计算
    slopes = np.gradient(normalized_data[significant_indices])
    fluctuation_rates = np.array([
        calculate_fluctuation_rate(
            errors=np.abs(normalized_data[significant_indices] - original_data[significant_indices]),
            current_idx=i,
            kp=0.8, 
            ks=0.1, 
            kd=0.1, 
            pi=5
        )
        for i in range(len(slopes))
    ])

    # 预算分配
    allocated_budgets = adaptive_w_event_budget_allocation(
        slopes, 
        fluctuation_rates, 
        total_budget=eps,  # 使用当前隐私预算
        w=160
    )

    # 扰动
    perturbed_values = sw_perturbation_w_event(
        values=normalized_data[significant_indices], 
        budgets=allocated_budgets
    )

    # 卡尔曼滤波
    smoothed_values = kalman_filter(
        perturbed_values,
        process_variance=5e-4,
        measurement_variance=5e-3
    )

    # 指标计算
    fitted_values = data_utils.generate_piecewise_linear_curve(
        data['date'], 
        significant_indices, 
        smoothed_values
    )
    dtw = data_utils.calculate_fdtw(original_data, fitted_values)
    mre = data_utils.calculate_mre(smoothed_values, original_data[significant_indices])

    return {
        "sampling_rate": x,
        "epsilon": eps,
        "dtw": dtw,
        "mre": mre
    }

def run_experiments_parallel(file_path):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(x, eps, file_path)
        for x in sampling_rates
        for eps in epsilon_values
    )
    df = pd.DataFrame(results)
    return df

# 可视化函数：三维曲面图
def plot_3d_tradeoff(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        df["sampling_rate"],
        df["epsilon"],
        df["dtw"],
        c=df["mre"],  # 使用MRE作为颜色映射
        cmap="viridis",
        s=50
    )
    fig.colorbar(sc, ax=ax, label="MRE")
    ax.set_xlabel("Sampling Rate (x)")
    ax.set_ylabel("Privacy Budget (ε)")
    ax.set_zlabel("DTW Distance")
    ax.set_title("Privacy-Utility Trade-off: Sampling Rate vs. Epsilon")
    plt.show()

# 可视化函数：二维切片分析（MRE）
def plot_slice_analysis_mre(df, y_min=None, y_max=None):
    plt.figure(figsize=(12, 6))
    for eps in [0.1, 0.5, 1.0]:
        subset = df[df["epsilon"] == eps]
        plt.plot(subset["sampling_rate"], subset["mre"], label=f"MRE (ε={eps})", linestyle="--", marker='s')

    # 高亮采样率在0.7-0.9之间的区域
    plt.axvspan(0.7, 0.9, color='yellow', alpha=0.3, label="Optimal Range (0.7-0.9)")

    # 设置纵坐标范围
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("MRE Value")
    plt.legend()
    plt.title("Impact of Sampling Rate on MRE under Different Privacy Budgets")
    plt.grid()
    plt.show()

# 可视化函数：二维切片分析（DTW）
def plot_slice_analysis_dtw(df, y_min=None, y_max=None):
    plt.figure(figsize=(12, 6))
    for eps in [0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = df[df["epsilon"] == eps]
        plt.plot(subset["sampling_rate"], subset["dtw"], label=f"DTW (ε={eps})", marker='o')

    # 高亮采样率在0.7-0.9之间的区域
    plt.axvspan(0.7, 0.9, color='yellow', alpha=0.3, label="Optimal Range (0.7-0.9)")

    # 设置纵坐标范围
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("DTW Value")
    plt.legend()
    plt.title("Impact of Sampling Rate on DTW under Different Privacy Budgets")
    plt.grid()
    plt.show()
# 可视化函数：热力图（调整范围）
def plot_heatmap(df, metric="dtw"):
    pivot_table = df.pivot(index="epsilon", columns="sampling_rate", values=metric)
    plt.figure(figsize=(12, 8))

    if metric == "mre":
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                    cbar_kws={'label': metric}, vmin=0, vmax=1.0)  # MRE范围调整为0.2-0.4
    else:
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                    cbar_kws={'label': metric}, vmin=0, vmax=4000)  # DTW范围调整为0.0-1.0

    plt.title(f"Heatmap of {metric.upper()} for Sampling Rate vs. Privacy Budget")
    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("Privacy Budget (ε)")
    plt.show()

# 主程序
if __name__ == "__main__":
    file_path = "../data/HKHS.csv"
    df_results = run_experiments_parallel(file_path)
    
    # 可视化
    plot_3d_tradeoff(df_results)
    
    # 自定义MRE的纵坐标范围为[0.2, 0.4]
    plot_slice_analysis_mre(df_results, y_min=0.1, y_max=1.0)
    
    # 自定义DTW的纵坐标范围为[300, 1500]
    plot_slice_analysis_dtw(df_results, y_min=0, y_max=4000)
    
    # 热力图
    plot_heatmap(df_results, metric="dtw")  # DTW热力图
    plot_heatmap(df_results, metric="mre")  # MRE热力图