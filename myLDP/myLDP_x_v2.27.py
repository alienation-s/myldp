import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
from joblib import Parallel, delayed
import seaborn as sns
import myLDP.PPLDP as PPLDP
''' x和total-budget的实验代码 '''
sampling_rates = np.round(np.arange(0.1, 1.0 + 0.05, 0.05), 2)[::-1]  # 降序排列
epsilon_values = np.round(np.arange(0.1, 1.0 + 0.1, 0.1), 1)

def run_single_experiment(x, eps, file_path):
    sample_data, origin_data = data_utils.preprocess_heartrate_data(
        file_path, 
        x,
    ) # ['date', 'normalized_value'] 两个都是这样的格式
    # sample_data, origin_data = data_utils.preprocess_HKHS_data(
    #     file_path, 
    #     x,
    # ) # ['date', 'normalized_value'] 两个都是这样的格式
    sample_normalized_data = sample_data['normalized_value'].values
    sample_data_length = len(sample_normalized_data)
    origin_normalized_data = origin_data['normalized_value'].values
    origin_data_length = len(sample_normalized_data)
    
    # 显著点采样（使用优化后的版本）
    significant_indices = PPLDP.remarkable_point_sampling(
        sample_normalized_data, 
        kp=0.8, 
        ks=0.1, 
        kd=0.1
    ) # 获取显著点的索引，是采样后的数据的显著点索引
    
    # 特征计算优化
    slopes = np.gradient(sample_normalized_data[significant_indices])
    fluctuation_rates = np.abs(slopes) + 1e-8  # 避免零值
    
    # 预算分配（使用优化后的版本）
    allocated_budgets = PPLDP.adaptive_w_event_budget_allocation(
        slopes, fluctuation_rates, x, 160, sample_data_length
    )
    
    # 扰动（向量化版本）
    perturbed_values = PPLDP.sw_perturbation_w_event(
        sample_normalized_data, allocated_budgets
    )
    
    # 卡尔曼滤波（优化版本）
    smoothed_values = PPLDP.kalman_filter(
        perturbed_values, process_variance=5e-4, measurement_variance=5e-3
    )
    sample_data["smoothed_value"] = smoothed_values

     # 插值
    interpolated_data = data_utils.interpolate_missing_points(origin_data, sample_data)
    interpolated_values = interpolated_data['smoothed_value']
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
    # plt.axvspan(0.7, 0.9, color='yellow', alpha=0.3, label="Optimal Range (0.7-0.9)")

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
    # plt.axvspan(0.7, 0.9, color='yellow', alpha=0.3, label="Optimal Range (0.7-0.9)")

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
                    cbar_kws={'label': metric}, vmin=0, vmax=100000)  # DTW范围调整为0.0-1.0

    plt.title(f"Heatmap of {metric.upper()} for Sampling Rate vs. Privacy Budget")
    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("Privacy Budget (ε)")
    plt.show()

# 主程序
if __name__ == "__main__":
    # file_path = "data/HKHS.csv"
    file_path = 'data/heartrate.csv'

    df_results = run_experiments_parallel(file_path)
    
    # 可视化
    plot_3d_tradeoff(df_results)
    
    plot_slice_analysis_mre(df_results, y_min=0.1, y_max=1.0)
    
    plot_slice_analysis_dtw(df_results, y_min=0, y_max=10000)
    
    # 热力图
    plot_heatmap(df_results, metric="dtw")  # DTW热力图
    plot_heatmap(df_results, metric="mre")  # MRE热力图