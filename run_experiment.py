#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# 并行相关
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm

# 可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ====== 这三行假设你已经有三个模块 LBD, PPLDP, patternLDP ======
#       模块里实现了 run_single_experiment(sampling_rate, epsilon, file_path)
import LBD.LBD as LBD
import myLDP.PPLDP as PPLDP
import PatternLDP.patternLDP as patternLDP

###########################################
# 1) 定义实验参数
###########################################
sampling_rates = np.round(np.arange(0.1, 1.0 + 0.05, 0.05), 2)[::-1]  # 0.1~1.0, 每0.05, 再逆序
epsilon_values = np.round(np.arange(0.1, 1.0 + 0.1, 0.1), 1)         # 0.1~1.0, 每0.1

###########################################
# 2) 实验并行函数
###########################################
def run_experiments_parallel(file_path, run_single_experiment):
    """
    并行运行实验，并使用 tqdm 监控进度。
    run_single_experiment: 某个方法的单次实验函数
    返回：包含 (sampling_rate, epsilon, dtw, mre, ...) 的 DataFrame
    """
    tasks = [(x, eps, file_path) for x in sampling_rates for eps in epsilon_values]
    results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(x, eps, file_path)
        for x, eps, file_path in tqdm(tasks, desc="Running Experiments", total=len(tasks))
    )
    df = pd.DataFrame(results)
    return df

###########################################
# 3) 可视化函数：3D 散点图
###########################################
def plot_3d_tradeoff(df, method_name, output_dir="results/comparison"):
    os.makedirs(output_dir, exist_ok=True)

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
    ax.set_title(f"Privacy-Utility Trade-off ({method_name})")

    save_path = os.path.join(output_dir, f"{method_name}_3d_tradeoff.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

###########################################
# 4) 可视化函数：二维切片分析 (MRE)
###########################################
def plot_slice_analysis_mre(df, method_name, output_dir="results/comparison", 
                            y_min=None, y_max=None):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    # 自定义取若干 epsilon
    for eps in [0.1, 0.5, 1.0]:
        subset = df[df["epsilon"] == eps]
        plt.plot(subset["sampling_rate"], subset["mre"], 
                 label=f"MRE (ε={eps})", linestyle="--", marker='s')

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("MRE Value")
    plt.legend()
    plt.title(f"Impact of Sampling Rate on MRE ({method_name})")
    plt.grid()

    save_path = os.path.join(output_dir, f"{method_name}_slice_mre.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

###########################################
# 5) 可视化函数：二维切片分析 (DTW)
###########################################
def plot_slice_analysis_dtw(df, method_name, output_dir="results/comparison", 
                            y_min=None, y_max=None):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    # 自定义取若干 epsilon
    for eps in [0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = df[df["epsilon"] == eps]
        plt.plot(subset["sampling_rate"], subset["dtw"], 
                 label=f"DTW (ε={eps})", marker='o')

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("DTW Value")
    plt.legend()
    plt.title(f"Impact of Sampling Rate on DTW ({method_name})")
    plt.grid()

    save_path = os.path.join(output_dir, f"{method_name}_slice_dtw.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

###########################################
# 6) 可视化函数：热力图
###########################################
def plot_heatmap(df, method_name, metric="dtw", output_dir="results/comparison"):
    os.makedirs(output_dir, exist_ok=True)

    pivot_table = df.pivot(index="epsilon", columns="sampling_rate", values=metric)
    plt.figure(figsize=(12, 8))
    
    if metric == "mre":
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                    cbar_kws={'label': metric}, vmin=0, vmax=1.0)
    else:  # dtw
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                    cbar_kws={'label': metric})
    plt.title(f"Heatmap of {metric.upper()} ({method_name})")
    plt.xlabel("Sampling Rate (x)")
    plt.ylabel("Privacy Budget (ε)")

    save_path = os.path.join(output_dir, f"{method_name}_heatmap_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

###########################################
# 7) 可选：每个采样率一张对比图
###########################################
def plot_comparison_for_sampling_rates(res, output_dir="results/comparison"):
    """
    对三种方法(LBD/PPLDP/PatternLDP)在相同 sampling_rate 下, 
    画出 epsilon vs. (MRE, DTW) 的对比图 (上下两个子图)。
    每个采样率一张图, 不展示, 直接保存。
    
    参数:
      res: List[pd.DataFrame]，长度=3，顺序假定 [LBD, PPLDP, PatternLDP]
      output_dir: 存图目录
    """
    method_names = ["LBD", "PPLDP", "PatternLDP"]
    df_LBD, df_PPLDP, df_PatternLDP = res
    
    s1 = set(df_LBD["sampling_rate"].unique())
    s2 = set(df_PPLDP["sampling_rate"].unique())
    s3 = set(df_PatternLDP["sampling_rate"].unique())
    common_sampling_rates = sorted(list(s1 & s2 & s3))

    os.makedirs(output_dir, exist_ok=True)
    
    for s in common_sampling_rates:
        sub_lbd   = df_LBD[df_LBD["sampling_rate"] == s].copy().sort_values("epsilon")
        sub_ppldp = df_PPLDP[df_PPLDP["sampling_rate"] == s].copy().sort_values("epsilon")
        sub_pldp  = df_PatternLDP[df_PatternLDP["sampling_rate"] == s].copy().sort_values("epsilon")
        
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # 上子图: MRE
        axs[0].plot(sub_lbd["epsilon"], sub_lbd["mre"], label="LBD", marker='o')
        axs[0].plot(sub_ppldp["epsilon"], sub_ppldp["mre"], label="PPLDP", marker='s')
        axs[0].plot(sub_pldp["epsilon"], sub_pldp["mre"], label="PatternLDP", marker='^')
        axs[0].set_ylabel("MRE")
        axs[0].set_title(f"Sampling Rate = {s}")
        axs[0].grid(True)
        axs[0].legend()
        
        # 下子图: DTW
        axs[1].plot(sub_lbd["epsilon"], sub_lbd["dtw"], label="LBD", marker='o')
        axs[1].plot(sub_ppldp["epsilon"], sub_ppldp["dtw"], label="PPLDP", marker='s')
        axs[1].plot(sub_pldp["epsilon"], sub_pldp["dtw"], label="PatternLDP", marker='^')
        axs[1].set_xlabel("Privacy Budget (epsilon)")
        axs[1].set_ylabel("DTW")
        axs[1].grid(True)
        
        save_path = os.path.join(output_dir, f"comparison_samprate_{s}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"Done! Generated {len(common_sampling_rates)} per-sampling-rate comparison figures in '{output_dir}'.")

###########################################
# 主程序
###########################################
if __name__ == "__main__":
    # 你可以在此修改输入数据与输出目录
    file_path = "data/LD.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("Running LBD Experiments...")
    df_lbd = run_experiments_parallel(file_path, LBD.run_single_experiment)

    print("Running PPLDP Experiments...")
    df_ppldp = run_experiments_parallel(file_path, PPLDP.run_single_experiment)

    print("Running PatternLDP Experiments...")
    df_pldp = run_experiments_parallel(file_path, patternLDP.run_single_experiment)

    # 下面分别为三种方法画图
    # (1) LBD
    plot_3d_tradeoff(df_lbd,  "LBD",      output_dir="results/comparison/LBD")
    plot_slice_analysis_mre(df_lbd,   "LBD", output_dir="results/comparison/LBD", y_min=0.1, y_max=2.0)
    plot_slice_analysis_dtw(df_lbd,   "LBD", output_dir="results/comparison/LBD", y_min=0,   y_max=150000)
    plot_heatmap(df_lbd, "LBD", metric="dtw", output_dir="results/comparison/LBD")
    plot_heatmap(df_lbd, "LBD", metric="mre", output_dir="results/comparison/LBD")

    # (2) PPLDP
    plot_3d_tradeoff(df_ppldp,  "PPLDP", output_dir="results/comparison/PPLDP")
    plot_slice_analysis_mre(df_ppldp, "PPLDP", output_dir="results/comparison/PPLDP", y_min=0.1, y_max=2.0)
    plot_slice_analysis_dtw(df_ppldp, "PPLDP", output_dir="results/comparison/PPLDP", y_min=0,   y_max=150000)
    plot_heatmap(df_ppldp, "PPLDP", metric="dtw", output_dir="results/comparison/PPLDP")
    plot_heatmap(df_ppldp, "PPLDP", metric="mre", output_dir="results/comparison/PPLDP")

    # (3) PatternLDP
    plot_3d_tradeoff(df_pldp,  "PatternLDP", output_dir="results/comparison/PatternLDP")
    plot_slice_analysis_mre(df_pldp, "PatternLDP", output_dir="results/comparison/PatternLDP", y_min=0.1, y_max=2.0)
    plot_slice_analysis_dtw(df_pldp, "PatternLDP", output_dir="results/comparison/PatternLDP", y_min=0,   y_max=150000)
    plot_heatmap(df_pldp, "PatternLDP", metric="dtw", output_dir="results/comparison/PatternLDP")
    plot_heatmap(df_pldp, "PatternLDP", metric="mre", output_dir="results/comparison/PatternLDP")

    # 若想要“相同采样率各自对比”：
    res = [df_lbd, df_ppldp, df_pldp]
    plot_comparison_for_sampling_rates(res, output_dir="results/comparison")

    print("All done! Check the 'results/comparison/' folder for figures.")