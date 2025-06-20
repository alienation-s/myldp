import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import LBDLDP.LBD as LBD
import PPLDP.ppldp as PPLDP
import PatternLDP.patternLDP as patternLDP

# sampling_rates = np.round(np.arange(0.50, 1.00 + 0.05, 0.05), 2)  # 0.2~1.0, 步长0.1
# sampling_rates = np.round(np.arange(0.50, 1.00 + 0.05, 0.05))  # 0.2~1.0, 步长0.1
# epsilon_values = np.round(np.arange(10.0, 10.0 + 0.5, 0.5))  # 0.5~5.0, 步长0.5
# sampling_rates = np.array([1.0])  # 取样率只需要1.0
# epsilon_values = np.array([5.0])  # 取样率只需要1.0
# epsilon_values = np.round(np.arange(5.0, 10.0 + 0.5, 0.5)) 
###########################################
# 2) 实验并行函数 (修改: 重复10次取平均)
###########################################
def run_experiments_parallel(file_path,
                             run_single_experiment,
                             sampling_rates=None,
                             epsilon_values=None,
                             w_values=None,
                             n_runs=10):
    """
    并行执行多组实验，参数组合包括 sampling_rate、epsilon 和 窗口大小 w。
    返回 DataFrame，列: ['sampling_rate','epsilon','w','dtw','mre','runtime','peak_memory']
    """
    # 默认值
    if sampling_rates is None:
        # sampling_rates = np.arange(0.5, 1.01, 0.1).tolist()  # 0.8~1.0, 步长0.05
        sampling_rates = [0.6,0.7,0.8,0.9,1.0]  # 0.8~1.0, 步长0.05
        # sampling_rates = [0.8]  # 0.8~1.0, 步长0.05
    if epsilon_values is None:
        # epsilon_values = np.arange(0.1, 1.1 , 0.3).tolist()
        epsilon_values = [10]  # 0.1~1.0, 步长0.1
    if w_values is None:
        # w_values = np.arange(50, 151, 50).tolist()
        w_values = [150]

    # 三重循环任务：sampling_rate × epsilon × w
    tasks = [
        (x, eps, w)
        for x in sampling_rates
        for eps in epsilon_values
        for w in w_values
    ]

    def remove_outliers(arr, scale=3.0):
        arr = np.array(arr)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-8:
            return arr
        filtered = arr[np.abs(arr - med) <= scale * mad]
        return filtered if len(filtered) > 0 else arr

    results = []
    for x, eps, w in tqdm(tasks, desc="Running Experiments", total=len(tasks)):
        # 对每组参数重复 n_runs 次
        runs = Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(x, eps, file_path, w)
            for _ in range(n_runs)
        )

        # 收集各项指标
        dtw_list         = [r["dtw"]         for r in runs]
        mre_list         = [r["mre"]         for r in runs]
        runtime_list     = [r["runtime"]     for r in runs]
        peak_mem_list    = [r["peak_memory"] for r in runs]

        results.append({
            "sampling_rate": x,
            "epsilon":      eps,
            "w":            w,
            "dtw":          remove_outliers(dtw_list).mean(),
            "mre":          remove_outliers(mre_list).mean(),
            "runtime":      remove_outliers(runtime_list).mean(),
            "peak_memory":  remove_outliers(peak_mem_list).mean(),
        })

    return pd.DataFrame(results)

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
    # 从 df 中动态获取 epsilon 值
    unique_epsilons = df["epsilon"].unique()
    for eps in unique_epsilons:
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
    # 从 df 中动态获取 epsilon 值
    unique_epsilons = df["epsilon"].unique()
    for eps in unique_epsilons:
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
    common_sampling_rates = sorted(list(s1  & s2 & s3))
    print("Common Sampling Rates:", common_sampling_rates)

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


def create_reduction_table(df, method_name):
    """
    给定一个包含 [sampling_rate, epsilon, mre, runtime] 等列的 DataFrame，
    以 sampling_rate = 1.0 情况下的 (mre, runtime) 作为基准，计算：
        - mre_reduction = (mre_ref - mre) / mre_ref
        - runtime_reduction = (runtime_ref - runtime) / runtime_ref
    并额外添加一列 method = method_name，方便后续拼接对比。
    最终输出按 (sampling_rate, epsilon, w, method) 聚合唯一。
    """
    # 1) 提取基准行（sampling_rate = 1.0）
    df_ref = df[df["sampling_rate"] == 1.0].copy()

    # 聚合处理，确保每个 epsilon 只有一行基准
    df_ref = (
        df_ref.groupby("epsilon", as_index=False)
        .agg({
            "mre": "mean",
            "runtime": "mean"
        })
        .rename(columns={
            "mre": "mre_ref",
            "runtime": "runtime_ref"
        })
    )

    # 2) 合并参考值
    df_merged = pd.merge(
        df,
        df_ref,  # epsilon 对应的 baseline mre/runtime
        on="epsilon",
        how="left"
    )

    # 3) 计算相对提升（注意处理除零）
    df_merged["mre_reduction"] = (df_merged["mre_ref"] - df_merged["mre"]) / df_merged["mre_ref"].replace(0, np.nan)
    df_merged["runtime_reduction"] = (df_merged["runtime_ref"] - df_merged["runtime"]) / df_merged["runtime_ref"].replace(0, np.nan)

    # 4) 添加方法列
    df_merged["method"] = method_name

    # 5) 聚合去重，确保 (x, ε, w, method) 唯一
    df_unique = (
        df_merged.groupby(["sampling_rate", "epsilon", "w", "method"], as_index=False)
        .agg({
            "dtw": "mean",
            "mre": "mean",
            "runtime": "mean",
            "peak_memory": "mean",
            "mre_reduction": "mean",
            "runtime_reduction": "mean"
        })
    )

    return df_unique


###########################################
# 主程序
###########################################
if __name__ == "__main__":
    # 你可以在此修改输入数据与输出目录
    # file_path = "data/LD.csv"
    # file_path = "data/heartrate.csv"
    # file_path = "data/HKHS.csv"   
    # file_path = "data/ETTh1.csv" #可用！！！电力变压器温度 (ETT) 是电力长期部署的关键指标。该数据集由来自中国两个分离县的2年数据组成。为了探索长序列时间序列预测 (LSTF) 问题的粒度，创建了不同的子集，{ETTh1，ETTh2} 为1小时级，ETTm1为15分钟级。每个数据点由目标值 “油温” 和6个功率负载特征组成。火车/val/测试为12/4/4个月。https://opendatalab.com/OpenDataLab/ETT
    file_path = "data/exchange_rate.csv"
    # file_path = "data/national_illness.csv"
    # file_path = "data/weather.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 下面设置 n_runs=10，表示每个 (sampling_rate, epsilon) 组合重复运行10次再取平均
    print("Running LBD Experiments...")
    df_lbd = run_experiments_parallel(file_path, LBD.run_single_experiment)

    print("Running PPLDP Experiments...")
    df_ppldp = run_experiments_parallel(file_path, PPLDP.run_single_experiment)

    print("Running PatternLDP Experiments...")
    df_pldp = run_experiments_parallel(file_path, patternLDP.run_single_experiment)

    # 下面分别为三种方法画图
    plot_slice_analysis_mre(df_lbd,   "LBD",       output_dir="results/comparison/LBD")
    plot_slice_analysis_dtw(df_lbd,   "LBD",       output_dir="results/comparison/LBD")

    plot_slice_analysis_mre(df_ppldp, "PPLDP",     output_dir="results/comparison/PPLDP")
    plot_slice_analysis_dtw(df_ppldp, "PPLDP",     output_dir="results/comparison/PPLDP")

    plot_slice_analysis_mre(df_pldp,  "PatternLDP", output_dir="results/comparison/PatternLDP")
    plot_slice_analysis_dtw(df_pldp,  "PatternLDP", output_dir="results/comparison/PatternLDP")

    # 如果需要“相同采样率各自对比”：
    res = [df_lbd, df_ppldp, df_pldp]
    plot_comparison_for_sampling_rates(res, output_dir="results/comparison")

    print("All done! Check the 'results/comparison/' folder for figures.")

    # # 生成减少率/提高率表格
    df_lbd_table   = create_reduction_table(df_lbd,   method_name="LBD")
    df_ppldp_table = create_reduction_table(df_ppldp, method_name="PPLDP")
    df_pldp_table  = create_reduction_table(df_pldp,  method_name="PatternLDP")

    # 仅演示打印 LBD 的表格:
    print("=== ppldp: Reduction Table (Head) ===")
    print(df_ppldp_table[[
        "method", 
        "sampling_rate", 
        "epsilon", 
        "w",
        "dtw",
        "mre", 
        "peak_memory",
        "runtime", 
        "mre_reduction", 
        "runtime_reduction"
    ]].head(10))

    # # 保存结果
    df_lbd_table.to_csv(os.path.join(output_dir, "LBD.csv"), index=False)
    df_ppldp_table.to_csv(os.path.join(output_dir, "PPLDP.csv"), index=False)
    df_pldp_table.to_csv(os.path.join(output_dir, "PatternLDP.csv"), index=False)

    print("Reduction tables saved in:", output_dir)