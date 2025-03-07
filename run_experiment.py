import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import LBD.LBD as LBD
import myLDP.PPLDP as PPLDP
import PatternLDP.patternLDP as patternLDP

sampling_rates = np.round(np.arange(0.2, 1.0 + 0.1, 0.1), 2)  # 0.2~1.0, 步长0.1
epsilon_values = np.round(np.arange(0.5, 5.0 + 0.5, 0.5), 1)  # 0.5~5.0, 步长0.5

###########################################
# 2) 实验并行函数 (修改: 重复10次取平均)
###########################################
def run_experiments_parallel(file_path, run_single_experiment, n_runs=10):
    """
    并行运行实验 ( sampling_rate x epsilon )，每个组合重复n_runs次取平均。
    
    参数：
      file_path: 数据路径
      run_single_experiment: 某个方法的“单次”实验函数
      n_runs: 对同一参数组合重复的次数 (默认10)
    
    返回：
      包含多行 (sampling_rate, epsilon, dtw, mre, runtime) 的 DataFrame，每行是平均结果。
    """
    tasks = [(x, eps) for x in sampling_rates for eps in epsilon_values]
    
    results = []
    # 这里的 tqdm 用来显示 (parameter-combinations) 级别的进度
    for x, eps in tqdm(tasks, desc="Running Experiments", total=len(tasks)):
        # 对于每个 (sampling_rate, epsilon)，调用 run_single_experiment N 次
        runs_output = Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(x, eps, file_path) 
            for _ in range(n_runs)
        )
        # runs_output 是一个列表，包含 n_runs 个返回结果
        # 每个返回结果通常形如 { "sampling_rate": x, "epsilon": eps, "dtw": val, "mre": val, "runtime": val }

        # 把对应的 dtw, mre, runtime 收集起来做平均
        dtw_list = [r["dtw"] for r in runs_output]
        mre_list = [r["mre"] for r in runs_output]
        runtime_list = [r["runtime"] for r in runs_output]

        results.append({
            "sampling_rate": x,
            "epsilon": eps,
            "dtw": np.mean(dtw_list),
            "mre": np.mean(mre_list),
            "runtime": np.mean(runtime_list),
        })
        
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
    # 自定义取若干 epsilon 进行展示
    for eps in [1.0, 2.5, 5.0]:
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
    for eps in [1.0, 2.5, 5.0]:
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


def create_reduction_table(df, method_name):
    """
    给定一个包含 [sampling_rate, epsilon, mre, runtime] 等列的 DataFrame，
    以 sampling_rate = 1.0 (不采样) 情况下的 (mre, runtime) 作为基准，计算：
        - mre_reduction = (mre_ref - mre) / mre_ref
        - runtime_reduction = (runtime_ref - runtime) / runtime_ref

    并额外添加一列 method = method_name，方便后续拼接对比。
    返回新增列的 DataFrame。
    """
    # 1) 找到 sampling_rate=1.0 的基准行
    df_ref = df[df["sampling_rate"] == 1.0].copy()
    df_ref = df_ref.rename(columns={
        "mre": "mre_ref",
        "runtime": "runtime_ref"
    })

    # 2) 把基准值合并回去（按照同样的 epsilon 来合并）
    df_merged = pd.merge(
        df,
        df_ref[["epsilon", "mre_ref", "runtime_ref"]],  # 只保留参考列用于合并
        on="epsilon",
        how="left"
    )

    # 3) 计算降低比例
    df_merged["mre_reduction"] = (df_merged["mre_ref"] - df_merged["mre"]) / df_merged["mre_ref"]
    df_merged["runtime_reduction"] = (df_merged["runtime_ref"] - df_merged["runtime"]) / df_merged["runtime_ref"]

    # 4) 新增一列表示是哪个方法
    df_merged["method"] = method_name

    return df_merged


###########################################
# 主程序
###########################################
if __name__ == "__main__":
    # 你可以在此修改输入数据与输出目录
    file_path = "data/LD.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 下面设置 n_runs=10，表示每个 (sampling_rate, epsilon) 组合重复运行10次再取平均
    print("Running LBD Experiments...")
    df_lbd = run_experiments_parallel(file_path, LBD.run_single_experiment, n_runs=10)

    print("Running PPLDP Experiments...")
    df_ppldp = run_experiments_parallel(file_path, PPLDP.run_single_experiment, n_runs=10)

    print("Running PatternLDP Experiments...")
    df_pldp = run_experiments_parallel(file_path, patternLDP.run_single_experiment, n_runs=10)

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

    # 生成减少率/提高率表格
    df_lbd_table   = create_reduction_table(df_lbd,   method_name="LBD")
    df_ppldp_table = create_reduction_table(df_ppldp, method_name="PPLDP")
    df_pldp_table  = create_reduction_table(df_pldp,  method_name="PatternLDP")

    # 仅演示打印 LBD 的表格:
    print("=== LBD: Reduction Table (Head) ===")
    print(df_lbd_table[[
        "method", 
        "sampling_rate", 
        "epsilon", 
        "mre", 
        "runtime", 
        "mre_reduction", 
        "runtime_reduction"
    ]].head(10))

    # 保存结果
    df_lbd_table.to_csv(os.path.join(output_dir, "LBD_reduction_table.csv"), index=False)
    df_ppldp_table.to_csv(os.path.join(output_dir, "PPLDP_reduction_table.csv"), index=False)
    df_pldp_table.to_csv(os.path.join(output_dir, "PatternLDP_reduction_table.csv"), index=False)

    print("Reduction tables saved in:", output_dir)