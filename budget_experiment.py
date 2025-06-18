import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import LBDLDP.LBD as LBD
import PPLDP.ppldp as PPLDP
import PatternLDP.patternLDP as patternLDP

# 2) 实验并行函数 (修改: 重复10次取平均)
def run_experiments_parallel(file_path,
                             run_single_experiment,
                             sample_method="uniform",
                             sampling_rates=None,
                             epsilon_values=None,
                             w_values=None,
                             n_runs=1):
    if sampling_rates is None:
        # sampling_rates = np.arange(0.5, 1.01, 0.1).tolist()  # 0.8~1.0, 步长0.05
        # sampling_rates = [0.6,0.7,0.8,0.9,1.0]  # 0.8~1.0, 步长0.05
        sampling_rates = [0.8, 1.0]  # 0.8~1.0, 步长0.05
    if epsilon_values is None:
        # epsilon_values = np.arange(0.1, 1.1 , 0.3).tolist()
        epsilon_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # 0.1~1.0, 步长0.1
    if w_values is None:
        # w_values = np.arange(50, 151, 50).tolist()
        w_values = [160]

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
            delayed(run_single_experiment)(x, eps, file_path, w, sample_method)
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

# 主程序
if __name__ == "__main__":
    file_paths = ["data/LD.csv","data/ETTh1.csv","data/exchange_rate.csv","data/weather.csv"]
    for file_path in file_paths:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        # 'uniform', 'reservoir', 'stratified'
        sample_method = "uniform"
        # 下面设置 n_runs=10，表示每个 (sampling_rate, epsilon) 组合重复运行10次再取平均
        print("Running LBD Experiments...")
        df_lbd = run_experiments_parallel(file_path, LBD.run_single_experiment,sample_method)

        print("Running PPLDP Experiments...")
        df_ppldp = run_experiments_parallel(file_path, PPLDP.run_single_experiment,sample_method)

        print("Running PatternLDP Experiments...")
        df_pldp = run_experiments_parallel(file_path, patternLDP.run_single_experiment,sample_method)


        # 为每个 DataFrame 添加方法标签
        df_lbd["method"] = "LBD"
        df_ppldp["method"] = "PPLDP"
        df_pldp["method"] = "PatternLDP"

        # 添加采样方式标记（便于后续统一分析）
        df_lbd["sampling_method"] = sample_method
        df_ppldp["sampling_method"] = sample_method
        df_pldp["sampling_method"] = sample_method

        # 合并为一个总表
        df_all = pd.concat([df_lbd, df_ppldp, df_pldp], ignore_index=True)

        # 保存单一文件
        df_all.to_csv(f"results/budget/{file_path}_budget_results.csv", index=False)