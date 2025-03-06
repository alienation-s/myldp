import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import LBD.LBD as LBD

# =========================================================================
# 统一接口调用
def compare_experiments(file_path, output_dir, target):
    """
    调用统一接口进行不同变量的实验，返回并对比结果。
    """
    sample_fraction = 1.0
    if target == "sample_fraction":
        # 实验 0: 只改变数据量
        sample_fractions = np.arange(0.5, 1.05, 0.05)  # 生成从 0.5 到 1.0，步长为 0.05 的数组
        results = []
        for sample_fraction in sample_fractions:
            result_sample = LBD.run_experiment(
                file_path, 
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=1.0, 
                w=160, 
                DTW_MRE=True)
            print(f"DTW for sample fraction {sample_fraction}: {result_sample['dtw_distance']}, MRE for sample fraction {sample_fraction}: {result_sample['mre']}")
            results.append(result_sample)
    elif target == "e":
        # 实验 1: 只改变隐私预算
        es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # es = [1,2,3,4,5,6,7,8,9,10]
        results = []
        for e in es:
            result_budget = LBD.run_experiment(
                file_path,
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=e, 
                w=160, 
                DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        # 实验 2: 只改变窗口大小
        ws = [80,100,120,140,160,180,200,220,240,260]
        results = []
        for w in ws:
            result_window = LBD.run_experiment(
                file_path, 
                output_dir,
                sample_fraction=sample_fraction, 
                total_budget=1.0,
                w=w,
                DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)
    return results

if __name__ == "__main__":
    file_path = 'data/HKHS.csv'  # 输入数据路径
    file_path = 'data/heartrate.csv'  # 输入数据路径
    file_path = 'data/LD.csv'  # 输入数据路径
    output_dir = 'results'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    compare_experiments(file_path, output_dir, target="sample_fraction")
    # compare_experiments(file_path, output_dir, target="e")
    # compare_experiments(file_path, output_dir, target="w")