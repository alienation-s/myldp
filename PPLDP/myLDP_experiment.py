import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import PPLDP.ppldp as PPLDP
import numpy as np
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
            result_sample = PPLDP.run_experiment(
                file_path, 
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=1.0, 
                w=50, 
                DTW_MRE=True)
            print(f"DTW for sample fraction {sample_fraction:.2f}: {result_sample['dtw_distance']:.2f}, MRE for sample fraction {sample_fraction:.2f}: {result_sample['mre']:.8f}")
            results.append(result_sample)
    elif target == "e":
        # 实验 1: 只改变隐私预算
        # es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # es = [1,2,3,4,5,6,7,8,9,10]
        es = [5, 10, 20]
        results = []
        for e in es:
            result_budget = PPLDP.run_experiment(
                file_path,
                output_dir, 
                sample_fraction=sample_fraction, 
                total_budget=e, 
                w=50, 
                DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        # 实验 2: 只改变窗口大小
        ws = [80,100,120,140,160,180,200,220,240,260]
        results = []
        for w in ws:
            result_window = PPLDP.run_experiment(
                file_path, 
                output_dir,
                sample_fraction=sample_fraction, 
                total_budget=1.0,
                w=w,
                DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)

if __name__ == "__main__":
    file_path = 'data/HKHS.csv'
    # file_path = 'data/heartrate.csv'
    # file_path = 'data/LD.csv'
    # file_path = "data/ETTh1.csv" #可用！！！电力变压器温度 (ETT) 是电力长期部署的关键指标。该数据集由来自中国两个分离县的2年数据组成。为了探索长序列时间序列预测 (LSTF) 问题的粒度，创建了不同的子集，{ETTh1，ETTh2} 为1小时级，ETTm1为15分钟级。每个数据点由目标值 “油温” 和6个功率负载特征组成。火车/val/测试为12/4/4个月。https://opendatalab.com/OpenDataLab/ETT
    # file_path = "data/exchange_rate.csv"
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    # effiency_utils.memory_function(run_experiment, file_path, output_dir, sample_fraction=1.0, total_budget=1.0, w=160, delta=0.5, kp=0.8, ks=0.1, kd=0.1, DTW_MRE=False)
    # compare_experiments(file_path, output_dir,target="sample_fraction")
    compare_experiments(file_path, output_dir,target="e")
    # compare_experiments(file_path, output_dir,target="w")