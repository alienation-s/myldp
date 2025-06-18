import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

random_seed = 42

def sample_data_method(data, sample_fraction, sample_method):
    if sample_fraction >= 1.0:
        return data

    if sample_method == "uniform":
        return data.sample(frac=sample_fraction, random_state=random_seed).sort_values(by='date')

    elif sample_method == "reservoir":
        # Reservoir sampling with fixed size
        k = int(len(data) * sample_fraction)
        if k <= 0:
            return data.iloc[[]]
        np.random.seed(random_seed)
        indices = np.random.choice(len(data), size=k, replace=False)
        return data.iloc[indices].sort_values(by='date')

    elif sample_method == "stratified":
        # Stratify by equally spaced time bins
        n_bins = int(1.0 / sample_fraction)
        data = data.copy()
        data['bin'] = pd.qcut(data['date'].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
        sampled = data.groupby('bin').apply(lambda g: g.sample(frac=sample_fraction, random_state=random_seed)).reset_index(drop=True)
        return sampled.sort_values(by='date')

    else:
        raise ValueError(f"Unsupported sample method: {sample_method}")

def preprocess_other_data(file_path, sample_fraction=1.0, sample_method="uniform"):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M')
    data = data.sort_values(by='date')
    data['normalized_value'] = data['value']  # 保留字段名，但不做归一化

    origin_data = data[['date', 'normalized_value']].copy()
    data = sample_data_method(data, sample_fraction, sample_method)
    sample_data = data[['date', 'normalized_value']].copy()
    return sample_data, origin_data

def preprocess_HKHS_data(file_path, sample_fraction=1.0, sample_method="uniform"):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
    data = data.sort_values(by='date')
    data['normalized_value'] = data['value']  # 保留字段名，但不做归一化

    origin_data = data[['date', 'normalized_value']].copy()
    data = sample_data_method(data, sample_fraction, sample_method)
    sample_data = data[['date', 'normalized_value']].copy()
    return sample_data, origin_data

def preprocess_heartrate_data(file_path, sample_fraction=1.0, sample_method="uniform"):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.sort_values(by='date')
    data['normalized_value'] = data['value']  # 保留字段名，但不做归一化

    origin_data = data[['date', 'normalized_value']].copy()
    data = sample_data_method(data, sample_fraction, sample_method)
    sample_data = data[['date', 'normalized_value']].copy()
    return sample_data, origin_data

def preprocess_ELD_data(file_path, sample_fraction=1.0, sample_method="uniform"):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by='date')
    data['normalized_value'] = data['value']  # 保留字段名，但不做归一化

    origin_data = data[['date', 'normalized_value']].copy()
    data = sample_data_method(data, sample_fraction, sample_method)
    sample_data = data[['date', 'normalized_value']].copy()
    return sample_data, origin_data

def preprocess_data(file_path, sample_fraction=1.0, sample_method="reservoir"):
    """
    通用预处理入口
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
        sample_method (str): 采样方法, 可选值: 'uniform', 'reservoir', 'stratified'
    返回:
        sample_data (pd.DataFrame): 采样后的数据
        origin_data (pd.DataFrame): 原始数据
    """
    # print("采样的方法："+ sample_method)
    if 'HKHS' in file_path:
        return preprocess_HKHS_data(file_path, sample_fraction, sample_method)
    elif 'heartrate' in file_path:
        return preprocess_heartrate_data(file_path, sample_fraction, sample_method)
    elif 'LD' in file_path:
        return preprocess_ELD_data(file_path, sample_fraction, sample_method)
    else:
        return preprocess_other_data(file_path, sample_fraction, sample_method)
    

def calculate_mre(perturbed_values, normalized_values):
    mre = np.mean(np.abs(perturbed_values - normalized_values) / (np.abs(normalized_values) + np.abs(perturbed_values) + 1e-8))
    return mre

from scipy.interpolate import interp1d
def pla_interpolation(sampled_data, origin_length):
    # 添加空值过滤
    sampled_data = sampled_data.dropna(subset=['perturbed_value']).copy()
    
    # 强制类型转换
    sampled_data['perturbed_value'] = pd.to_numeric(
        sampled_data['perturbed_value'], 
        errors='coerce'
    )
    
    # 处理不足两个采样点的情况
    if len(sampled_data) < 2:
        default_value = sampled_data['perturbed_value'].mean() if not sampled_data.empty else 0.0
        return np.full(origin_length, default_value)
    
    # 输入数据校验
    if not isinstance(sampled_data, pd.DataFrame) or len(sampled_data) < 2:
        return np.zeros(origin_length)
    
    # 强制类型转换确保数值类型
    try:
        sampled_data = sampled_data.copy()
        # 转换索引为整数类型
        sampled_data.index = sampled_data.index.astype(int)
        # 转换值为浮点类型（处理可能的字符串或空值）
        sampled_data['perturbed_value'] = pd.to_numeric(
            sampled_data['perturbed_value'], 
            errors='coerce'
        ).fillna(method='ffill').astype(float)
    except Exception as e:
        raise ValueError(f"Data type conversion failed: {str(e)}")

    interpolated = np.zeros(origin_length, dtype=np.float64)
    sampled_indices = sampled_data.index.values.astype(int)
    sampled_values = sampled_data['perturbed_value'].values.astype(np.float64)

    # 必须保证至少两个点才能插值
    if len(sampled_indices) < 2:
        interpolated[:] = sampled_values[0] if len(sampled_indices) > 0 else 0.0
        return interpolated

    # 主插值逻辑
    for i in range(len(sampled_indices) - 1):
        start_idx = int(sampled_indices[i])
        end_idx = int(sampled_indices[i+1])
        start_val = float(sampled_values[i])
        end_val = float(sampled_values[i+1])
        
        if start_idx >= end_idx:  # 防止无效区间
            continue
            
        try:
            x = np.arange(start_idx, end_idx+1, dtype=np.int64)
            interpolated[x] = np.interp(
                x.astype(np.float64),  # 确保x是float类型
                np.array([start_idx, end_idx], dtype=np.float64),
                np.array([start_val, end_val], dtype=np.float64)
            )
        except Exception as e:
            raise RuntimeError(f"Interpolation failed at segment {i}: {str(e)}")

    # 处理末尾区间
    last_idx = int(sampled_indices[-1])
    if last_idx < origin_length - 1:
        last_val = float(sampled_values[-1])
        interpolated[last_idx:] = last_val

    return interpolated
def pla_interpolationbk(sampled_data, origin_length):
    """
    基于分段线性近似（PLA）的插值方法
    参数:
        sampled_data (pd.DataFrame): 采样后的显著点数据，包含'timestamp'和'perturbed_value'
        origin_length (int): 原始时间序列长度
    返回:
        interpolated_values (np.array): 插值后的完整时间序列
    """
    interpolated = np.zeros(origin_length)
    sampled_indices = sampled_data.index.values
    sampled_values = sampled_data['perturbed_value'].values
    
    for i in range(len(sampled_indices) - 1):
        start_idx = sampled_indices[i]
        end_idx = sampled_indices[i+1]
        start_val = sampled_values[i]
        end_val = sampled_values[i+1]
        
        # 线性插值
        x = np.arange(start_idx, end_idx+1)
        interpolated[x] = np.interp(
            x,
            [start_idx, end_idx],
            [start_val, end_val]
        )
    
    # 处理最后一个区间到序列末尾
    if sampled_indices[-1] < origin_length - 1:
        start_idx = sampled_indices[-1]
        start_val = sampled_values[-1]
        interpolated[start_idx:] = start_val  # 保持最后一个值
    
    return interpolated

def interpolate_missing_points(origin_data, sample_data):
    """
    如果长度一致不进行插值，长度不一致则进行插值。
    参数:
        origin_data (pd.DataFrame): 原始数据，包括日期和归一化后的值
        sample_data (pd.DataFrame): 采样后的数据，包括日期和归一化后的值
    返回:
        interpolated_data (pd.DataFrame): 插值后的数据，包括日期和插值后的值
    """
    if len(origin_data) == len(sample_data):
        return sample_data[['date', 'smoothed_value']]

    # 进行插值
    interpolation_func = interp1d(sample_data['date'].astype(np.int64), sample_data['smoothed_value'], kind='linear', fill_value="extrapolate")
    interpolated_values = interpolation_func(origin_data['date'].astype(np.int64))

    # 创建插值后的 DataFrame
    interpolated_data = origin_data.copy()
    interpolated_data['smoothed_value'] = interpolated_values

    return interpolated_data

from fastdtw import fastdtw

def calculate_fdtw(original_series, fitted_series):
    distance, _ = fastdtw(original_series, fitted_series)
    return distance


from dtaidistance import dtw, dtw_visualisation as dtwvis
def visualize_dtw(series1, series2, output_path):
    """
    可视化 DTW 匹配路径
    参数:
        series1 (np.ndarray): 时间序列 1
        series2 (np.ndarray): 时间序列 2
        output_path (str): 保存图像路径
    """
    path = dtw.warping_path(series1, series2)
    dtwvis.plot_warping(series1, series2, path, filename=output_path)
