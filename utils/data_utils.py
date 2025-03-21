import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

def preprocess_HKHS_data(file_path, sample_fraction=1.0):
    """
    预处理 HKHS 数据。
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
    返回:
        sample_data (pd.DataFrame): 采样后的数据，包括日期和归一化后的值
        origin_data (pd.DataFrame): 原始数据，包括日期和归一化后的值
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
    data = data.sort_values(by='date')

    # 归一化处理
    data['normalized_value'] = (data['value'] - np.mean(data['value'])) / np.std(data['value'])

    # 原始数据
    origin_data = data[['date', 'normalized_value']].copy()

    # 如果需要采样
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction).sort_values(by='date')

    # 采样后的数据
    sample_data = data[['date', 'normalized_value']].copy()

    # 返回采样后的数据和原始数据
    return sample_data, origin_data

def preprocess_heartrate_data(file_path, sample_fraction=1.0):
    """
    预处理心率数据。
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
    返回:
        sample_data (pd.DataFrame): 采样后的数据，包括日期和归一化后的值
        origin_data (pd.DataFrame): 原始数据，包括日期和归一化后的值
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 将时间戳转换为日期时间
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.sort_values(by='date')  # 按时间排序
    data['normalized_value'] = (data['value'] - np.mean(data['value'])) / np.std(data['value'])

    # 原始数据
    origin_data = data[['date', 'normalized_value']].copy()

    # 如果需要采样
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction).sort_values(by='date')

    # 采样后的数据
    sample_data = data[['date', 'normalized_value']].copy()

    # 返回采样后的数据和原始数据
    return sample_data, origin_data

def preprocess_ELD_data(file_path, sample_fraction=1.0):
    """
    预处理 Electricity Load Diagrams 数据。
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
    返回:
        sample_data (pd.DataFrame): 采样后的数据，包括日期和归一化后的值
        origin_data (pd.DataFrame): 原始数据，包括日期和归一化后的值
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by='date')

    # 归一化处理
    data['normalized_value'] = (data['value'] - np.mean(data['value'])) / np.std(data['value'])

    # 原始数据
    origin_data = data[['date', 'normalized_value']].copy()

    # 如果需要采样
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction).sort_values(by='date')

    # 采样后的数据
    sample_data = data[['date', 'normalized_value']].copy()

    # 返回采样后的数据和原始数据
    return sample_data, origin_data

def preprocess_data(file_path, sample_fraction=1.0):
    """
    统一接口，根据文件名调用不同的预处理函数。
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
    返回:
        sample_data (pd.DataFrame): 采样后的数据，包括日期和归一化后的值
        origin_data (pd.DataFrame): 原始数据，包括日期和归一化后的值
    """
    if 'HKHS' in file_path:
        return preprocess_HKHS_data(file_path, sample_fraction)
    elif 'heartrate' in file_path:
        return preprocess_heartrate_data(file_path, sample_fraction)
    elif 'LD' in file_path:
        return preprocess_ELD_data(file_path, sample_fraction)
    else:
        raise ValueError("Unsupported file type")

def calculate_mre(perturbed_values, normalized_values):
    mre = np.mean(np.abs(perturbed_values - normalized_values) / (np.abs(normalized_values) + np.abs(perturbed_values) + 1e-8))
    return mre

from scipy.interpolate import interp1d
def pla_interpolation(sampled_data, origin_length):
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
