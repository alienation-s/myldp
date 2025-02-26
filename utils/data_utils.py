import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns

def preprocess_HKHS_data(file_path, sample_fraction=1.0):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
    data = data.sort_values(by='date')

    data['normalized_value'] = (data['value'] - np.mean(data['value'])) / np.std(data['value'])
    
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=9999).sort_values(by='date')

    # 采样后的归一化列（与原始归一化列相同）
    data['sample_normalized_value'] = data['normalized_value']

    return data, data['value'].values


def preprocess_heartrate_data(file_path, sample_fraction=1.0):
    """
    预处理心率数据。
    参数:
        file_path (str): CSV 文件路径
        sample_fraction (float): 采样比例, 默认为 1.0 表示使用全部数据
    返回:
        data (pd.DataFrame): 预处理后的 DataFrame, 包括归一化后的数据
        original_values (np.ndarray): 原始心率值数组
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 将时间戳转换为日期时间
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.sort_values(by='date')  # 按时间排序
    data['normalized_value'] = (data['value'] - np.mean(data['value'])) / np.std(data['value'])
    # 如果需要采样
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=9999).sort_values(by='date')

    # 采样后的归一化列（与原始归一化列相同）
    data['sample_normalized_value'] = data['normalized_value']

    # 返回预处理后的数据和原始数值
    return data, data['value'].values

def calculate_mre(perturbed_values, normalized_values):
    mre = np.mean(np.abs(perturbed_values - normalized_values) / (np.abs(normalized_values) + np.abs(perturbed_values) + 1e-8))
    return mre

from scipy.interpolate import interp1d

def generate_piecewise_linear_curve(original_dates, significant_indices, values):
    # 将显著点日期转换为时间戳（significant_indices, values）对应着索引和扰动后的值
    significant_dates = original_dates.iloc[significant_indices].astype('int64') / 1e9  # 转换为秒
    original_dates_float = original_dates.astype('int64') / 1e9  # 转换为秒
    # 构造插值函数
    interpolation_func = interp1d(significant_dates, values, kind='linear', fill_value="extrapolate")
    # 插值
    piecewise_values = interpolation_func(original_dates_float)
    return piecewise_values


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
