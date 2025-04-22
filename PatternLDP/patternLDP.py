import numpy as np
import pandas as pd
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import math
import datetime
import time
import tracemalloc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils

def pattern_aware_sampling_feasible_space(df, delta=0.02):
    vals = df['normalized_value'].to_numpy()
    n = len(vals)
    if n == 0:
        return []

    sampled_indices = [0]
    i = 0
    while i < n - 1:
        llow = -np.inf
        lup = np.inf
        j = i + 1
        while j < n:
            low_list = []
            up_list = []
            for k in range(i + 1, j):
                denom = k - i
                if denom == 0:
                    continue
                low_k = (vals[k] - delta - vals[i]) / denom
                up_k = (vals[k] + delta - vals[i]) / denom
                low_list.append(low_k)
                up_list.append(up_k)
            if low_list:
                llow = max(llow, max(low_list))
                lup = min(lup, min(up_list))
            if llow > lup:
                sampled_indices.append(j - 1)
                i = j - 1
                break
            j += 1
        else:
            sampled_indices.append(n - 1)
            break
    return sampled_indices

def compute_importance(df, Kp=0.8, Ki=0.1, Kd=0.1, pi=5):
    if 'date' not in df.columns or 'normalized_value' not in df.columns:
        raise ValueError("DataFrame must include 'date' and 'normalized_value'.")
    df = df.copy()
    df['time_diff'] = df['date'].diff().fillna(pd.Timedelta(seconds=1)).dt.total_seconds().astype(float)
    df['predicted'] = df['normalized_value'].shift(1).fillna(method='bfill')
    df['F'] = (df['normalized_value'] - df['predicted']).abs()

    F_arr = df['F'].values
    time_diff_arr = df['time_diff'].values
    n = len(df)
    window = deque()
    sum_window = 0.0
    int_list, der_list, pro_list = [], [], []

    for i in range(n):
        f_i = F_arr[i]
        window.append(f_i)
        sum_window += f_i
        if len(window) > pi:
            sum_window -= window.popleft()
        i_val = (sum_window / len(window)) * Ki
        d_val = ((f_i - F_arr[i - 1]) / time_diff_arr[i]) * Kd if i > 0 and time_diff_arr[i] != 0 else 0.0
        p_val = f_i * Kp
        int_list.append(i_val)
        der_list.append(d_val)
        pro_list.append(p_val)

    df['importance'] = np.array(pro_list) + np.array(int_list) + np.array(der_list)
    return df['importance'].values

def allocate_privacy_budget(df, total_budget, w=160, init_alpha=0.5):
    """
    Allocate per-point privacy budgets with dynamic equilibrium factor as in PatternLDP paper.
    """
    df = df.copy()
    n = len(df)
    gamma_arr = np.clip(df['importance'].values, 1e-6, None)
    budgets = np.zeros(n)
    alpha = init_alpha
    # Correct threshold: total_budget divided by window size w
    epsilon_w = total_budget / w

    for i in range(n):
        # Compute sum of budgets in the current sliding window
        start_i = max(0, i - w + 1)
        remaining = total_budget - budgets[start_i:i].sum()
        # Gradient of equilibrium factor B(α) = α / (1 - α)
        grad = 1.0 / (1.0 - alpha)**2
        # Adjust alpha dynamically per Paper Eq.(12)
        if remaining > total_budget / 2.0:
            alpha = max(0.1, alpha - grad)
        elif remaining < epsilon_w:
            alpha = min(0.9, alpha + grad)
        beta = 1.0 - alpha
        gamma_i = gamma_arr[i]
        # Proportional function p = 1 - exp(-(α/γ + β γ))
        p = 1 - np.exp(-(alpha / gamma_i + beta * gamma_i))
        budgets[i] = remaining * p

    df['privacy_budget'] = budgets
    return df, alpha, 1.0 - alpha

def sample_exponential_interval(v, eps, lower, upper):
    L, R = v - lower, upper - v
    A_left = (1 / eps) * (1 - math.exp(-eps * L))
    A_right = (1 / eps) * (1 - math.exp(-eps * R))
    Z = A_left + A_right
    u = random.random() * Z

    if u <= A_left:
        x_prime = - (1 / eps) * math.log(max(1e-15, 1 - eps * u))
        return v - x_prime
    else:
        M2 = u - A_left
        x_prime = - (1 / eps) * math.log(max(1e-15, 1 - eps * M2))
        return v + x_prime

def importance_aware_randomization_precise(df, theta=1.0, mu=0.1):
    df = df.copy()
    v_arr = df['normalized_value'].values
    eps_arr = df['privacy_budget'].values
    gamma_arr = np.clip(df['importance'].values, 1e-6, None)
    b_arr = np.clip(np.log(theta / gamma_arr + mu), 1e-3, None)
    perturbed_values = []

    for i in range(len(df)):
        v, eps, b = v_arr[i], eps_arr[i], b_arr[i]
        if eps < 1e-6 or b <= 0:
            perturbed_values.append(v)
            continue
        lower, upper = v - b, v + b
        perturbed_values.append(sample_exponential_interval(v, eps, lower, upper))
    return perturbed_values

def run_experiment(file_path,
                   output_dir=None,
                   w=100,
                   total_budget=1.0,
                   sample_fraction=1.0,
                   delta=0.5,
                   DTW_MRE=True,
                   Kp=0.8, Ki=0.1, Kd=0.1, pi=5,
                   theta=1.0, mu=0.1):

    sample_data, origin_data = data_utils.preprocess_data(file_path, sample_fraction)
    origin_length = len(origin_data)
    assert 'date' in sample_data.columns, "'date' column required for PID-based importance."

    sample_data = sample_data.reset_index(drop=True)
    idx_pla = pattern_aware_sampling_feasible_space(sample_data, delta=delta)
    sample_data = sample_data.loc[idx_pla].copy().reset_index(drop=True)

    sample_data['importance'] = compute_importance(sample_data, Kp=Kp, Ki=Ki, Kd=Kd, pi=pi)
    updated_data, alpha, beta = allocate_privacy_budget(sample_data, total_budget, w)
    perturbed_vals = importance_aware_randomization_precise(updated_data, theta=theta, mu=mu)
    updated_data['perturbed_value'] = perturbed_vals

    updated_data['timestamp'] = updated_data.index
    interpolated_values = data_utils.pla_interpolation(
        updated_data[['timestamp', 'perturbed_value']], origin_length
    )

    dtw_distance, mre = None, None
    if DTW_MRE:
        original_series = origin_data['normalized_value'].values
        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(data_utils.calculate_fdtw, original_series, interpolated_values)
            f2 = executor.submit(data_utils.calculate_mre, interpolated_values, original_series)
            dtw_distance = f1.result()
            mre = f2.result()

    return {
        'origin_data': origin_data,
        'sampled_data': updated_data,
        'interpolated_values': interpolated_values,
        'dtw_distance': dtw_distance,
        'mre': mre,
        'alpha': alpha,
        'beta': beta
    }

def run_single_experiment(x, eps, file_path, w, delta=0.5, theta=1.0, mu=0.1):
    # 开始时间和内存监控
    start_time = time.time()
    tracemalloc.start()
    sample_data, origin_data = data_utils.preprocess_data(file_path, x)
    origin_length = len(origin_data)
    assert 'date' in sample_data.columns, "'date' column required for PID-based importance."

    sample_data = sample_data.reset_index(drop=True)
    idx_pla = pattern_aware_sampling_feasible_space(sample_data, delta=delta)
    sample_data = sample_data.loc[idx_pla].copy().reset_index(drop=True)

    sample_data['importance'] = compute_importance(sample_data)
    updated_data, alpha, beta = allocate_privacy_budget(sample_data, eps, w)
    perturbed_vals = importance_aware_randomization_precise(updated_data, theta=theta, mu=mu)
    updated_data['perturbed_value'] = perturbed_vals

    updated_data['timestamp'] = updated_data.index
    interpolated_values = data_utils.pla_interpolation(
        updated_data[['timestamp', 'perturbed_value']], origin_length
    )
    # 统计耗时、内存峰值
    elapsed_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    original_series = origin_data['normalized_value'].values
    with ThreadPoolExecutor() as executor:
        f1 = executor.submit(data_utils.calculate_fdtw, original_series, interpolated_values)
        f2 = executor.submit(data_utils.calculate_mre, interpolated_values, original_series)
        dtw_distance = f1.result()
        mre = f2.result()

    return {
        "sampling_rate": x,
        "epsilon": eps,
        "dtw": dtw_distance,
        "mre": mre,
        "runtime": elapsed_time,
        "peak_memory": peak_mem / 10**6,
    }

if __name__ == "__main__":
    file_path = 'data/LD.csv'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    result = run_experiment(
        file_path,
        output_dir,
        sample_fraction=1.0,
        total_budget=1.0,
        w=160,
        DTW_MRE=True
    )

    print(f"DTW: {result['dtw_distance']:.4f}, MRE: {result['mre']:.4f}")