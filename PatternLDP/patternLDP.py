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
def pattern_aware_sampling(df, delta=0.02, pattern_slack=1.2):
    """
    按论文式 (5)-(6) 实现 PLA，但用 pattern_slack 放大 delta：
      effective_delta = delta * pattern_slack
    这样容差更大，采样点更少，模式保留相对更弱一些。
    """
    vals = df['normalized_value'].to_numpy()
    n = len(vals)
    if n == 0:
        return []

    eff_delta = delta * pattern_slack
    sampled = [0]
    last = 0

    while last < n - 1:
        llow, lup = -np.inf, np.inf
        for i in range(last + 1, n):
            slope_low  = (vals[i] - vals[last] - eff_delta) / (i - last)
            slope_up   = (vals[i] - vals[last] + eff_delta) / (i - last)
            llow = max(llow, slope_low)
            lup  = min(lup, slope_up)
            if llow > lup:
                sampled.append(i - 1)
                last = i - 1
                break
        else:
            sampled.append(n - 1)
            break

    return sorted(set(sampled))

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

def allocate_privacy_budget(df, total_budget, w=160, init_alpha=0.5, eps_alpha=1e-6):
    """
    改动说明：
      1. eps_alpha：用来保护 (1-alpha) 的下限，避免除以零。
      2. 更新后将 alpha 限制到 [0, 1-eps_alpha]。
    """
    gamma = np.clip(df['importance'].values, 1e-6, None)
    n = len(gamma)
    budgets = np.zeros(n)
    alpha = init_alpha
    epsilon_w = total_budget / w

    for i in range(n):
        used = budgets[max(0, i - w + 1):i].sum()
        remaining = total_budget - used

        # 分母下限保护，避免 alpha → 1 时除以 0
        denom = max(1.0 - alpha, eps_alpha)
        grad = 1.0 / (denom ** 2)

        # 按式 (12) 更新 α
        if remaining > total_budget / 2.0:
            alpha = alpha - grad
        elif remaining < epsilon_w:
            alpha = alpha + grad
        # else: α 不变

        # 将 alpha 限制在 [0, 1-eps_alpha]
        alpha = min(max(alpha, 0.0), 1.0 - eps_alpha)

        beta = 1.0 - alpha
        # 按式 (11) 计算采样概率 p
        p = 1.0 - np.exp(-(alpha / gamma[i] + beta * gamma[i]))
        budgets[i] = remaining * p

    df['privacy_budget'] = budgets
    return df, alpha, 1.0 - alpha


def sample_trunc_exp(v, eps, b):
    """
    论文式 (14–15) 的截断指数逆变换采样：
    在区间 [v − b, v + b] 上以密度 ∝ exp(−ε|x−v|) 采样。
    """
    if eps <= 0 or b <= 0:
        return v
    # 归一化常数 Z = 1−exp(−ε b)
    Z = 1.0 - math.exp(-eps * b)
    # 随机落点 u ∈ [0, Z]
    u = random.random() * Z
    # 左侧累计 A_left = (1−exp(−ε b)) / ε
    A_left = (1.0 - math.exp(-eps * b)) / eps

    if u <= A_left:
        # 落在左侧
        x = - (1.0 / eps) * math.log(1.0 - eps * u)
        return v - x
    else:
        # 落在右侧
        u2 = u - A_left
        x = - (1.0 / eps) * math.log(1.0 - eps * u2)
        return v + x

def importance_aware_randomization(df, theta=1.0, mu=0.1, eps_Z=1e-6):
    """
    严格按论文式 (14),(15)：
    — 先计算 b = log(θ/γ + μ)
    — 以概率 q = ε/(2Z) 保留原值；否则在两侧截断指数分布上逆变换采样
      这里对 Z 做下限保护，并对 q 做 [0,1] 限制，避免除零或 q>1。
    """
    v_arr   = df['normalized_value'].values
    eps_arr = df['privacy_budget'].values
    gamma   = np.clip(df['importance'].values, 1e-6, None)
    perturbed = np.empty_like(v_arr)

    for i, (v, eps, g) in enumerate(zip(v_arr, eps_arr, gamma)):
        if eps <= 0:
            perturbed[i] = v
            continue

        b = math.log(theta / g + mu)
        if b <= 0:
            perturbed[i] = v
            continue

        # 归一化常数 Z = 1 − exp(−ε b)，并做下限保护
        Z = 1.0 - math.exp(-eps * b)
        Z = max(Z, eps_Z)

        # 计算保留原值的概率，并限制在 [0,1]
        q = eps / (2.0 * Z)
        q = min(max(q, 0.0), 1.0)

        if random.random() < q:
            perturbed[i] = v
        else:
            perturbed[i] = sample_trunc_exp(v, eps, b)

    return perturbed

def run_single_experiment(x, eps, file_path, w, sample_method="uniform", delta=0.5, theta=1.0, mu=0.1):
    # 开始时间和内存监控
    start_time = time.time()
    tracemalloc.start()
    sample_data, origin_data = data_utils.preprocess_data(file_path, x, sample_method)
    origin_length = len(origin_data)
    assert 'date' in sample_data.columns, "'date' column required for PID-based importance."

    sample_data = sample_data.reset_index(drop=True)
    idx_pla = pattern_aware_sampling(sample_data, delta=delta)
    sample_data = sample_data.loc[idx_pla].copy().reset_index(drop=True)

    sample_data['importance'] = compute_importance(sample_data)
    updated_data, alpha, beta = allocate_privacy_budget(sample_data, eps, w)
    perturbed_vals = importance_aware_randomization(updated_data, theta=theta, mu=mu)
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