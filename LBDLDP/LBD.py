import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time
import tracemalloc

def adaptive_w_event_budget_allocation_lbd(
    num_points: int,
    total_budget: float,
    w: int,
    min_budget: float = 1e-5
) -> np.ndarray:
    """
    只分配差异估计预算 ε_{t,1} = ε/(2w)，
    发布预算 ε_{t,2} 将在 perturbation 函数中按滑动窗口动态分配。
    """
    # ε/2 均摊到 w 个时刻上的差异估计预算
    dis_eps = total_budget / (2.0 * w)
    budgets = np.zeros((num_points, 2))
    budgets[:, 0] = np.maximum(dis_eps, min_budget)
    # 暂时不填 budgets[:,1]，留给 perturbation 时动态设置
    return budgets


def sw_perturbation_w_event_lbd(
    values: np.ndarray,
    budgets: np.ndarray,
    total_budget: float = 1.0,
    w: int = 160,
    min_budget: float = 1e-5
) -> np.ndarray:
    """
    按论文算法 1 的 M_{t,1} & M_{t,2}：
     1) 子机制 M_{t,1}：用 ε_{t,1} 做 Laplace 差异估计；
     2) 计算滑动窗口内已用发布预算，得剩余 ε_rm；
     3) 设潜在发布预算 ε_{t,2} = max(ε_rm/2, min_budget)；
     4) 若 dis > Var(ε_{t,2}) 则按 ε_{t,2} 发布，否则跳过。
    """
    n = len(values)
    perturbed = np.zeros(n)
    last_release = None

    # 用 deque 跟踪过去 w−1 个时刻实际消耗的发布预算
    recent_pub_eps = deque(maxlen=w-1)

    # 拉普拉斯机制
    def laplace(v, eps):
        if eps <= 0:
            return v
        return v + np.random.laplace(scale=1.0/eps)

    def laplace_var(eps):
        return 2.0 / (eps**2) if eps > 0 else np.inf

    for t in range(n):
        # --- M_{t,1}: 差异估计 ---
        eps1 = budgets[t, 0]
        c_t1 = laplace(values[t], eps1)
        var1 = laplace_var(eps1)

        # 初始化第一条发布
        if t == 0:
            perturbed[t] = c_t1
            last_release = c_t1
            recent_pub_eps.append(0.0)
            continue

        # 计算 dis = (c_t1 - r_{t-1})^2 − Var(ε_{t,1})
        dis = (c_t1 - last_release)**2 - var1
        dis = max(dis, 0.0)

        # --- 计算滑动窗口剩余发布预算 ε_rm ---
        used_pub = sum(recent_pub_eps)
        eps_rm = total_budget/2.0 - used_pub

        # 动态分配潜在发布预算 ε_{t,2}
        if eps_rm > min_budget:
            eps2 = max(eps_rm / 2.0, min_budget)
        else:
            eps2 = 0.0
        budgets[t, 1] = eps2

        # --- M_{t,2}: 决定是否发布 ---
        var2 = laplace_var(eps2)
        if eps2 > 0 and dis > var2:
            # 真正发布
            c_t2 = laplace(values[t], eps2)
            perturbed[t] = c_t2
            last_release = c_t2
            recent_pub_eps.append(eps2)
        else:
            # 跳过发布
            perturbed[t] = last_release
            recent_pub_eps.append(0.0)

    return perturbed

def run_single_experiment(x, eps, file_path, w, sample_method="uniform"):
    # 开始时间和内存监控
    start_time = time.time()
    tracemalloc.start()

    # 预处理得到采样后的归一化值
    sample_data, origin_data = data_utils.preprocess_data(file_path, x, sample_method)
    sample_normalized = sample_data['normalized_value'].values
    origin_normalized = origin_data['normalized_value'].values

    # 这里不再把 slopes / fluctuation_rates 传给预算分配
    n = len(sample_normalized)
    budgets = adaptive_w_event_budget_allocation_lbd(
        num_points=n,
        total_budget=eps,
        w=w,
        min_budget=0.01
    )

    perturbed_values = sw_perturbation_w_event_lbd(
        sample_normalized,
        budgets,
        total_budget=eps,
        w=w,
        min_budget=0.01
    )

    # 插值、DTW/MRE 等后续不变
    sample_data["smoothed_value"] = perturbed_values
    interpolated = data_utils.interpolate_missing_points(origin_data, sample_data)
    interp_vals = interpolated['smoothed_value'].values

    # 停止监控，计算指标
    elapsed_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with ThreadPoolExecutor() as executor:
        dtw_future = executor.submit(data_utils.calculate_fdtw, origin_normalized, interp_vals)
        mre_future = executor.submit(data_utils.calculate_mre, interp_vals, origin_normalized)
        dtw_distance = dtw_future.result()
        mre = mre_future.result()

    return {
        "sampling_rate": x,
        "epsilon": eps,
        "dtw": dtw_distance,
        "mre": mre,
        "runtime": elapsed_time,
        "peak_memory": peak_mem / 10**6
    }