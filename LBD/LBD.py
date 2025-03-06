import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
from concurrent.futures import ThreadPoolExecutor
from collections import deque

def adaptive_w_event_budget_allocation_lbd(
    slopes,
    fluctuation_rates,
    total_budget,
    w,
    min_budget=1e-5
):
    """
    仅用于返回“差异估计预算 (M_{t,1})”和“潜在发布预算 (M_{t,2})”的初始值。

    对照论文 LBD 思路: 
    1) 在 w 个时刻内, 差异估计累计使用 <= total_budget/2
    2) 发布累计使用 <= total_budget/2
    3) 合计 <= total_budget -> 满足 w-event LDP
    对单条序列, 这里做简化:
      - 每个时刻的差异估计预算都固定为 dis_eps = total_budget / (2 * w)
      - 发布预算初始置 0, 最终在 sw_perturbation_w_event_lbd(...) 内根据 dis>err 的决策确定实际使用多少

    参数 slopes, fluctuation_rates 在论文 LBD 中不是必须,
    这里保留是为了兼容原函数签名, 并示范仅做“恒定分配”即可.
    """
    num_points = len(slopes)
    # budgets 数组形状为 (num_points, 2):
    #   budgets[i, 0] -> 子机制 M_{t,1} (差异估计)
    #   budgets[i, 1] -> 子机制 M_{t,2} (潜在发布), 先填 0
    budgets = np.zeros((num_points, 2))

    dis_eps = total_budget / (2.0 * w)
    for i in range(num_points):
        budgets[i, 0] = max(dis_eps, min_budget)  # 差异估计预算
        budgets[i, 1] = 0.0                      # 潜在发布预算后面决策

    return budgets

def sw_perturbation_w_event_lbd(
    values,
    budgets,
    total_budget=1.0,
    w=160,
    min_budget=0.01
):
    """
    对应论文中 "Algorithm 1: LBD" 的核心流程:
      - 子机制 M_{t,1}: 
        -> 用 budgets[i,0] (dis_epsilon) 做“私有差异估计”, 拿到 c_{t,1}, 
           计算 dis = (c_{t,1} - r_{t-1})^2 - Var(c_{t,1}).
      - 子机制 M_{t,2}:
        -> 计算过去 w-1 个时刻已经用掉的发布预算 used_pub_sum, 
           remain_pub = total_budget/2 - used_pub_sum
        -> 假设要用潜在发布预算 pub_eps = remain_pub / 2 (或别的策略), 计算潜在 err
        -> 如果 dis > err, 则真正发布(再做一次加噪, 并扣除 pub_eps); 否则跳过(近似).
      - w-event 原则: 在任意长度为 w 的窗里, 差异估计预算总和 + 发布预算总和 <= total_budget.
        本函数通过:
          1) 差异估计固定占 total_budget/2 (在 adaptive... 中已设置)
          2) 发布占 total_budget/2
          3) 用一个队列 recent_pub_budgets 跟踪过去 w-1 个发布预算之和.
    
    返回:
      perturbed_values: 对应每个时刻 t 的最终发布值 (单条序列).
    """
    n = len(values)
    perturbed_values = np.zeros(n)

    # Laplace 机制 (单变量)
    def laplace_perturb(v, eps):
        if eps < 1e-12:
            return v
        scale = 1.0 / eps
        noise = np.random.laplace(0.0, scale)
        return v + noise

    # Laplace 机制的方差 (单变量)
    def laplace_var(eps):
        if eps < 1e-12:
            return 1e12
        return 2.0 / (eps**2)

    # 跟踪过去 w-1 个时刻的“发布预算”
    recent_pub_budgets = deque()
    last_released = 0.0

    # 初始化: 第一个时刻可视情况直接当作一次发布, 或仅当作差异估计
    if n > 0:
        # 用差异估计预算给出 r_0, 也可以认为是 c_{0,1}
        eps_0 = budgets[0, 0]
        r0 = laplace_perturb(values[0], eps_0)
        perturbed_values[0] = r0
        last_released = r0
        # 第一时刻不消耗发布预算
        recent_pub_budgets.append(0.0)

    # 主循环
    for i in range(1, n):
        # ---- (A) 子机制 M_{t,1}: 差异估计 ----
        dis_eps = budgets[i, 0]  # 在 adaptive_w_event_budget_allocation_lbd 中设的
        c_t1 = laplace_perturb(values[i], dis_eps)

        var_ct1 = laplace_var(dis_eps)
        dis = (c_t1 - last_released)**2 - var_ct1
        if dis < 0:
            dis = 0.0

        # ---- (B) 计算可用发布预算 remain_pub ----
        # 维护滑动窗口: 若超 w-1 个已发布预算记录, 则 pop 出队
        while len(recent_pub_budgets) >= (w - 1):
            recent_pub_budgets.popleft()
        used_pub_sum = sum(recent_pub_budgets)
        remain_pub = (total_budget / 2.0) - used_pub_sum  # LBD中, 发布预算最多占 total_budget/2

        if remain_pub < min_budget:
            # 发布预算不足, 只能跳过本次发布
            perturbed_values[i] = last_released
            budgets[i, 1] = 0.0
            recent_pub_budgets.append(0.0)
            continue

        # ---- (C) 子机制 M_{t,2}: 决定是否发布 ----
        # 假设要投入一部分潜在预算 pub_eps
        pub_eps = max(remain_pub / 2.0, min_budget)
        err = laplace_var(pub_eps)

        # (C1) 若 dis > err, 执行真正发布
        if dis > err:
            new_release = laplace_perturb(values[i], pub_eps)
            perturbed_values[i] = new_release
            # 记录实际使用的发布预算
            budgets[i, 1] = pub_eps
            recent_pub_budgets.append(pub_eps)
            last_released = new_release
        else:
            # (C2) 否则跳过, 近似上一时刻
            perturbed_values[i] = last_released
            budgets[i, 1] = 0.0
            recent_pub_budgets.append(0.0)

    return perturbed_values

def run_experiment(
    file_path,
    output_dir,
    sample_fraction=1.0,
    total_budget=1.0,
    w=160,
    delta=0.5,
    kp=0.8,
    ks=0.1,
    kd=0.1,
    DTW_MRE=True
):
    """
    与原先的 run_experiment 同名, 保持接口一致, 流程:
      1) 数据预处理 (采样+归一化)
      2) 调用 adaptive_w_event_budget_allocation_lbd(...) 得到 (N,2) budgets
      3) 用 sw_perturbation_w_event_lbd(...) 执行论文式 LBD 机制
      4) 插值 + 评估 DTW / MRE
    """
    # Step 1: 数据预处理
    sample_data, origin_data = data_utils.preprocess_data(file_path, sample_fraction)
    sample_normalized_data = sample_data['normalized_value'].values
    origin_normalized_data = origin_data['normalized_value'].values

    # Step 2: 计算 slopes (兼容原代码, 不用于分配)
    slopes = np.gradient(sample_normalized_data)
    fluctuation_rates = np.abs(slopes)

    # Step 3: 生成差异估计和潜在发布预算 (论文 LBD)
    budgets = adaptive_w_event_budget_allocation_lbd(
        slopes,
        fluctuation_rates,
        total_budget=total_budget,
        w=w
    )

    # Step 4: 执行 LBD 机制
    perturbed_values = sw_perturbation_w_event_lbd(
        sample_normalized_data,
        budgets,
        total_budget=total_budget,
        w=w,
        min_budget=0.01
    )

    # 将扰动结果赋给 sample_data
    sample_data["smoothed_value"] = perturbed_values

    # Step 5: 插值并计算评估指标
    interpolated_data = data_utils.interpolate_missing_points(origin_data, sample_data)
    interpolated_values = interpolated_data['smoothed_value']

    dtw_distance = None
    mre = None
    if DTW_MRE:
        with ThreadPoolExecutor() as executor:
            dtw_future = executor.submit(
                data_utils.calculate_fdtw,
                origin_normalized_data,
                interpolated_values
            )
            mre_future = executor.submit(
                data_utils.calculate_mre,
                interpolated_values,
                origin_normalized_data
            )
            dtw_distance = dtw_future.result()
            mre = mre_future.result()

    return {
        'sample_normalized_data': sample_normalized_data,
        'sample_perturbed_values': perturbed_values,
        'interpolated_values': interpolated_values,
        'dtw_distance': dtw_distance,
        'mre': mre
    }

def run_single_experiment(x, eps, file_path):
    """
    与原先同名, 保持接口, 在指定采样率 x 和隐私预算 eps 下运行一次.
    同样调用上述两阶段逻辑
    """
    sample_data, origin_data = data_utils.preprocess_data(file_path, x)
    sample_normalized_data = sample_data['normalized_value'].values
    origin_normalized_data = origin_data['normalized_value'].values

    slopes = np.gradient(sample_normalized_data)
    fluctuation_rates = np.abs(slopes)

    budgets = adaptive_w_event_budget_allocation_lbd(
        slopes,
        fluctuation_rates,
        total_budget=eps,
        w=160
    )
    perturbed_values = sw_perturbation_w_event_lbd(
        sample_normalized_data,
        budgets,
        total_budget=eps,
        w=160,
        min_budget=0.01
    )

    sample_data["smoothed_value"] = perturbed_values
    interpolated_data = data_utils.interpolate_missing_points(origin_data, sample_data)
    interpolated_values = interpolated_data['smoothed_value']

    with ThreadPoolExecutor() as executor:
        dtw_future = executor.submit(
            data_utils.calculate_fdtw,
            origin_normalized_data,
            interpolated_values
        )
        mre_future = executor.submit(
            data_utils.calculate_mre,
            interpolated_values,
            origin_normalized_data
        )
        dtw_distance = dtw_future.result()
        mre = mre_future.result()

    return {
        "sampling_rate": x,
        "epsilon": eps,
        "dtw": dtw_distance,
        "mre": mre
    }

if __name__ == "__main__":
    file_path = 'data/LD.csv'  # 示例
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
    print("DTW =", result['dtw_distance'], "MRE =", result['mre'])