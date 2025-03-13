import numpy as np
import pandas as pd
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.data_utils as data_utils
import datetime
def pattern_aware_sampling(df, delta=0.02, min_spacing=100, debug=False):
    """
    依据折线近似的方式，从时间序列中选取关键点，
    若某段拟合误差超过 delta 或者长度超过 min_spacing，则在此段插入最大误差点。

    优化要点：
      1) 将 df['normalized_value'] 转为 NumPy 数组，避免在内层循环反复 df.loc 取值。
      2) 对内层区间 [last_selected+1, i) 的误差一次性用向量化计算，减少 Python 解释器开销。

    参数：
      df           : DataFrame，至少包含 'normalized_value' 列。
      delta        : 允许的最大拟合误差阈值。
      min_spacing  : 在两个采样点之间允许的最大距离（索引差）。
      debug        : 若为 True，可在采样结束后打印调试信息。

    返回：
      sampled_indices : 关键点在原 df 中的行索引列表。
    """
    n = len(df)
    if n == 0:
        return []

    # 将 'normalized_value' 提前转成 NumPy 数组，减少后续循环中对 DataFrame 的访问
    vals = df['normalized_value'].to_numpy()

    # 采样点索引列表，初始只包含第 0 个点
    sampled_indices = [0]
    last_selected = 0

    for i in range(1, n):
        x1, y1 = last_selected, vals[last_selected]
        x2, y2 = i, vals[i]

        dx = x2 - x1
        if dx == 0:
            # 避免除零错误（除非这里本身就有重复索引）
            continue

        # 拟合直线 c(x) = slope * x + intercept
        slope = (y2 - y1) / dx
        intercept = y1 - slope * x1

        # 如果中间没有点可供评估，直接看是否超 min_spacing
        if (i - last_selected) <= 1:
            # 区间内无中间点
            if (i - last_selected) >= min_spacing:
                sampled_indices.append(i)
                last_selected = i
            continue

        # 对区间 [last_selected+1, i-1] 的所有点做矢量化预测并计算误差
        jarr = np.arange(last_selected + 1, i)
        predicted = slope * jarr + intercept
        actual = vals[jarr]
        errorarr = np.abs(actual - predicted)

        idxmax = np.argmax(errorarr)          # 最大误差值对应下标在 jarr 的相对位置
        max_error = errorarr[idxmax]          # 最大误差
        max_error_idx = jarr[idxmax]          # 转换成 df 中的绝对索引

        # 如果超过允许误差 或者 已经到达 min_spacing
        if max_error > delta or (i - last_selected) >= min_spacing:
            # 在最大误差点（或 i）上进行分段
            # 注：如果 max_error_idx=-1 理论上不太会出现，这里保留对原逻辑的一致性
            sampled_indices.append(max_error_idx if max_error_idx != -1 else i)
            last_selected = sampled_indices[-1]

    # 确保最后一个点被加入
    if sampled_indices[-1] != n - 1:
        sampled_indices.append(n - 1)

    if debug:
        print(f"[pattern_aware_sampling_optimized] Selected Indices = {sampled_indices}")
        print(f"Total {len(sampled_indices)} points from {n} original points.")

    return sampled_indices

def pattern_aware_sampling_bk(df, delta=0.02, min_spacing=100, debug=True):
    n = len(df)
    if n == 0:
        return []
    
    df = df.reset_index(drop=True)
    sampled_indices = [0]  # 采样点索引，包含起始点
    last_selected = 0       # 记录上一个被选中的关键点索引

    for i in range(1, n):
        # 计算当前点到已采样点的拟合误差
        x1, y1 = last_selected, df.loc[last_selected, 'normalized_value']
        x2, y2 = i, df.loc[i, 'normalized_value']

        if x2 - x1 == 0:
            continue  # 避免除零错误
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # 计算该段的拟合误差
        max_error = 0
        max_error_idx = -1
        for j in range(last_selected + 1, i):
            actual_y = df.loc[j, 'normalized_value']
            predicted_y = slope * j + intercept
            error = abs(actual_y - predicted_y)
            
            if error > max_error:
                max_error = error
                max_error_idx = j

        # 误差超过 delta，或超出最小间隔
        if max_error > delta or (i - last_selected) >= min_spacing:
            sampled_indices.append(max_error_idx if max_error_idx != -1 else i)
            last_selected = sampled_indices[-1]

    # 确保最后一个点被保留
    if sampled_indices[-1] != n - 1:
        sampled_indices.append(n - 1)

    # if debug:
        # print(f"Final sampled indices: {sampled_indices}, Total points: {len(sampled_indices)}")
    
    return sampled_indices

def compute_importance(df, Kp=0.8, Ki=0.1, Kd=0.1, pi=5):
    """
    计算每个数据点的 PID 误差（importance）。
    df：至少包含 ['date','normalized_value']。
    返回：importance数组，与df行一一对应（index相同）。
    """
    df = df.copy()
    if 'date' not in df.columns or 'normalized_value' not in df.columns:
        raise ValueError("DataFrame 必须包含 'date' 和 'normalized_value' 列。")

    # 时间差(秒)
    df['time_diff'] = df['date'].diff().fillna(pd.Timedelta(seconds=1)).dt.total_seconds().astype(float)

    # 简易预测：前一个点值
    df['predicted'] = df['normalized_value'].shift(1).fillna(method='bfill')
    # F = 当前实际与预测之差
    df['F'] = (df['normalized_value'] - df['predicted']).abs()

    # 分别计算 P、I、D 三项
    F_arr = df['F'].values
    time_diff_arr = df['time_diff'].values
    
    n = len(df)
    int_list = []
    der_list = []
    pro_list = []
    
    window = deque()
    sum_window = 0.0
    
    for i in range(n):
        f_i = F_arr[i]
        # 滚动窗口均值：只取最近 pi 个 F
        window.append(f_i)
        sum_window += f_i
        if len(window) > pi:
            sum_window -= window.popleft()
        i_val = (sum_window / len(window)) * Ki  # I 项
        
        # D 项
        if i == 0:
            d_val = 0.0
        else:
            f_diff = F_arr[i] - F_arr[i - 1]
            dt = time_diff_arr[i]
            if dt == 0:
                d_val = 0.0
            else:
                d_val = (f_diff / dt) * Kd
        
        # P 项
        p_val = f_i * Kp
        
        int_list.append(i_val)
        der_list.append(d_val)
        pro_list.append(p_val)
    
    df['integral']     = int_list
    df['derivative']   = der_list
    df['proportional'] = pro_list
    df['importance']   = df['proportional'] + df['integral'] + df['derivative']
    
    return df['importance'].values

def allocate_privacy_budget(df, total_budget, w=160, init_alpha=0.5, grad=0.05):
    """
    重要性自适应预算分配：
    对于每个采样点 i，首先计算当前滑动窗口内剩余预算 ε′ = total_budget - sum(ε[j])，j 从 i-w+1 到 i-1。
    然后利用论文中公式（11）：p = 1 - exp( - (α/γ[i] + β·γ[i]) )，其中 β = 1 - α，
    最后分配预算：ε[i] = p * ε′。
    
    同时，根据剩余预算动态调整权重 α：
      - 当剩余预算足够（ε′ > total_budget/2）时，α 下降（分配更多给重要点）；
      - 当剩余预算不足（ε′ < ε_w，取 ε_w = total_budget/4）时，α 上升；
      - 否则保持不变。
      
    返回：更新后包含 'privacy_budget' 列的 df，以及最后的 α 和 β 值。
    """
    df = df.copy()
    budgets = [0.0] * len(df)
    # 初始权重因子
    alpha = init_alpha
    beta = 1.0 - alpha
    epsilon_w = total_budget / 4.0  # 剩余预算下限（可根据需要调整）
    
    for i in range(len(df)):
        start_i = max(0, i - w + 1)
        used = sum(budgets[start_i:i])
        remaining = total_budget - used
        
        # 动态更新 α（权重因子）
        if remaining > total_budget / 2.0:
            alpha = max(0.1, alpha - grad)
        elif remaining < epsilon_w:
            alpha = min(0.9, alpha + grad)
        # β 保持 = 1 - α
        beta = 1.0 - alpha
        
        gamma_i = df.loc[i, 'importance']
        if gamma_i < 1e-15:
            p = 0.0
        else:
            # 根据论文公式（11）：p = 1 - exp( - (α/γ + β·γ) )
            p = 1 - math.exp(- (alpha / gamma_i + beta * gamma_i))
        allocated = remaining * p
        budgets[i] = allocated
        
    df['privacy_budget'] = budgets
    return df, alpha, beta


def importance_aware_randomization(df, theta=1.0, mu=0.1):
    """
    重要性自适应随机响应：
    根据论文公式（13）–（15），对于每个采样点：
      1) 计算误差界限 b[i] = ln(theta/γ[i] + mu)；
      2) 根据预算 ε[i]计算概率因子 q[i] = ε[i] / (2*(1 - exp(-ε[i]*b[i])))；
      3) 在区间 [v - b, v + b] 上按照概率密度函数
             Pr(v*|v) = q[i] * exp(-ε[i]*|v* - v|)
         进行采样（该分布的归一化保证了两边各占 1/2 的概率质量）。
    
    下面给出的实现利用逆变换采样：
      - 对于 x ∈ [v-b, v]（左侧），累积分布函数为
            F(x) = (q/ε)*(exp(-ε*(v - x)) - exp(-ε*b))
        解得 x = v - (1/ε)*ln((ε/q)*U + exp(-ε*b))，其中 U ~ Uniform(0, 0.5)；
      - 对于 x ∈ [v, v+b]（右侧），累积分布函数为
            F(x) = 0.5 + (q/ε)*(1 - exp(-ε*(x - v)))
        解得 x = v + (1/ε)*(-ln(1 - (ε/q)*(U - 0.5)))，其中 U ~ Uniform(0.5, 1)；
      - 最后确保输出在 [v-b, v+b] 内。
    
    返回：扰动后的值列表（顺序与 df 对应）。
    """
    df = df.copy()
    perturbed_values = []
    for i in range(len(df)):
        v = df.loc[i, 'normalized_value']
        eps = df.loc[i, 'privacy_budget']
        gamma_i = df.loc[i, 'importance']
        
        # 计算误差界限 b[i]
        # 为防止 gamma_i 过小导致分母问题
        if gamma_i < 1e-15:
            b = math.log(mu)
        else:
            b = math.log(theta / gamma_i + mu)
        
        # 若预算极小或 b 非正，则不扰动
        if eps < 1e-15 or b <= 0:
            perturbed_values.append(v)
            continue
        
        # 根据归一化要求，q 确定为：
        denom = 2 * (1 - math.exp(-eps * b))
        if abs(denom) < 1e-15:
            q = 1 / (2 * b)
        else:
            q = eps / denom
        
        U = random.random()
        if U < 0.5:
            # 左侧分支：x ∈ [v-b, v]
            # F(x) = (q/eps)*(exp(-eps*(v - x)) - exp(-eps*b)) = U
            term = (eps / q) * U + math.exp(-eps * b)
            x = v - (1 / eps) * math.log(term)
        else:
            # 右侧分支：x ∈ [v, v+b]
            # F(x) = 0.5 + (q/eps)*(1 - exp(-eps*(x - v))) = U
            term = 1 - (eps / q) * (U - 0.5)
            # 为防止 log(0)
            term = max(term, 1e-15)
            x = v + (1 / eps) * (-math.log(term))
        
        # 保证 x 在 [v-b, v+b] 内
        x = max(v - b, min(v + b, x))
        perturbed_values.append(x)
    return perturbed_values

def sample_exponential_interval(v, eps, lower, upper):
    """
    在区间 [lower, upper] 上对 PDF(x) = (1/Z)*exp(-eps*|x - v|) 做逆变换采样。
    """
    # 左侧长度 L, 右侧长度 R
    L = v - lower
    R = upper - v
    
    # 计算左侧面积
    A_left  = (1/eps) * (1 - math.exp(-eps * L))
    # 右侧面积
    A_right = (1/eps) * (1 - math.exp(-eps * R))
    # 总面积
    Z = A_left + A_right
    
    # 在 [0, Z] 间采样
    u = random.random() * Z
    
    if u <= A_left:
        # 落在左侧
        # A = (1/eps)*(1 - e^(-eps*x')) = u => e^(-eps*x') = 1 - eps*u => x' = ...
        # 但要注意这里 x' = L 范围内
        left_part = eps * u
        tmp = max(1e-15, 1 - left_part)
        x_prime = - (1/eps)*math.log(tmp)
        return v - x_prime
    else:
        # 落在右侧
        M2 = u - A_left
        left_part2 = eps * M2
        tmp2 = max(1e-15, 1 - left_part2)
        x_prime = - (1/eps)*math.log(tmp2)
        return v + x_prime
    
def run_experiment(file_path, 
                   output_dir=None,
                   w=160, 
                   total_budget=1.0, 
                   sample_fraction=1.0, 
                   delta=0.5,
                   DTW_MRE=True,
                   Kp=0.8, Ki=0.1, Kd=0.1, pi=5,
                   theta=1.0, mu=0.1):
    """
    参数说明与前一致：基于PatternLDP思路，对时间序列做采样+扰动+插值。
    返回: 结果dict, 包含 dtw_distance / mre 等
    """
    # ========== 1) 数据预处理 ========== 
    # sample_data, origin_data = patternldp_preprocess_HKHS_data(file_path, sample_fraction)
    sample_data, origin_data = data_utils.preprocess_data(file_path, sample_fraction)
    origin_length = len(origin_data)  # 用于后面插值
    
    # ========== 2) PLA采样 ==========
    sample_data = sample_data.reset_index(drop=True)
    idx_pla = pattern_aware_sampling(sample_data, delta=delta)
    sample_data = sample_data.loc[idx_pla].copy().reset_index(drop=True)

    # ========== 3) PID重要度 ==========
    imp_arr = compute_importance(sample_data, Kp=Kp, Ki=Ki, Kd=Kd, pi=pi)
    sample_data['importance'] = imp_arr

    # ========== 4) 分配预算 (w-event) ==========
    updated_data, alpha, beta = allocate_privacy_budget(sample_data, total_budget, w)

    # ========== 5) 指数随机化 ==========
    perturbed_vals = importance_aware_randomization(updated_data)
    updated_data['perturbed_value'] = perturbed_vals
    
    # ========== 6) 插值 ==========
    # 需要 'timestamp','perturbed_value'
    # 这里把 row index 当 timestamp
    updated_data['timestamp'] = updated_data.index
    interpolated_values = data_utils.pla_interpolation(updated_data[['timestamp','perturbed_value']], 
                                            origin_length)

    # ========== 7) 评价指标 (DTW, MRE) ==========
    dtw_distance = None
    mre = None
    if DTW_MRE:
        original_series = origin_data['normalized_value'].values
        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(data_utils.calculate_fdtw, original_series, interpolated_values)
            f2 = executor.submit(data_utils.calculate_mre, interpolated_values, original_series)
            dtw_distance = f1.result()
            mre = f2.result()

    # ========== 8) 返回结果 ==========
    return {
        'origin_data': origin_data,             # 原序列 (归一化后)
        'sampled_data': updated_data,           # 采样后 + 扰动结果
        'interpolated_values': interpolated_values,    # 重构序列
        'dtw_distance': dtw_distance,
        'mre': mre,
        'alpha': alpha,
        'beta': beta
    }

def run_single_experiment(x, eps, file_path):
    sample_data, origin_data = data_utils.preprocess_data(file_path, x)
    origin_length = len(origin_data)  # 用于后面插值
    start_time = datetime.datetime.now()
    # ========== 2) PLA采样 ==========
    sample_data = sample_data.reset_index(drop=True)
    idx_pla = pattern_aware_sampling(sample_data, delta=0.5)
    sample_data = sample_data.loc[idx_pla].copy().reset_index(drop=True)

    # ========== 3) PID重要度 ==========
    imp_arr = compute_importance(sample_data, Kp=0.8, Ki=0.1, Kd=0.1, pi=5)
    sample_data['importance'] = imp_arr

    # ========== 4) 分配预算 (w-event) ==========
    updated_data, alpha, beta = allocate_privacy_budget(sample_data, eps, 160)

    # ========== 5) 指数随机化 ==========
    perturbed_vals = importance_aware_randomization(updated_data, theta=1.0, mu=0.1)
    updated_data['perturbed_value'] = perturbed_vals
    end_time = datetime.datetime.now()    # 记录结束时间
    elapsed_time_seconds = (end_time - start_time).total_seconds()
    # ========== 6) 插值 ==========
    # 需要 'timestamp','perturbed_value'
    # 这里把 row index 当 timestamp
    updated_data['timestamp'] = updated_data.index
    interpolated_values = data_utils.pla_interpolation(updated_data[['timestamp','perturbed_value']], 
                                            origin_length)

    # ========== 7) 评价指标 (DTW, MRE) ==========
    dtw_distance = None
    mre = None
    if True:
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
        "runtime": elapsed_time_seconds
    }

if __name__ == "__main__":
    # file_path = 'data/HKHS.csv'  # 输入数据路径
    # file_path = 'data/heartrate.csv'  # 输入数据路径
    file_path = 'data/LD.csv'  # 输入数据路径
    output_dir = 'results'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    result = run_experiment(
        file_path, 
        output_dir, 
        sample_fraction=1.0, 
        total_budget=1.0, 
        w=160, 
        DTW_MRE=True
    )
    print(result['dtw_distance'], result['mre'])