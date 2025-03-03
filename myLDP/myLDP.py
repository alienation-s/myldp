import numpy as np
def adaptive_w_event_budget_allocation(slopes, fluctuation_rates, total_budget, w, data_length):
    """
    论文方法：对所有数据点分配预算，而不仅是显著点
    """
    allocated = np.zeros(data_length)  # 初始化所有数据点预算
    window_sum = 0.0  # 滑动窗口预算总和

    for i in range(data_length):
        # 维护滑动窗口预算总和
        if i >= w:
            window_sum -= allocated[i - w]
        remaining = max(total_budget - window_sum, 0)  # 计算当前可分配预算
        
        # 计算 k 和 gamma
        k = slopes[i] if i < len(slopes) else 0  # 避免索引超界
        gamma = fluctuation_rates[i] if i < len(fluctuation_rates) else 0

        # 计算 pk
        if k >= 0:
            pk = 1 - np.exp(-k)
        else:
            pk = 1 - np.exp(k)

        # 计算 pγ
        pγ = 1 - np.exp(-gamma)

        # 计算 pkγ
        if k != 0:
            pkγ = 1 - np.exp(-1 / (abs(k) * gamma + 1e-8))
        else:
            pkγ = 1.0  # 处理 k=0 情况

        # 计算最终的 p
        p = 1 - np.exp(- (pk + pγ) / (pkγ + 1e-8))
        p = np.clip(p, 0, 1)  # 预算比例在 [0,1] 之间

        # 计算最终预算 εi
        allocated[i] = p * remaining
        window_sum += allocated[i]  # 更新滑动窗口总预算

    return allocated

def sw_perturbation_w_event(values, budgets, min_budget=0.01):
    """
    论文中的 SW 机制，对所有数据点添加扰动
    """
    epsilons = np.maximum(budgets, min_budget)  # 确保每个点至少有最小预算
    denominators = 2 * np.exp(epsilons) * (np.exp(epsilons) - 1 - epsilons)
    valid_mask = denominators > 1e-10  # 过滤有效值

    # 计算 b[i]，根据论文公式
    b = np.zeros_like(epsilons)
    b[valid_mask] = (epsilons[valid_mask] * np.exp(epsilons[valid_mask]) 
                    - np.exp(epsilons[valid_mask]) + 1) / denominators[valid_mask]

    # 计算扰动概率
    perturb_probs = np.exp(epsilons) / (2 * b * np.exp(epsilons) + 1)

    # 生成随机扰动
    rand = np.random.rand(len(values))
    perturb_mask = rand <= perturb_probs

    # 添加噪声
    laplace_noise = np.random.laplace(scale=b, size=len(values))
    perturbed = np.where(perturb_mask, values, values + laplace_noise)

    return perturbed

def kalman_filter(perturbed_values, process_variance=5e-4, measurement_variance=5e-3):
    """
    论文中的 Kalman 滤波器，优化扰动数据
    """
    n = len(perturbed_values)
    estimates = np.empty(n)
    variance = 1.0
    estimates[0] = perturbed_values[0]

    for t in range(1, n):
        # 预测步骤
        predicted_estimate = estimates[t-1]
        predicted_variance = variance + process_variance

        # 更新步骤
        kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
        estimates[t] = predicted_estimate + kalman_gain * (perturbed_values[t] - predicted_estimate)
        variance = (1 - kalman_gain) * predicted_variance  # 更简洁的更新

    return estimates