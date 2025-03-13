import numpy as np
import matplotlib.pyplot as plt

# 定义采样率区间，例如从0.1到1.0
r = np.linspace(0.1, 1, 100)

# 设定参数：C_s和σ_LDP²
C_s = 1.0
sigma_LDP_sq = 0.1

# 计算总噪声方差
noise_variance = C_s / r + sigma_LDP_sq

# 计算成本，这里假设成本正比于采样率 r
cost = r

# 绘制 Pareto 前沿图：横坐标为计算成本（采样率），纵坐标为总噪声方差
plt.figure(figsize=(8, 6))
plt.plot(cost, noise_variance, label='Pareto Frontier', linewidth=2)
plt.xlabel('Computational Cost (Sampling Rate)', fontsize=12)
plt.ylabel('Total Noise Variance', fontsize=12)
plt.title('Privacy-Utility Pareto Frontier', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()