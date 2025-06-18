# plot_compare_stages_distinguish.py

import os
import matplotlib.pyplot as plt
import numpy as np

from PPLDP.ppldp import (
    remarkable_point_sampling,
    adaptive_w_event_budget_allocation,
    sw_perturbation,
    kalman_filter,
)
import utils.data_utils as data_utils

# —— 全局绘图风格，同 window_plot.py —— 
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman'
})

# 采样率对应的线型、marker 和颜色
line_styles = {0.8: '--', 1.0: '-'}
markers    = {0.8: 'o', 1.0: 's'}
colors     = {0.8: 'C0', 1.0: 'C1'}

def get_series(x, eps, file_path, w, sample_method="uniform"):
    sample_df, origin_df = data_utils.preprocess_data(file_path, x, sample_method)
    origin_dates = origin_df["date"].values
    origin_vals  = origin_df["normalized_value"].values

    sample_dates = sample_df["date"].values
    sample_vals  = sample_df["normalized_value"].values

    # 可选显著点采样
    sig_idx = remarkable_point_sampling(sample_vals, kp=0.8, ks=0.1, kd=0.1)
    slopes = np.gradient(sample_vals[sig_idx])
    fluct  = np.abs(slopes) + 1e-8

    budgets = adaptive_w_event_budget_allocation(
        slopes, fluct, total_budget=eps, w=w, data_length=len(sample_vals)
    )
    perturbed = sw_perturbation(sample_vals, budgets)
    smooth    = kalman_filter(perturbed, process_variance=5e-4, measurement_variance=5e-3)

    sample_df = sample_df.copy()
    sample_df["smoothed_value"] = smooth
    interp_df = data_utils.interpolate_missing_points(origin_df, sample_df)

    return {
        'raw':       (origin_dates, origin_vals),
        'perturbed': (sample_dates, perturbed),
        'interp':    (interp_df["date"].values, interp_df["smoothed_value"].values),
    }

def plot_compare(file_path="data/LD.csv", eps=1.0, w=160, rates=(0.8, 1.0)):
    data_dict = {r: get_series(r, eps, file_path, w) for r in rates}
    out_dir = "figures/compare_stages"
    os.makedirs(out_dir, exist_ok=True)

    for stage in ['raw', 'perturbed', 'interp']:
        plt.figure(figsize=(8, 5))
        for r in rates:
            dates, vals = data_dict[r][stage]
            # 如果两条曲线重叠，区分：100%采样用实心marker，80%用空心
            if r == 1.0:
                mfc = colors[r]       # marker face color 填充
            else:
                mfc = 'none'          # 空心
            plt.plot(
                dates, vals,
                linestyle=line_styles[r],
                marker=markers[r],
                color=colors[r],
                markerfacecolor=mfc,
                markeredgewidth=1.2,
                markersize=6,
                linewidth=1.2,
                label=f"{int(r*100)}% sampling"
            )
        plt.xlabel("Time")
        if stage == 'raw':
            plt.ylabel("Original Value")
            plt.title("Raw Signal Comparison")
        elif stage == 'perturbed':
            plt.ylabel("After Sampling + Perturbation")
            plt.title("Sampled + Perturbed Signal Comparison")
        else:
            plt.ylabel("After Interpolation + Kalman")
            plt.title("Interpolated + Smoothed Signal Comparison")

        plt.grid(linestyle='--', alpha=0.4)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stage}_comparison_distinguish.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    plot_compare(
        file_path="data/HKHS.csv",
        eps=1.0,
        w=160,
        rates=(0.8, 1.0)
    )