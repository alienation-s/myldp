import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob
import os

# 设置全局绘图风格
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

# 定义三种方法对应的颜色、标签和 Marker
method_styles = {
    'LBD':        {'color': 'blue',   'label': 'LBD',        'marker': 'o'},
    'PPLDP':      {'color': 'red',    'label': 'PPLDP',      'marker': 's'},
    'PatternLDP': {'color': 'green',  'label': 'PatternLDP', 'marker': '^'},
}

# 定义采样率对应的线型
line_styles = {
    0.8: '--',
    1.0: '-',
}

result_files = glob.glob("results/budget/data/*_budget_results.csv")
for file_path in result_files:
    df = pd.read_csv(file_path)
    dataset  = os.path.basename(file_path).replace("_budget_results.csv", "")
    methods  = df['method'].unique()
    rates    = sorted(df['sampling_rate'].unique())
    epsilons = sorted(df['epsilon'].unique())

    # --------- DTW vs ε ---------
    plt.figure(figsize=(8, 5))
    for method in methods:
        style = method_styles[method]
        for r in rates:
            sub = df[(df['method'] == method) & (df['sampling_rate'] == r)]
            if sub.empty:
                continue
            plt.plot(
                sub['epsilon'], sub['dtw'],
                color=style['color'],
                marker=style['marker'], markerfacecolor='none', ms=5.5,
                linestyle=line_styles[r],
                label=f"{style['label']}  r={r}"
            )
    # 科学计数法格式化 y 轴
    ax = plt.gca()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    plt.xlabel("Privacy Budget")
    plt.ylabel("DTW")
    plt.xticks(epsilons)
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(
        f"results/budget/plot/{dataset}_DTW_budget.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # --------- MRE vs ε ---------
    plt.figure(figsize=(8, 5))
    for method in methods:
        style = method_styles[method]
        for r in rates:
            sub = df[(df['method'] == method) & (df['sampling_rate'] == r)]
            if sub.empty:
                continue
            plt.plot(
                sub['epsilon'], sub['mre'],
                color=style['color'],
                marker=style['marker'], markerfacecolor='none', ms=5.5,
                linestyle=line_styles[r],
                label=f"{style['label']}  r={r}"
            )
    plt.xlabel("Privacy Budget")
    plt.ylabel("MRE")
    plt.xticks(epsilons)
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(
        f"results/budget/plot/{dataset}_MRE_budget.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()