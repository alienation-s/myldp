
# 高层次接口函数 compare_experiments
def compare_experiments(file_path, output_dir, target):
    sample_fraction = 1.0
    if target == "e":
        es = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        results = []
        for e in es:
            result_budget = process(file_path, output_dir, sample_fraction=sample_fraction, total_budget=e, w=160, delta=0.5, DTW_MRE=True)
            print(f"DTW for budget {e}: {result_budget['dtw_distance']}, MRE for budget {e}: {result_budget['mre']}")
            results.append(result_budget)
    elif target == "w":
        ws = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
        results = []
        for w in ws:
            result_window = process(file_path, output_dir, sample_fraction=sample_fraction, total_budget=1.0, w=w, delta=0.5, DTW_MRE=True)
            print(f"DTW for window size {w}: {result_window['dtw_distance']}, MRE for window size {w}: {result_window['mre']}")
            results.append(result_window)
