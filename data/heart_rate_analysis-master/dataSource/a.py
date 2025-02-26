import pandas as pd

# 指定要合并的文件列表（按顺序）
input_files = [
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-09.csv",
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-10.csv",
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-11.csv",
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-12.csv",
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-14.csv",
    "data/heart_rate_analysis-master/dataSource/heartrate_2017-01-16.csv"
]

# 定义输出文件路径
output_file = "data/heart_rate_analysis-master/dataSource/merged_heartrate.csv"

# 初始化一个空的 DataFrame 用于存储合并结果
merged_data = pd.DataFrame()

# 遍历每个文件进行读取和合并
for file in input_files:
    try:
        # 读取当前文件
        data = pd.read_csv(file)
        # 合并数据
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    except FileNotFoundError:
        print(f"文件未找到: {file}")
    except Exception as e:
        print(f"读取文件 {file} 时发生错误: {e}")

# 按日期排序（可选）
merged_data = merged_data.sort_values(by="date")

# 保存合并后的数据
merged_data.to_csv(output_file, index=False)

print(f"数据已按指定顺序合并并保存到 {output_file}")