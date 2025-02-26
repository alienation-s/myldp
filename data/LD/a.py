import pandas as pd

# 定义文件路径
input_file = "data/LD2011_2014.csv"  # 原始数据文件
output_file = "data/converted_dataset.csv"  # 输出的 CSV 文件

# # 读取数据，注意分隔符为 ';'，小数点为 ','
# data = pd.read_csv(input_file, sep=";", decimal=",", quotechar='"')

# # 去除列名中的多余引号（如果有）
# data.columns = data.columns.str.strip('"')

# # 转换日期列为标准格式（可选）
# data['date'] = pd.to_datetime(data['date'])

# # 将数据保存为标准 CSV 格式，使用逗号分隔，小数点为 '.'
# data.to_csv(output_file, index=False, sep=",", float_format="%.3f")

# print(f"数据已成功保存为 {output_file}")

# 指定要保留的列
columns_to_keep = ["date", "MT_158"]  # 修改为你需要的列名

# 读取 CSV 文件
data = pd.read_csv(input_file)

# 检查是否所有指定列都存在
missing_columns = [col for col in columns_to_keep if col not in data.columns]
if missing_columns:
    raise ValueError(f"以下列在数据中不存在：{missing_columns}")

# 保留指定的列
filtered_data = data[columns_to_keep]

# 保存为新的 CSV 文件
filtered_data.to_csv(output_file, index=False)

print(f"已成功保存过滤后的数据到 {output_file}")