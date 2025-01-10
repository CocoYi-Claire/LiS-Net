import pandas as pd
import os

def calculate_tv(csv_path, steering_column_name):
    """
    计算 Total Variation (TV)。

    参数:
    - csv_path (str): CSV文件路径
    - steering_column_name (str): 转向角度所在的列名

    返回:
    - total_variation (float): 计算出的Total Variation
    """

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查是否存在指定的转向列
    if steering_column_name not in df.columns:
        raise ValueError(f"Column '{steering_column_name}' not found in the CSV file.")

    # 提取转向列数据
    steering = df[steering_column_name]

    # 清洗数据（如果需要，可以跳过缺失值）
    steering = steering.dropna()  # 删除任何NaN值，避免计算时出现问题

    # 计算Total Variation (TV)
    total_variation = sum(abs(steering[i + 1] - steering[i]) for i in range(len(steering) - 1))

    return total_variation


csv_CNN ='CNN_result/deviation_all_sequences.csv'
tv_CNN = calculate_tv(csv_CNN, 'steering_cnn')
print(f"Total Variation for CNN: {tv_CNN}")

csv_NCP = 'NCP_result/deviation_all_sequences.csv'
tv_NCP = calculate_tv(csv_NCP,'steering_ncp')
print(f"Total Variation for NCP: {tv_NCP}")

csv_lis = 'NCP-CfC_result/deviation_all_sequences.csv'
tv_lis = calculate_tv(csv_lis, 'steering_ncp')
print(f"Total Variation for LiS-Net: {tv_lis}")

csv_lis = 'NCP-CfC_result/deviation_all_sequences.csv'
tv_gt = calculate_tv(csv_lis, 'steering_truth')
print(f"Total Variation for gt: {tv_gt}")
