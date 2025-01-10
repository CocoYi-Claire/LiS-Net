
import sys
# print(sys.path)
sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_core.event_io import EventsIterator
import metavision_sdk_core as mv
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, Window
from tqdm import tqdm
from metavision_ml.preprocessing.viz import filter_outliers
from metavision_ml.preprocessing.hdf5 import generate_hdf5

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from utils import make_dirs
import h5py

input_h5_path = "./2023_12月_06_16_23_37_640x360.h5"
csv_file_path = "/home/xky/HD2/20231206-16-23-27/output/offline-20240124-11-45-06/sample/vehicle-sample-v4.csv" #include steering monitor
output_h5_path = "./eval_2023_12月_06_16_23_37.h5"

frame_intervel = 50000
delta_t = 10000000 #between steering and images

f  = h5py.File(input_h5_path, 'r')  # open the HDF5 tensor file in read mode
print(f['data'])  # show the 'data' dataset

hdf5_shape = f['data'].shape
print(hdf5_shape)
print(f['data'].dtype)

#visualization

# for i, histogram in enumerate(f['data'][160:170], start=160):
#     # frame = np.full(histogram.shape[1:], 0.5, dtype=np.float32)
#     # frame[histogram[1] > histogram[0]] = 1.0
#     # frame[histogram[1] < histogram[0]] = 0.0
#     # frame = np.expand_dims(frame, axis=0)
#     plt.imshow(filter_outliers(histogram[0], 0.01))
#     # plt.imshow(frame)
#     # plt.imshow(filter_outliers(histogram[0], 3)) #filter out some noise
#     plt.title("{:s} feature (before downsample) computed at time {:d} μs".format(
#         f['data'].attrs['events_to_tensor'].decode(),
#         f['data'].attrs["delta_t"] * i
#     ))
#
#     plt.pause(0.01)

# 读取 CSV 文件并处理
df = pd.read_csv(csv_file_path)
df['Time'] = pd.to_numeric(df['Time'], errors='coerce') * 1e6  # 将时间转换为微秒
df['Steer angle[°]'] = pd.to_numeric(df['Steer angle[°]'], errors='coerce')
steering_angles = df['Steer angle[°]'].values
steering_amps = steering_angles / 900  # 归一化到 -1 到 1
timestamps = df['Time'].values

with h5py.File(input_h5_path, 'r') as input_h5, h5py.File(output_h5_path, 'w') as output_h5:
    data = input_h5['data']  # 原始数据
    dvs_pics_group = output_h5.create_group("dvs_pics")  # 创建组用于存储新的帧
    actions_group = output_h5.create_group("actions")  # 创建组用于存储动作

      # 微秒级间隔（可以根据需求调整）

    for idx, histogram in enumerate(data):
        # 创建初始 frame
        frame = np.full(histogram.shape[1:], 0.5, dtype=np.float32)

        # 对 histogram 进行滤波
        histogram[0] = filter_outliers(histogram[0], 3)
        histogram[1] = filter_outliers(histogram[1], 3)

        # 更新符合条件的像素值
        frame[histogram[1] > histogram[0]] = 1.0
        frame[histogram[1] < histogram[0]] = 0.0

        # 转换为 (1, h, w) 形状
        frame = np.expand_dims(frame, axis=0)

        # 保存 frame 到 dvs_pics 中
        dvs_pics_group.create_dataset(f"dvs_pic{idx}", data=frame, dtype=np.float32)

        # 计算对应的时间戳并找到最接近的 steering 值
        ts = idx * frame_intervel  # 假设每个帧的时间戳间隔为 delta_t
        closest_idx = np.argmin(np.abs(timestamps - delta_t- ts))
        steering_angle = steering_amps[closest_idx]

        # 保存 steering 到 actions 中
        actions_group.create_dataset(f"act{idx}", data=steering_angle, dtype=np.float32)

    print(f"Processed {len(data)} histograms and saved to {output_h5_path}")