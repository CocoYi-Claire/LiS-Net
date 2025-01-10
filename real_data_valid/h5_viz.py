import sys
# print(sys.path)
sys.path.append("/usr/lib/python3/dist-packages/")

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from utils import make_dirs
import h5py

h5_path = "./eval_2023_12月_06_16_23_37.h5"
output_folder = "./eval_gt"
start_idx = 1
end_idx = 2000


with h5py.File(h5_path, 'r') as h5_file:
    dvs_pics_group = h5_file['dvs_pics']  # 获取 dvs_pics 组
    actions_group = h5_file['actions']  # 获取 actions 组


    for idx in range(start_idx, end_idx):
        # 读取图像帧
        frame = dvs_pics_group[f'dvs_pic{idx}'][:]
        frame = np.squeeze(frame)  # 去掉维度为 1 的轴

        # 读取对应的 steering 值
        steering_angle = actions_group[f'act{idx}']

        # 如果数据集是标量，直接读取值
        if steering_angle.ndim == 0:
            steering_angle = steering_angle[()]
        else:
            steering_angle = steering_angle[:]

        # 可视化图像
        plt.imshow(frame, cmap='gray')  # 显示图像，灰度图
        plt.title(f"Frame {idx}, Steering Angle: {steering_angle:.2f}°")
        plt.colorbar()  # 可选：显示颜色条

        save_path = os.path.join(output_folder, f"frame_{idx}_steering_{steering_angle:.2f}.png")
        plt.savefig(save_path)  # 保存图片
        plt.close()  # 关闭当前的图形，释放内存

        plt.show()

        # 打印对应的 steering 值
        print(f"Frame {idx}: Steering Angle: {steering_angle:.2f}°")
