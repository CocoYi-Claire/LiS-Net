import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import make_dirs, __crop
from nets.models_all import Convolution_Model, LSTM_Model, \
    CTGRU_Model, GRU_Model, NCP_Model, NCPCfC_Model
from nets.cfc_cell import CfCCell,WiredCfCCell
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.ltc_cell import LTCCell
from wiring import NCPWiring
import h5py


MODEL_NAME = "NCP-CfC-smooth"
TESTDATA = "./eval_2023_12月_06_16_23_37.h5"
MODEL_DIR = "/home/xky/Event2Steering/End-to-End-learning-for-Autonomous-Driving-main/result/NCP-CfC_smooth_Training_Result/SmoothWeight_1/"
R_DIR = "./NCP-CfC-smooth_result_adjust/"

if not os.path.exists(R_DIR):
    os.makedirs(R_DIR)
    print(f"Directory '{R_DIR}' created successfully.")

# 创建保存图像的文件夹，如果不存在
cropped_image_dir = os.path.join(R_DIR, "cropped_image")
if not os.path.exists(cropped_image_dir):
    os.makedirs(cropped_image_dir)
    print(f"Directory '{cropped_image_dir}' created successfully.")

S_DIM = (1, 66, 200)
A_DIM = 1
SEQ_LENGTH = 1  # for test
transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Lambda(lambda img:__crop(img, (30, 100), (400, 250))), # original (10, 80), (500, 176)
         transforms.Resize((66, 200)),
         transforms.ToTensor()]
    )

if MODEL_NAME == "NCP-CfC-smooth" or MODEL_NAME == "NCP-CfC":
    cnn_head = ConvolutionHead_Nvidia(S_DIM, SEQ_LENGTH,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)

    wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                       motor_neurons=1, sensory_fanout=16,
                       inter_fanout=8, recurrent_command=8,
                       motor_fanin=6)

    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    cfc_cell = WiredCfCCell(input_size=input_shape, wiring=wiring)
    model = NCPCfC_Model(ltc_cell=cfc_cell, conv_head=cnn_head)

if MODEL_NAME == "NCP":
    conv_head_ncp = ConvolutionHead_Nvidia(S_DIM,
                                           SEQ_LENGTH,
                                           num_filters=32,
                                           features_per_filter=4)

    input_shape = (1, conv_head_ncp.num_filters * conv_head_ncp.features_per_filter)
    wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                       motor_neurons=1, sensory_fanout=16,
                       inter_fanout=8, recurrent_command=8, motor_fanin=6)
    wiring.build(input_shape)
    ltc_cell = LTCCell(wiring=wiring, time_interval=0.04)

    model = NCP_Model(ltc_cell=ltc_cell, conv_head=conv_head_ncp)

if MODEL_NAME == "CNN":
    model = Convolution_Model(S_DIM, A_DIM)

# load params
model.load(MODEL_DIR)
model.eval()

if MODEL_NAME == "NCP-CfC-smooth" or MODEL_NAME == "NCP-CfC" or MODEL_NAME == "NCP" :
    hidden_state_ncp = None
    sequence_data = []
    image_indices = []
    s_ncp, s_truth = [], []

    with h5py.File(TESTDATA, 'r') as h5_file:
        dvs_pics_group = h5_file['dvs_pics']  # 获取 dvs_pics 组
        actions_group = h5_file['actions']  # 获取 actions 组

        # 获取所有的 'dvs_pic' 和 'act' 键
        dvs_pic_keys = list(dvs_pics_group.keys())  # 获取所有图像帧的键
        action_keys = list(actions_group.keys())  # 获取所有 steer angle 的键

        assert len(dvs_pic_keys) == len(action_keys), "Mismatch between the number of dvs_pics and actions"

        for idx in range(len(dvs_pic_keys)):
            # 读取图像帧
            frame = dvs_pics_group["dvs_pic" + str(idx)][:]
            frame = np.squeeze(frame)  # 去掉维度为 1 的轴


            states = transform(frame)
            sequence_data.append(states)

            plt.imshow(np.squeeze(states), cmap='gray')  # 使用灰度图显示
            plt.title("Frame Visualization")  # 可选：设置标题
            plt.colorbar()  # 可选：添加颜色条
            output_image_path = os.path.join(cropped_image_dir, f"frame_{idx}.png")
            plt.savefig(output_image_path)  # 保存图像
            plt.close()  # 关闭当前图像以避免内存溢出

            if len(sequence_data) == SEQ_LENGTH:
                states_rnn = torch.stack(sequence_data)  # Shape: (SEQ_LENGTH, C, H, W)
                states_rnn = states_rnn.unsqueeze(0)  # Add batch dimension: (1, SEQ_LENGTH, C, H, W)

                actions_ncp, hidden_state_ncp = model.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_ncp
                )
                s_ncp.append(float(actions_ncp[0][0][0])-0.1)

                steering_angle = actions_group["act" + str(idx)]
                if steering_angle.ndim == 0:
                    steering_angle = steering_angle[()]
                else:
                    steering_angle = steering_angle[:]
                steering_angle = float(steering_angle)
                s_truth.append(steering_angle)
                image_indices.append(idx)

                sequence_data.pop(0)

    assert len(s_truth) == len(s_ncp), "Check number of steering commands"


if MODEL_NAME == "CNN":
    image_indices = []
    s_cnn, s_truth = [], []
    with h5py.File(TESTDATA, 'r') as h5_file:
        dvs_pics_group = h5_file['dvs_pics']  # 获取 dvs_pics 组
        actions_group = h5_file['actions']  # 获取 actions 组

        # 获取所有的 'dvs_pic' 和 'act' 键
        dvs_pic_keys = list(dvs_pics_group.keys())  # 获取所有图像帧的键
        action_keys = list(actions_group.keys())  # 获取所有 steer angle 的键

        assert len(dvs_pic_keys) == len(action_keys), "Mismatch between the number of dvs_pics and actions"

        for idx in range(len(dvs_pic_keys)):
            # 读取图像帧
            frame = dvs_pics_group["dvs_pic" + str(idx)][:]
            frame = np.squeeze(frame)  # 去掉维度为 1 的轴
            states = transform(frame)

            predicted_action = model(states)
            s_cnn.append(float(predicted_action[0][0]))

            steering_angle = actions_group["act" + str(idx)]
            if steering_angle.ndim == 0:
                steering_angle = steering_angle[()]
            else:
                steering_angle = steering_angle[:]
            steering_angle = float(steering_angle)
            s_truth.append(steering_angle)
            image_indices.append(idx)

n = 1  # 每隔n个点绘制一次

# 绘制总图
plt.figure(figsize=(16, 8))
# plt.plot(s_truth, label="Ground Truth", color="blue", linewidth=1)
plt.plot(image_indices[::n], s_truth[::n], label="Ground Truth", color="blue", linewidth=1)
if MODEL_NAME == "NCP-CfC-smooth" or MODEL_NAME == "NCP-CfC" or MODEL_NAME == "NCP" :
    plt.plot(image_indices[::n], s_ncp[::n], label=MODEL_NAME, color="red", linewidth=1)
if MODEL_NAME == "CNN" :
    plt.plot(image_indices[::n], s_cnn[::n], label=MODEL_NAME, color="red", linewidth=1)
plt.xlabel("Image Index", fontsize=12)
plt.ylabel("Steering Angle", fontsize=12)
plt.title("Sequential Steering Plot for All Sequences", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(ls="--")
plt.savefig(R_DIR + "steering_all_sequences.pdf")
plt.show()

if MODEL_NAME == "NCP-CfC-smooth" or MODEL_NAME == "NCP-CfC" or MODEL_NAME == "NCP" :
    dev_ncp_s = np.absolute(np.array(s_ncp) - np.array(s_truth)).tolist()

    # 创建一个包含 s_ncp, s_truth 和偏差的字典
    DEVIATION = {
        'steering_ncp': s_ncp,  # 模型预测值
        'steering_truth': s_truth,  # 真实值
        'deviation_of_steering': dev_ncp_s  # 偏差
    }
    df_deviation = pd.DataFrame(DEVIATION)

    output_file = R_DIR + "/deviation_all_sequences.csv"  # 确保路径正确
    df_deviation.to_csv(output_file, index=False)

    print(f"Deviation data saved to {output_file}")

if MODEL_NAME == "CNN":
    dev_cnn_s = np.absolute(np.array(s_cnn) - np.array(s_truth)).tolist()

    # 创建一个包含 s_ncp, s_truth 和偏差的字典
    DEVIATION = {
        'steering_cnn': s_cnn,  # 模型预测值
        'steering_truth': s_truth,  # 真实值
        'deviation_of_steering': dev_cnn_s  # 偏差
    }
    df_deviation = pd.DataFrame(DEVIATION)

    output_file = R_DIR + "/deviation_all_sequences.csv"  # 确保路径正确
    df_deviation.to_csv(output_file, index=False)

    print(f"Deviation data saved to {output_file}")