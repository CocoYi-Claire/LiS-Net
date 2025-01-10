"""Train all the models.

Use CARLA/LGSVL Data to train NCP, CTGRU, LSTM, RNN, CNN.
"""

import sys
import argparse
import time
import numpy as np
import torch
from torchvision import transforms
import h5py
from wiring import NCPWiring
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.ltc_cell import LTCCell
from nets.cfc_cell import CfCCell,WiredCfCCell
from nets.models_all import Convolution_Model, GRU_Model, LSTM_Model, \
    CTGRU_Model, NCP_Model, NCPCfC_Model, AttCLF_CfC_Model, NCPCfC_Model_smooth
from dataset_all import MarkovProcessRNNHdf5, MarkovProcessCNNHdf5
from early_stopping import EarlyStopping
from utils import make_dirs, save_result, __crop, epoch_policy, epoch_policy_CLF, \
    evaluate_on_single_sequence, evaluate_on_single_sequence_CLF, epoch_policy_smooth, evaluate_on_single_sequence_smooth

start_time = time.time()
parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)

# mode ------------------------------------------------------------------

parser.add_argument("--network", type=str, default='NCP-CfC_smooth')
# CNN, GRU, LSTM, CTGRU, NCP, NCP-CfC, CLF-NCP-CfC, NCP-CfC_smooth
parser.add_argument("--name", type=str, default='NCP-CfC_smooth_Training_Result')
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--sequence", type=int, default=16)  # only for RNN
parser.add_argument("--hidden", type=int, default=64)   # hidden units
# 64 for throttle, brake, steering, 16 for only steering.
parser.add_argument("--output", type=int, default=1)   # output predictions
# 3 for throttle, brake, steering, 1 for only steering.
parser.add_argument("--seed", type=int, default=100)  # default seed,
parser.add_argument("--hdf5_train", type=str, default="/home/xky/HD2/EventScape/train_hd5_try/All_Sequence",
                    help="Path to the training HDF5 data")
parser.add_argument("--hdf5_valid", type=str, default="/home/xky/HD2/EventScape/valid_hd5_try/All_Sequence",
                    help="Path to the validation HDF5 data")
parser.add_argument("--smooth_weight", type=float, default=1,
                    help="If choose model with smooth, this param determines smooth loss weight in the total loss")
parser.add_argument("--clf_weight", type=float, default=0.3,
                    help="If choose model with CLF, this param determines clf weight in the loss")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="Choose which gpu to use")


start_time = time.time()
# Parse arguments -------------------------------------------------------
args = parser.parse_args()


HDF5_TRAIN = args.hdf5_train
HDF5_VALID = args.hdf5_valid

SAVE_DIR = "./result/" + args.name + "/"


BATCH_SIZE = args.batch
seq_length = args.sequence
EARLY_LENGTH = 35     # change from 5 to 45
N_EPOCH = args.epoch  # for debug, in order to accelerate
METHOD = args.name
Network = args.network
HIDDEN_UNITS = args.hidden
OUTPUT = args.output
seed_value = args.seed
torch.manual_seed(seed_value)  # for CPU
torch.cuda.manual_seed(seed_value)  # for GPU
np.random.seed(seed_value)


# s_dim, a_dim of the pic.
s_dim = (1, 66, 200)
a_dim = args.output

print(f"Configuration parsed in {time.time() - start_time:.2f} seconds.")
start_time = time.time()

hf5_train = h5py.File(HDF5_TRAIN, 'r')  # read the hdf5 group of train data
hf5_valid = h5py.File(HDF5_VALID, 'r')  # read the hdf5 group of valid data
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

# prepare save directories
start_time = time.time()
SDIR = SAVE_DIR + "/"
make_dirs(SDIR)
print(f"Directories created in {time.time() - start_time:.2f} seconds.")

 # for CARLA
start_time = time.time()
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Lambda(lambda img: __crop(img, (10, 80), (500, 176))),
     transforms.Resize((66, 200)),
     transforms.ToTensor()]
)
print(f"Data transformation defined in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
if Network == 'CNN':
    dataset_train = MarkovProcessCNNHdf5(hf5_train,
                                         output=OUTPUT,
                                         transform=transform,
                                         mode='train')
    dataset_valid = MarkovProcessCNNHdf5(hf5_valid,
                                         output=OUTPUT,
                                         transform=transform,
                                         mode='eval')

elif Network in ["GRU", "LSTM", "CTGRU", "NCP", "NCP-CfC","CLF-NCP-CfC", "NCP-CfC_smooth"]:
    dataset_train = MarkovProcessRNNHdf5(hf5_train,
                                         output=OUTPUT,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='train')
    dataset_valid = MarkovProcessRNNHdf5(hf5_valid,
                                         output=OUTPUT,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='eval')
else:
    print('unknown network type: ', Network)
    sys.exit()
print(f"Dataset initialized in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=16,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=16,
                                           pin_memory=True)
print(f"DataLoader initialized in {time.time() - start_time:.2f} seconds.")


POLICY = None
start_time = time.time()
if Network == 'CNN':
    POLICY = Convolution_Model(s_dim, a_dim)   # initialize the CNN model
elif Network == 'GRU':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = GRU_Model(cnn_head,
                       time_step=seq_length,
                       input_size=32*4,
                       hidden_size=HIDDEN_UNITS,
                       output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands

elif Network == 'LSTM':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = LSTM_Model(cnn_head,
                        time_step=seq_length,
                        input_size=32*4,
                        hidden_size=HIDDEN_UNITS,
                        output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands

elif Network == 'CTGRU':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = CTGRU_Model(num_units=HIDDEN_UNITS,
                         conv_head=cnn_head,
                         output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands
    # time interval:0.04s for training, 0.2s for test.
    # in line 421 of models_all.py

elif Network == 'NCP':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)
    # This is for predicting all the actions.
    if OUTPUT == 3:   # predict steering, throttle, brake
        wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                           motor_neurons=3, sensory_fanout=48,
                           inter_fanout=24, recurrent_command=24,
                           motor_fanin=16)
    else:  # predict only steering
        wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                           motor_neurons=1, sensory_fanout=16,
                           inter_fanout=8, recurrent_command=8,
                           motor_fanin=6)
    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    ltc_cell = LTCCell(wiring=wiring, time_interval=0.04)
    POLICY = NCP_Model(ltc_cell=ltc_cell, conv_head=cnn_head)

elif Network == 'NCP-CfC':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)
    # This is for predicting all the actions.
    if OUTPUT == 3:   # predict steering, throttle, brake
        wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                           motor_neurons=3, sensory_fanout=48,
                           inter_fanout=24, recurrent_command=24,
                           motor_fanin=16)
    else:  # predict only steering
        wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                           motor_neurons=1, sensory_fanout=16,
                           inter_fanout=8, recurrent_command=8,
                           motor_fanin=6)
        # wiring = AutoNCP(28,1)
    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    cfc_cell = WiredCfCCell(input_size=input_shape,wiring=wiring)
    POLICY = NCPCfC_Model(ltc_cell=cfc_cell, conv_head=cnn_head)

elif Network == 'NCP-CfC_smooth':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)
    # This is for predicting all the actions.
    if OUTPUT == 3:  # predict steering, throttle, brake
        wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                           motor_neurons=3, sensory_fanout=48,
                           inter_fanout=24, recurrent_command=24,
                           motor_fanin=16)
    else:  # predict only steering
        wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                           motor_neurons=1, sensory_fanout=16,
                           inter_fanout=8, recurrent_command=8,
                           motor_fanin=6)
        # wiring = AutoNCP(28,1)
    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    cfc_cell = WiredCfCCell(input_size=input_shape, wiring=wiring)
    POLICY = NCPCfC_Model_smooth(ltc_cell=cfc_cell, conv_head=cnn_head, smooth_weight= args.smooth_weight)

elif Network == 'CLF-NCP-CfC':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)
    # This is for predicting all the actions.
    if OUTPUT == 3:   # predict steering, throttle, brake
        wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                           motor_neurons=3, sensory_fanout=48,
                           inter_fanout=24, recurrent_command=24,
                           motor_fanin=16)
    else:  # predict only steering
        wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                           motor_neurons=1, sensory_fanout=16,
                           inter_fanout=8, recurrent_command=8,
                           motor_fanin=6)
        # wiring = AutoNCP(28,1)
    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    cfc_cell = WiredCfCCell(input_size=input_shape,wiring=wiring)
    POLICY = AttCLF_CfC_Model(ltc_cell=cfc_cell, conv_head=cnn_head, clf_weight=args.clf_weight)

print(f"Model initialized in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
if str(POLICY.device) == "cuda":    # feed models to GPU
    print("there is GPU")
    POLICY.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    POLICY.to(POLICY.device)
    print(f"Model moved to {POLICY.device} in {time.time() - start_time:.2f} seconds.")
else:
    print("No GPU available, using CPU.")




print(POLICY)
print(f"the total params of Network {POLICY.count_params()}")
# print(f"the params of cnn head {POLICY.conv_head.count_params()}")

stopper = EarlyStopping(length=EARLY_LENGTH)  # to avoid over fitting,

# prepare buffers to store results
train_loss_policy_o = []
valid_loss_policy_o = []
if Network == 'CLF-NCP-CfC' or Network == 'NCP-CfC_smooth':
    train_mse_loss_policy_o = []
    valid_mse_loss_policy_o = []

print("Start learning policy!")
for n_epoch in range(1, N_EPOCH + 1):

    if Network == 'CLF-NCP-CfC':
        l_origin, l_mse = epoch_policy_CLF(train_loader, POLICY, n_epoch, "train", Network)
        train_mse_loss_policy_o.append(l_mse)
    elif Network == 'NCP-CfC_smooth':
        l_origin, l_mse = epoch_policy_smooth(train_loader, POLICY, n_epoch, "train", Network)
        train_mse_loss_policy_o.append(l_mse)
    else:
        l_origin = epoch_policy(train_loader, POLICY, n_epoch, "train", Network)
    train_loss_policy_o.append(l_origin)
    with torch.no_grad():
        if Network == 'CNN':
            valid_loss_policy_o.append(epoch_policy(
                valid_loader,
                POLICY,
                n_epoch,
                "valid",
                Network)
            )
        elif Network == 'CLF-NCP-CfC':
            l_origin, l_mse = evaluate_on_single_sequence_CLF(
                dataset_valid.sequence_number,
                dataset_valid,
                POLICY,
                n_epoch,
                "valid")
            valid_loss_policy_o.append(l_origin)
            valid_mse_loss_policy_o.append(l_mse)
        elif Network == 'NCP-CfC_smooth':
            l_origin, l_mse = evaluate_on_single_sequence_smooth(
                dataset_valid.sequence_number,
                dataset_valid,
                POLICY,
                n_epoch,
                "valid")
            valid_loss_policy_o.append(l_origin)
            valid_mse_loss_policy_o.append(l_mse)

        else:
            l_origin = evaluate_on_single_sequence(
                dataset_valid.sequence_number,
                dataset_valid,
                POLICY,
                n_epoch,
                "valid")
            valid_loss_policy_o.append(l_origin)

    # policy.scheduler.step()  # update the lr rate after every set epoch
    # early stopping
    if stopper(valid_loss_policy_o[-1]):  # origin loss
        print("Early Stopping to avoid Overfitting!")
        break

# save trained model
POLICY.release(SDIR)

# close the hdf5 file.
hf5_train.close()
hf5_valid.close()

# csv and plot
if Network == 'CLF-NCP-CfC' or Network == 'NCP-CfC_smooth': # CLF加入后总LOSS不再是MSE,因此单独保存MSE的结果
    save_result(SDIR,
                "mse_loss_policy",
                {"train": train_mse_loss_policy_o, "valid": valid_mse_loss_policy_o}
                )
save_result(SDIR,
            "loss_policy_origin",
            {"train": train_loss_policy_o, "valid": valid_loss_policy_o}
            )

# output the parameters and settings of the NN，
# save the params, varies from network to network:
# the structure of the network, fetch it before sending to GPU
dict_layer = POLICY.nn_structure()
if Network == 'CNN':
    dict_params = {'batch_size': BATCH_SIZE,
                   'total_params of Network': POLICY.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'GRU':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'LSTM':
    dict_params = {'batch_size': BATCH_SIZE,
                   'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'CTGRU':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'NCP':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_wiring = wiring.get_config()  # ltc_cell._wiring.get_config()
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params,
                  'NCP Wiring': dict_wiring}

elif Network == 'NCP-CfC':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_wiring = wiring.get_config()  # ltc_cell._wiring.get_config()
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params,
                  'NCP Wiring': dict_wiring}

elif Network == 'NCP-CfC_smooth':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_wiring = wiring.get_config()  # ltc_cell._wiring.get_config()
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params,
                  'NCP Wiring': dict_wiring}

elif Network == 'CLF-NCP-CfC':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_wiring = wiring.get_config()  # ltc_cell._wiring.get_config()
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params,
                  'NCP Wiring': dict_wiring}

path = SDIR + "/network_settings.pth"
torch.save(dict_whole, path)

# calculate the total execution time of the code
end_time = time.time()
execution_time = end_time - start_time
hours = execution_time//3600
mins = (execution_time % 3600) // 60
seconds = (execution_time % 3600) % 60
print(f"The execution time is {hours}h {mins}m {seconds}s")

# output the time duration of the training.
F = SDIR + "/summary.txt"  # output the information of transformation
SUMM = " Totally" + str(n_epoch) + "training are done, it takes " \
       + str(hours) + "h " + str(mins) + "min " + str(seconds) + "s"
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(SUMM)
