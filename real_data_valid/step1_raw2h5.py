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

event_file_path = "/home/xky/HD2/20231206-16-23-27/1206/215/2023_12月_06_16_23_37.raw"
output_folder = "."
output_path = output_folder + os.sep + os.path.basename(event_file_path).replace('.raw', '.h5')

frame_interval = 50000  # 1 second in microseconds (1e6)

mv_iterator_trigger = EventsIterator(input_path=event_file_path, delta_t=1e4)  # 1e5 is the time unit in microseconds
mv_iterator = EventsIterator(input_path=event_file_path, delta_t=1e4)  # 1e6 is the time unit in microseconds
e_height, e_width = mv_iterator_trigger.get_size()  # camera resolution


if not os.path.exists(output_path):
    generate_hdf5(paths=event_file_path, output_folder=output_folder, preprocess="histo", delta_t=frame_interval, #height=360, width=640,
              start_ts=0, max_duration=None)

print('\nOriginal file \"{}" is of size: {:.3f}MB'.format(event_file_path, os.path.getsize(event_file_path)/1e6))
print('\nResult file \"{}" is of size: {:.3f}MB'.format(output_path, os.path.getsize(output_path)/1e6))

